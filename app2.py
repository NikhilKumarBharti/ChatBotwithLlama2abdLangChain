import streamlit as st
from streamlit_chat import message
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

DB_FAISS_PATH = 'vectorstore/db_faiss'

# Define the system prompt
system_prompt = """You are an airline booking/review website. You will be given a context to answer from. Be precise in your answers wherever possible. In case you are sure you don't know the answer then you say that based on the context you don't know the answer. In all other instances you provide an answer to the best of your capability. Cite urls when you can access them related to the context. If any information is mentioned as None or Not Available or Not Provided by the user, mention it as such."""

# Loading the model
def load_llm(api_key):
    # Load the model using the API key
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        load_in_8bit=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    llm = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        api_key=api_key,
        max_new_tokens=512,
        temperature=0.5,
        trust_remote_code=True,
    )
    return llm

st.title("Chat with CSV using Llama2 🦙🦜")

uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")
api_key = st.sidebar.text_input("Enter your API key", type="password")

if uploaded_file and api_key:
    # Use tempfile because CSVLoader only accepts a file_path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
        'delimiter': ','})
    data = loader.load()

    # st.json(data)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'})
    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)

    llm = load_llm(api_key)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever(), conditioned_prompt=system_prompt, return_source_documents=True)

    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"], result["source_documents"]))
        return result["answer"]

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]

    # Container for the chat history
    response_container = st.container()

    # Container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk to your csv data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
else:
    st.warning("Please upload a CSV file and enter your API key.")