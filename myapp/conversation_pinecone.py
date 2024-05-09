# myapp/conversation.py
# TODO: replace Chroma implementation with Pinecone
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

def initialize_session_state():
    """
    Initialize session state variables if not already initialized.
    """
    if "history" not in st.session_state:
        st.session_state["history"] = []

    if "generated" not in st.session_state:
        st.session_state["generated"] = ["Hi! "]

    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey! 👋"]

    # Check if the conversation log file exists, if not create it
    if not os.path.exists("conversation_log.txt"):
        with open("conversation_log.txt", "w"):
            pass

def load_model():
    """
    Load the necessary components for the conversational chain.

    Returns:
        tuple: Vector store and model name.
    """
    # Load documents from a specific directory instead of user upload
    document_directory = "./myapp/doc/"
    text = []
    for filename in os.listdir(document_directory):
        file_path = os.path.join(document_directory, filename)
        file_extension = os.path.splitext(filename)[1]
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == ".docx" or file_extension == ".doc":
            loader = Docx2txtLoader(file_path)
        elif file_extension == ".txt":
            loader = TextLoader(file_path)
        else:
            continue
        text.extend(loader.load())

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=100,
        chunk_overlap=1,
        length_function=len
    )
    text_chunks = text_splitter.split_documents(text)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vector_store = Chroma.from_documents(
        documents=text_chunks,
        embedding=embedding,
        persist_directory="chroma_store_groq"
    )

    model_name = select_model()
    return vector_store, model_name

def create_conversational_chain(vector_store, model_name):
    """
    Create a conversational chain.

    Args:
        vector_store: Vector store for retriever.
        model_name (str): Name of the conversational model.

    Returns:
        ConversationalRetrievalChain: Initialized conversational chain.
    """
    llm = ChatGroq(
        temperature=0.5,
        model_name=model_name,
        groq_api_key=groq_api_key
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory
    )

    return chain

def select_model():
    """
    Allow user to select a conversational model.

    Returns:
        str: Selected model name.
    """
    model_options = ["mixtral-8x7b-32768", "llama2-70b-4096", "llama3-70b-8192"]
    model_name = st.selectbox("Select a model", model_options)
    return model_name

def conversation_chat(query, chain, history):
    """
    Conduct a conversation with the conversational chain and save the conversation to a text file.

    Args:
        query (str): User query.
        chain: Conversational chain.
        history (list): Chat history.

    Returns:
        str: Response from the conversational chain.
    """
    result = chain({
        "question": query,
        "chat_history": history
    })
    history.append((query, result["answer"]))

    # Save the conversation to a text file
    with open("conversation_log.txt", "a") as file:
        file.write(f"User: {query}\n")
        file.write(f"ChatBot: {result['answer']}\n\n")

    return result["answer"]

def save_message(user_message, chatbot_message):
    """
    Save the conversation message to a text file.

    Args:
        user_message (str): User message.
        chatbot_message (str): Chatbot message.
    """
    with open("conversation_log.txt", "a") as file:
        file.write(f"User: {user_message}\n")
        file.write(f"ChatBot: {chatbot_message}\n\n")
