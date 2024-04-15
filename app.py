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
from streamlit_chat import message

load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]


def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state["history"] = []

    if "generated" not in st.session_state:
        st.session_state["generated"] = [
            "Hi! Finley this side"
        ]

    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey! 👋"]


def conversation_chat(query, chain, history):
    result = chain({
        "question": query,
        "chat_history": history
    })
    history.append((query, result["answer"]))
    return result["answer"]


def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_input(
                "Query:",
                placeholder="Ask me anything about OnFinance",
                key="input"
            )
            submit_button = st.form_submit_button(label="Resolve")

        if submit_button and user_input:
            with st.spinner("Let me check ......"):
                output = conversation_chat(
                    query=user_input,
                    chain=chain,
                    history=st.session_state["history"]
                )

            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(output)

    if st.session_state["generated"]:
        with reply_container:
            for i in range(len(st.session_state["generated"])):
                message(
                    st.session_state["past"][i],
                    is_user=True,
                    key=str(i) + "_user",
                    avatar_style="initials",
                    seed="ME"
                )
                message(
                    st.session_state["generated"][i],
                    key=str(i),
                    avatar_style="initials",
                    seed="OF"
                )


def create_conversational_chain(vector_store, model_name):
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
    model_options = ["mixtral-8x7b-32768", "llama2-70b-4096"]
    model_name = st.selectbox("Select a model", model_options)
    return model_name


def main():
    initialize_session_state()
    st.image("https://iangroup.vc/wp-content/uploads/2023/11/onfinance.jpg", width=300)

    # Load documents from a specific directory instead of user upload
    document_directory = "C:\\Users\\265600\\PycharmProjects\\Firm_managment\\document_directory"
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
        chunk_overlap=110,
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
    chain = create_conversational_chain(vector_store=vector_store, model_name=model_name)

    display_chat_history(chain=chain)


if __name__ == "__main__":
    main()
