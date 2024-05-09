# main.py
import os
import streamlit as st
from dotenv import load_dotenv
from myapp import conversation_chroma, streamlit_ui

# Load environment variables
load_dotenv()

def initialize_app():
    conversation_chroma.initialize_session_state()  # Initialize session state
    vector_store, model_name = conversation_chroma.load_model()
    chain = conversation_chroma.create_conversational_chain(vector_store, model_name)
    return chain

def main():
    st.title("Firm Chat")
    chain = initialize_app()
    streamlit_ui.display_chat_history(chain)

if __name__ == "__main__":
    main()