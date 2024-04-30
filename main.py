# main.py
import os
import streamlit as st
from dotenv import load_dotenv
from myapp import conversation, streamlit_ui

# Load environment variables
load_dotenv()

def main():
    st.title("Firm Chat")
    conversation.initialize_session_state()
    vector_store, model_name = conversation.load_model()
    chain = conversation.create_conversational_chain(vector_store, model_name)
    streamlit_ui.display_chat_history(chain)

if __name__ == "__main__":
    main()
