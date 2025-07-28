import streamlit as st
from rag_utils import load_and_embed_doc, get_groq_answer
from dotenv import load_dotenv
import os

def main():
    load_dotenv()
    st.set_page_config(page_title="📄 Chat with Document", layout="wide")
    st.title("📄 Chat with Your PDF")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectordb" not in st.session_state:
        st.session_state.vectordb = None

    # Upload PDF
    uploaded_file = st.file_uploader("Upload a PDF to begin", type=["pdf"])

    if uploaded_file and st.session_state.vectordb is None:
        with st.spinner("Reading and embedding document..."):
            vectordb, _ = load_and_embed_doc(uploaded_file)
            st.session_state.vectordb = vectordb
        st.success("Document ready! Start chatting below 👇")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input box
    if st.session_state.vectordb:
        user_input = st.chat_input("Ask a question from the document...")

        if user_input:
            # Show user message
            st.chat_message("user").markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_groq_answer(user_input, st.session_state.vectordb)
                    st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

# Run main
if __name__ == "__main__":
    main()
