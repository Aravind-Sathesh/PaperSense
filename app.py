import streamlit as st
from rag_core import process_and_store_documents, get_qa_chain


def main():
    st.set_page_config(page_title="Chat with Your Docs", page_icon="📚")
    st.title("Chat with Your Documents using Ollama 💬")
    st.write("Upload your PDFs and ask questions about their content.")


if __name__ == '__main__':
    main()
