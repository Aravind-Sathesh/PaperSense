import streamlit as st
from rag_core import process_and_store_documents, get_qa_chain


def main():
    st.set_page_config(page_title="Chat with Your Docs", page_icon="📚")
    st.title("Chat with Your Documents using Ollama 💬")
    st.write("Upload your PDFs and ask questions about their content.")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar for document upload
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click 'Process'", accept_multiple_files=True, type="pdf")

        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing documents... This may take a moment."):
                    vector_store = process_and_store_documents(pdf_docs)
                    st.session_state.conversation = get_qa_chain(vector_store)
                    st.success(
                        "Processing complete! You can now ask questions.")
            else:
                st.warning("Please upload at least one PDF file.")

    # Main chat interface
    if st.session_state.conversation:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_question := st.chat_input("Ask a question about your documents:"):
            st.session_state.chat_history.append(
                {"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.spinner("Assistant is thinking..."):
                response = st.session_state.conversation(
                    {'question': user_question})
                ai_response = response['answer']

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": ai_response})
                with st.chat_message("assistant"):
                    st.markdown(ai_response)
    else:
        st.info(
            "Please upload and process your documents in the sidebar to start chatting.")


if __name__ == '__main__':
    main()
