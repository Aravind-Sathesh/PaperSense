import streamlit as st
from rag_core import process_and_store_documents, get_qa_chain, get_quick_summary_chain, get_comprehensive_summary_chain

st.set_page_config(
    page_title="PaperSense",
    page_icon="📖",
    layout="wide"
)


def main():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "doc_chunks" not in st.session_state:
        st.session_state.doc_chunks = None

    with st.sidebar:
        st.title("PaperSense")
        st.markdown("Your AI-powered research assistant.")
        st.markdown("---")
        st.header("Upload Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs and click 'Process'.",
            accept_multiple_files=True,
            type="pdf"
        )

        if st.button("Process Documents", use_container_width=True, type="primary"):
            if pdf_docs:
                with st.spinner("Processing documents... This may take a moment."):
                    vector_store, doc_chunks = process_and_store_documents(
                        pdf_docs)
                    st.session_state.conversation = get_qa_chain(vector_store)
                    st.session_state.doc_chunks = doc_chunks
                    st.toast("Documents processed successfully!", icon="🎉")
            else:
                st.warning("Please upload at least one PDF file.", icon="⚠️")

        if st.session_state.conversation:
            st.markdown("---")
            st.header("Features")
            if st.button("⚡️ Quick Summary", use_container_width=True):
                with st.spinner("Generating quick summary..."):
                    doc_chunks = st.session_state.doc_chunks
                    chunks_for_summary = doc_chunks[:4] + doc_chunks[-4:]
                    quick_summary_chain = get_quick_summary_chain()
                    summary = quick_summary_chain.invoke(chunks_for_summary)

                    summary_message = {
                        "role": "assistant",
                        "content": f"### ⚡️ Quick Summary\n\n{summary['output_text']}"
                    }
                    st.session_state.chat_history.append(summary_message)
                    st.toast("Quick summary generated!", icon="⚡️")

            if st.button("📚 Comprehensive Summary", use_container_width=True):
                st.info(
                    "This may take several minutes depending on document length...", icon="⏳")
                with st.spinner("Generating comprehensive summary... Please be patient."):
                    comprehensive_summary_chain = get_comprehensive_summary_chain()
                    summary = comprehensive_summary_chain.invoke(
                        st.session_state.doc_chunks)

                    summary_message = {
                        "role": "assistant",
                        "content": f"### 📚 Comprehensive Summary\n\n{summary['output_text']}"
                    }
                    st.session_state.chat_history.append(summary_message)
                    st.toast("Comprehensive summary generated!", icon="📚")

    st.header("Chat with PaperSense")

    if st.session_state.conversation:
        for message in st.session_state.chat_history:
            icon = "👤" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=icon):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "sources" in message and message["sources"]:
                    with st.expander("View Sources"):
                        for i, source in enumerate(message["sources"]):
                            st.info(
                                f"Source {i+1} (Page {source.metadata.get('page', 'N/A')})")
                            st.markdown(
                                f"> {source.page_content.replace('$', 'S')}")

        if user_question := st.chat_input("Ask a question about your documents..."):
            st.session_state.chat_history.append(
                {"role": "user", "content": user_question})
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_question)

            with st.spinner("Assistant is thinking..."):
                response = st.session_state.conversation.invoke(
                    {'question': user_question})
                ai_response = response['answer']
                source_documents = response['source_documents']

                assistant_message = {
                    "role": "assistant",
                    "content": ai_response,
                    "sources": source_documents
                }
                st.session_state.chat_history.append(assistant_message)

                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(ai_response)
                    if source_documents:
                        with st.expander("View Sources"):
                            for i, source in enumerate(source_documents):
                                st.info(
                                    f"Source {i+1} (Page {source.metadata.get('page', 'N/A')})")
                                st.markdown(
                                    f"> {source.page_content.replace('$', 'S')}")

    else:
        st.info(
            "Welcome! Please upload and process your documents in the sidebar to begin.")
        st.markdown("""
            This tool allows you to chat with your research papers, articles, and other PDF documents. 
            
            **How to get started:**
            1.  **Upload** your file(s) in the sidebar on the left.
            2.  Click the **"Process Documents"** button.
            3.  Once processing is complete, this area will become an interactive chat window.
        """)


if __name__ == '__main__':
    main()
