from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks.base import BaseCallbackHandler


class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """
        This method is called every time the LLM generates a new token.
        """
        self.text += token
        self.container.markdown(self.text)


def process_and_store_documents(pdf_docs):
    """
    Takes a list of uploaded PDF files, processes them, and returns a vector store.
    """
    all_chunks = []
    for pdf in pdf_docs:
        with open(pdf.name, "wb") as f:
            f.write(pdf.getbuffer())

        loader = PyPDFLoader(pdf.name)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        all_chunks.extend(chunks)

    embedding_model = GPT4AllEmbeddings()
    vector_store = FAISS.from_documents(all_chunks, embedding=embedding_model)

    return vector_store, all_chunks


def get_quick_summary_chain():
    """
    Creates and returns a simple 'stuff' summarization chain for speed.
    """
    llm = ChatOllama(model="mistral", temperature=0.1)
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="stuff",
        verbose=False
    )
    return summary_chain


def get_comprehensive_summary_chain():
    """
    Creates and returns a 'refine' summarization chain for thoroughness.
    """
    llm = ChatOllama(model="mistral", temperature=0.1)
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        verbose=True
    )
    return summary_chain


CUSTOM_PROMPT_TEMPLATE = """
You are a specialized AI assistant. Your task is to provide a concise answer to a question based strictly on a given 'Context'.

**IMPORTANT RULES:**
1. Your entire response MUST be based ONLY on the text provided in the 'Context'. Do not use any of your own knowledge.
2. If the context does not contain the answer, you MUST respond with the single sentence: 'I cannot answer this question based on the provided document context.'

**RESPONSE STYLE:**
- Your final answer MUST be written entirely in the following language: {language}.
- Your final answer MUST be in a {tone} tone.

Context: {context}
Chat History: {chat_history}
Question: {question}

Answer (in {language} with a {tone} tone):
"""


def get_compression_retriever(vector_store):
    """
    Creates and returns a retriever with a cross-encoder re-ranker.
    """
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=3)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    return compression_retriever


def get_qa_chain(retriever, tone="Professional", language="English", callbacks=None):
    """
    Creates and returns a conversational Q&A chain with dynamic prompt controls.
    """
    llm = ChatOllama(
        model="mistral",
        temperature=0.2,
        callbacks=callbacks
    )

    full_prompt_template = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "chat_history",
                         "question", "tone", "language"]
    )

    partial_prompt_object = full_prompt_template.partial(
        tone=tone,
        language=language
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": partial_prompt_object},
        return_source_documents=True
    )

    return conversation_chain
