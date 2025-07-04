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

    return vector_store


CUSTOM_PROMPT_TEMPLATE = """
You are a specialized AI assistant for answering questions based strictly on a given document.
You are given a context from the document and a question. Your task is to provide a concise answer.

**IMPORTANT: Your entire response MUST be based ONLY on the text provided in the 'Context' section.**
Do not use any of your own knowledge.

Follow these rules with no exceptions:
1.  Analyze the 'Context' to see if it contains the information needed to answer the 'Question'.
2.  If the context contains the answer, formulate a direct and concise response based on that information.
3.  If the context does NOT contain the information, you MUST respond with the single sentence: 'I cannot answer this question based on the provided document context.'

Context: {context}
Chat History: {chat_history}
Question: {question}

Based strictly on the context provided, here is the answer:
"""


def get_qa_chain(vector_store):
    """
    Creates and returns a conversational Q&A chain.
    """
    llm = ChatOllama(model="mistral", temperature=0.2)

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )

    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

    compressor = CrossEncoderReranker(model=model, top_n=3)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    custom_prompt_object = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "chat_history", "question"]
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=compression_retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt_object},
        return_source_documents=True
    )

    return conversation_chain
