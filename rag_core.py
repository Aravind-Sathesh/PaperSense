from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


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
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Your answer should be concise and formatted in markdown.

Context: {context}
Chat History: {chat_history}
Question: {question}

Helpful Answer:
"""


def get_qa_chain(vector_store):
    """
    Creates and returns a conversational Q&A chain with a custom prompt.
    """
    llm = ChatOllama(model="mistral", temperature=0.2)

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )

    custom_prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "chat_history", "question"]
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )
    return conversation_chain
