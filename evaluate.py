from langchain_community.document_loaders import PyPDFLoader
from rag_core import process_and_store_documents, get_qa_chain
import os
from datasets import Dataset
from langchain_ollama import ChatOllama
from langchain_community.embeddings import GPT4AllEmbeddings
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

EVALUATION_DATASET = {
    'question': [
        "What are the two main components of the Transformer architecture?",
        "What is the core mechanism the Transformer is based on?",
        "What is the title of the paper mentioned in the context?",
        "What BLEU score did the big transformer model achieve on the WMT 2014 English-to-German translation task?",
    ],
    'ground_truth': [
        "The Transformer architecture is based on two main components: an encoder and a decoder.",
        "The Transformer is the first transduction model relying entirely on self-attention.",
        "Attention Is All You Need",
        "The big transformer model achieved a BLEU score of 28.4.",
    ]
}


class MockUploadedFile:
    def __init__(self, name, data):
        self.name = name
        self._data = data

    def getbuffer(self):
        return self._data


def create_rag_pipeline(pdf_path):
    """Helper function to set up the RAG pipeline from a local PDF path."""
    print(f"Processing PDF: {pdf_path}...")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    mock_file = MockUploadedFile(os.path.basename(pdf_path), pdf_bytes)

    vector_store = process_and_store_documents([mock_file])
    qa_chain = get_qa_chain(vector_store)
    return qa_chain


def run_evaluation(qa_chain, dataset_dict):
    """Runs the RAG pipeline on the dataset and collects results."""
    results = []
    for question in dataset_dict['question']:
        print(f"Running question: '{question}'...")
        response = qa_chain.invoke({'question': question})
        results.append({
            'question': question,
            'answer': response['answer'],
            'contexts': [doc.page_content for doc in response['source_documents']],
        })
    return results


def main():
    PDF_FILE_PATH = "Attention Is All You Need.pdf"
    qa_pipeline = create_rag_pipeline(PDF_FILE_PATH)
    results_list = run_evaluation(qa_pipeline, EVALUATION_DATASET)
    eval_dataset = Dataset.from_dict({
        "question": [res['question'] for res in results_list],
        "answer": [res['answer'] for res in results_list],
        "contexts": [res['contexts'] for res in results_list],
        "ground_truth": EVALUATION_DATASET['ground_truth']
    })

    metrics = [
        faithfulness,
        answer_relevancy,
        context_recall,
    ]

    ragas_llm = ChatOllama(model="mistral")
    ragas_embeddings = GPT4AllEmbeddings()

    print("Running RAGAs evaluation...")
    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    print("Evaluation Results:")
    print(result.to_pandas())


if __name__ == '__main__':
    main()
