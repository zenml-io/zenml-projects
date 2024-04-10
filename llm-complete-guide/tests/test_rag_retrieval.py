import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.llm_utils import get_db_conn, get_embeddings, get_topn_similar_docs

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


question_doc_pairs = [
    {
        "question": "How do I get going with the Label Studio integration? What are the first steps?",
        "url_ending": "stacks-and-components/component-guide/annotators/label-studio",
    },
    {
        "question": "How can I write my own custom materializer?",
        "url_ending": "user-guide/advanced-guide/data-management/handle-custom-data-types",
    },
    {
        "question": "How do I generate embeddings as part of a RAG pipeline when using ZenML?",
        "url_ending": "user-guide/llmops-guide/rag-with-zenml/embeddings-generation",
    },
    {
        "question": "How do I use failure hooks in my ZenML pipeline?",
        "url_ending": "user-guide/advanced-guide/pipelining-features/use-failure-success-hooks",
    },
    {
        "question": "Can I deploy ZenML self-hosted with Helm? How do I do it?",
        "url_ending": "deploying-zenml/zenml-self-hosted/deploy-with-helm",
    },
]


def query_similar_docs(question: str, url_ending: str) -> tuple:
    """
    Queries for the most similar documents based on the embedded question and returns
    whether the expected URL ending is found in the top similar documents.

    Args:
        question (str): The question to query.
        url_ending (str): The expected URL ending in the similar documents.

    Returns:
        tuple: A tuple containing the question, the expected URL ending, and the query result URLs.
    """
    embedded_question = get_embeddings(question)
    db_conn = get_db_conn()
    top_similar_docs_urls = get_topn_similar_docs(
        embedded_question, db_conn, n=5, only_urls=True
    )
    urls = [
        url[0] for url in top_similar_docs_urls
    ]  # Assuming URLs are the first element in tuples
    return (question, url_ending, urls)


def test_retrieved_docs_retrieve_best_url(question_doc_pairs: list) -> float:
    total_tests = len(question_doc_pairs)
    failures = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Schedule the queries to be executed
        future_to_query = {
            executor.submit(
                query_similar_docs, pair["question"], pair["url_ending"]
            ): pair
            for pair in question_doc_pairs
        }
        for future in as_completed(future_to_query):
            question, url_ending, urls = future.result()
            if all(url_ending not in url for url in urls):
                logging.error(
                    f"Failed for question: {question}. Expected URL ending: {url_ending}. Got: {urls}"
                )
                failures += 1
    logging.info(f"Total tests: {total_tests}. Failures: {failures}")
    failure_rate = (failures / total_tests) * 100
    return round(failure_rate, 2)


if __name__ == "__main__":
    failure_rate = test_retrieved_docs_retrieve_best_url(question_doc_pairs)
    logging.info(f"Retrieval failure rate: {failure_rate}%")
