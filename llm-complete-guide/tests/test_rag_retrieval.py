from utils.llm_utils import get_embeddings, get_db_conn, get_topn_similar_docs


question_doc_pairs = [
    {
        "question": "Does ZenML support the Label Studio orchestrator?",
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
]


def test_retrieved_docs_retrieve_best_url(question_doc_pairs: list) -> float:
    """
    This function tests if the most similar documents retrieved for a given question contain the expected URL.
    It prints the total number of tests and the number of failures.

    Args:
        question_doc_pairs (list): A list of dictionaries. Each dictionary contains a question and the expected URL ending.

    Returns:
        float: The failure rate as a percentage, rounded to 2 decimal places.
    """
    total_tests = len(question_doc_pairs)
    failures = 0
    for question_doc_pair in question_doc_pairs:
        question = question_doc_pair["question"]
        url_ending = question_doc_pair["url_ending"]
        embedded_question = get_embeddings(question)
        db_conn = get_db_conn()
        top_similar_docs_urls = get_topn_similar_docs(
            embedded_question, db_conn, n=3, only_urls=True
        )
        # Unpack URLs from tuples and check if url_ending is in any of them
        urls = [
            url[0] for url in top_similar_docs_urls
        ]  # Assuming URLs are the first element in tuples
        if all(url_ending not in url for url in urls):
            print(
                f"Failed for question: {question}. Expected URL ending: {url_ending}. Got: {urls}"
            )
            failures += 1
    print(f"Total tests: {total_tests}. Failures: {failures}")
    failure_rate = (failures / total_tests) * 100
    return round(failure_rate, 2)


if __name__ == "__main__":
    failure_rate = test_retrieved_docs_retrieve_best_url(question_doc_pairs)
    print(f"Retrieval failure rate: {failure_rate}%")
