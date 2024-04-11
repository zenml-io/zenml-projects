from steps.synthetic_data import generate_questions_from_chunks
from zenml import pipeline


@pipeline
def generate_chunk_questions():
    generate_questions_from_chunks()
