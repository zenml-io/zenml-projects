# Vector Store constants
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50
EMBEDDING_DIMENSIONALITY = (
    384  # Update this to match the dimensionality of the new model
)

# Scraping constants
RATE_LIMIT = 5  # Maximum number of requests per second

# LLM Utils constants
OPENAI_MODEL = "gpt-3.5-turbo"
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
MODEL_NAME_MAP = {
    "gpt4": "gpt-4-0125-preview",
    "gpt35": "gpt-3.5-turbo",
    "claude3": "claude-3-opus-20240229",
    "claudehaiku": "claude-3-haiku-20240307",
}
