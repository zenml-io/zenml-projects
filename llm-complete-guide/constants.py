# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Vector Store constants
CHUNK_SIZE = 2000
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
    "gpt4": "gpt-4",
    "gpt35": "gpt-3.5-turbo",
    "claude3": "claude-3-opus-20240229",
    "claudehaiku": "claude-3-haiku-20240307",
}

# CHUNKING_METHOD = "split-by-document"
CHUNKING_METHOD = "split-by-header"
DATASET_NAME = f"zenml/rag_qa_embedding_questions_{CHUNKING_METHOD}"
MODEL_PATH = "all-MiniLM-L6-v2"
# MODEL_PATH = "embedding-data/distilroberta-base-sentence-transformer"
NUM_EPOCHS = 30
WARMUP_STEPS = 0.1  # 10% of train data
NUM_GENERATIONS = 2
EVAL_BATCH_SIZE = 64

DUMMY_DATASET_NAME = "embedding-data/sentence-compression"
# DUMMY_MODEL_PATH = "embedding-data/distilroberta-base-sentence-transformer"
DUMMY_MODEL_PATH = "all-MiniLM-L6-v2"
DUMMY_EPOCHS = 10

# Markdown Loader constants
FILES_TO_IGNORE = [
    "toc.md",
]

# embeddings finetuning constants
EMBEDDINGS_MODEL_NAME_ZENML = "finetuned-zenml-docs-embeddings"
DATASET_NAME_EMBEDDINGS = "zenml/rag_qa_embedding_questions_0_60_0"
DATASET_NAME_DISTILABEL_EMBEDDINGS = f"{DATASET_NAME_EMBEDDINGS}_distilabel"
DATASET_NAME_ARGILLA_EMBEDDINGS = DATASET_NAME_EMBEDDINGS.replace("zenml/", "")
OPENAI_MODEL_EMBEDDINGS = "gpt-4o"
OPENAI_MODEL_GEN_KWARGS_EMBEDDINGS = {
    "temperature": 0.7,
    "max_new_tokens": 512,
}
EMBEDDINGS_MODEL_NAME_BASELINE = "Snowflake/snowflake-arctic-embed-m"
EMBEDDINGS_MODEL_NAME_FINE_TUNED = "finetuned-snowflake-arctic-embed-m"
EMBEDDINGS_MODEL_MATRYOSHKA_DIMENSIONS: list[int] = [384, 256, 128, 64]  # Important: large to small
