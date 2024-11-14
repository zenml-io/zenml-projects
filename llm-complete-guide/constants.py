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
import os

# Vector Store constants
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 50
EMBEDDING_DIMENSIONALITY = (
    384  # Update this to match the dimensionality of the new model
)

# ZenML constants
ZENML_CHATBOT_MODEL = "zenml-docs-qa-chatbot"

# Scraping constants
RATE_LIMIT = 5  # Maximum number of requests per second

# LLM Utils constants
OPENAI_MODEL = "gpt-4o"
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
# DATASET_NAME_DEFAULT = "zenml/rag_qa_embedding_questions_0_60_0"
DATASET_NAME_DEFAULT = "zenml/rag_qa_embedding_questions"
DATASET_NAME_DISTILABEL = f"{DATASET_NAME_DEFAULT}_distilabel"
DATASET_NAME_ARGILLA = DATASET_NAME_DEFAULT.replace("zenml/", "")
OPENAI_MODEL_GEN = "gpt-4o"
OPENAI_MODEL_GEN_KWARGS_EMBEDDINGS = {
    "temperature": 0.7,
    "max_new_tokens": 512,
}
EMBEDDINGS_MODEL_ID_BASELINE = "Snowflake/snowflake-arctic-embed-m-v1.5"
EMBEDDINGS_MODEL_ID_FINE_TUNED = "finetuned-snowflake-arctic-embed-m-v1.5"
EMBEDDINGS_MODEL_MATRYOSHKA_DIMS: list[int] = [
    384,
    256,
    128,
    64,
]  # Important: large to small
USE_ARGILLA_ANNOTATIONS = False

SECRET_NAME = os.getenv("ZENML_PROJECT_SECRET_NAME", "llm-complete")
SECRET_NAME_ELASTICSEARCH = "elasticsearch-zenml"
