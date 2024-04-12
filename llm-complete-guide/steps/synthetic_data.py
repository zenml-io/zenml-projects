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

from typing import List

from litellm import completion
from structures import Document
from zenml import step

LOCAL_MODEL = "ollama/mixtral"


def generate_question(chunk: str, local: bool = False) -> str:
    """Generate a question from a chunk.

    Args:
        chunk: Text chunk to generate a question from.

    Returns:
        Generated question.
    """
    model = LOCAL_MODEL if local else "gpt-3.5-turbo"
    response = completion(
        model=model,
        messages=[
            {
                "content": "This is some text from ZenML's documentation. Please generate a question from this text.",
                "role": "user",
            }
        ],
        api_base="http://localhost:11434",
    )
    breakpoint()
    return response[0]["text"]


@step
def generate_questions_from_chunks(
    docs_with_embeddings: List[Document],
    local: bool = False,
) -> List[Document]:
    """Generate questions from chunks.

    Args:
        docs_with_embeddings: List of documents with embeddings.
        local: Whether to run the pipeline with a local LLM.

    Returns:
        List of documents with generated questions added.
    """
    for doc in docs_with_embeddings:
        doc.generated_questions = [generate_question(doc.page_content, local)]

    assert all(doc.generated_questions for doc in docs_with_embeddings)
    return docs_with_embeddings
