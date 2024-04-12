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

from steps.synthetic_data import generate_questions_from_chunks
from zenml import pipeline
from zenml.client import Client


@pipeline
def generate_chunk_questions(local: bool = False):
    client = Client()
    docs_with_embeddings = client.get_artifact_version(
        name_id_or_prefix="documents_with_embeddings"
    )
    generate_questions_from_chunks(docs_with_embeddings)
