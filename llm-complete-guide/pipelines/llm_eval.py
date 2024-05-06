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

from steps.eval_e2e import e2e_evaluation, e2e_evaluation_llm_judged
from steps.eval_retrieval import (
    retrieval_evaluation_full,
    retrieval_evaluation_small,
)
from zenml import pipeline


@pipeline
def llm_eval() -> None:
    """Executes the pipeline to evaluate a RAG pipeline."""
    # Retrieval evals
    failure_rate_retrieval = retrieval_evaluation_small()
    full_retrieval_answers = retrieval_evaluation_full()

    # E2E evals
    e2e_eval_tuple = e2e_evaluation()
    e2e_llm_judged_tuple = e2e_evaluation_llm_judged()

    # visualize_evaluation_results(
    #     failure_rate_retrieval,
    #     e2e_answers,
    #     full_retrieval_answers,
    # )
