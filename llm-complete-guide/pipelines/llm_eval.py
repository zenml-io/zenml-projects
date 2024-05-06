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
    retrieval_evaluation_full_with_reranking,
    retrieval_evaluation_small,
    retrieval_evaluation_small_with_reranking,
)
from steps.eval_visualisation import visualize_evaluation_results
from zenml import pipeline


@pipeline
def llm_eval() -> None:
    """Executes the pipeline to evaluate a RAG pipeline."""
    # Retrieval evals
    failure_rate_retrieval = retrieval_evaluation_small()
    full_retrieval_answers = retrieval_evaluation_full()
    failure_rate_retrieval_reranking = (
        retrieval_evaluation_small_with_reranking()
    )
    full_retrieval_answers_reranking = (
        retrieval_evaluation_full_with_reranking()
    )

    # E2E evals
    (
        failure_rate_bad_answers,
        failure_rate_bad_immediate_responses,
        failure_rate_good_responses,
    ) = e2e_evaluation()
    (
        average_toxicity_score,
        average_faithfulness_score,
        average_helpfulness_score,
        average_relevance_score,
    ) = e2e_evaluation_llm_judged()

    visualize_evaluation_results(
        failure_rate_retrieval,
        failure_rate_retrieval_reranking,
        full_retrieval_answers,
        full_retrieval_answers_reranking,
        failure_rate_bad_answers,
        failure_rate_bad_immediate_responses,
        failure_rate_good_responses,
        average_toxicity_score,
        average_faithfulness_score,
        average_helpfulness_score,
        average_relevance_score,
    )
