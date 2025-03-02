from typing import Optional

from steps.create_prompt import create_prompt
from steps.eval_langfuse import fast_eval, visualize_fast_eval_results
from zenml import pipeline


@pipeline(enable_cache=False)
def llm_langfuse_evaluation(after: Optional[str] = None) -> None:
    """Evaluate the LLM using Langfuse."""
    # create prompt
    prompt = create_prompt()
    results = fast_eval(after=after, prompt=prompt)
    visualize_fast_eval_results(results)


if __name__ == "__main__":
    llm_langfuse_evaluation()
