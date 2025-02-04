from typing import Optional

from steps.fast_eval import fast_eval, visualize_fast_eval_results
from zenml import pipeline


@pipeline(enable_cache=False)
def llm_fast_eval(after: Optional[str] = None) -> None:
    results = fast_eval(after=after)
    visualize_fast_eval_results(results)


if __name__ == "__main__":
    llm_fast_eval()
