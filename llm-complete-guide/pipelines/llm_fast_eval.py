from typing import Optional

from steps.fast_eval import fast_eval
from zenml import pipeline


@pipeline(enable_cache=False)
def llm_fast_eval(after: Optional[str] = None) -> None:
    fast_eval(after=after)


if __name__ == "__main__":
    llm_fast_eval()
