from zenml import step


@step(enable_cache=False)
def fast_eval() -> None:
    pass
