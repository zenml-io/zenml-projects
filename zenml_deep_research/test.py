from zenml import pipeline, step
from zenml.integrations.discord.steps.discord_alerter_ask_step import (
    discord_alerter_ask_step,
)


@step
def my_step() -> str:
    return "Do you approve?"


@step
def my_step2(response: bool):
    if response:
        return "User approved the operation"
    else:
        return "User did not approve the operation"


@pipeline(enable_cache=False)
def my_pipeline():
    message = my_step()
    response = discord_alerter_ask_step(message)
    my_step2(response)


if __name__ == "__main__":
    my_pipeline()
