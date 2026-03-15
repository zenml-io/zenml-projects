"""LangGraph rollout function for the email search agent.

This module implements the core agent loop using LangGraph's ReAct pattern.
Each rollout represents one episode of the agent attempting to answer a
question by searching through emails.
"""

import uuid
from textwrap import dedent
from typing import Optional

import art
from environment.models import EmailScenario, FinalAnswer
from environment.tools import create_email_tools
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from agent.judge import judge_correctness

# Maximum number of agent turns before stopping
MAX_TURNS = 20


class ProjectTrajectory(art.Trajectory):
    """Extended trajectory that captures the agent's final answer."""

    final_answer: Optional[FinalAnswer] = None


async def rollout(
    model: art.Model,
    email_scenario: EmailScenario,
    db_path: str = "./enron_emails.db",
    judge_model: str = "openai/gpt-4.1",
) -> ProjectTrajectory:
    """Execute a single rollout of the email search agent.

    This function:
    1. Sets up the LangGraph ReAct agent with email search tools
    2. Runs the agent on the given scenario
    3. Judges the correctness of the final answer
    4. Returns a trajectory with metrics for training

    Args:
        model: The ART model to use for inference.
        email_scenario: The scenario containing the question and metadata.
        db_path: Path to the email database.
        judge_model: LiteLLM model identifier for correctness judging.

    Returns:
        ProjectTrajectory with the agent's conversation and metrics.
    """
    # Import here to avoid circular imports and allow lazy loading
    from art.langgraph import init_chat_model

    scenario = email_scenario.scenario

    traj = ProjectTrajectory(
        reward=0.0,
        messages_and_choices=[],
        metadata={
            "scenario_id": scenario.id,
            "step": email_scenario.step,
        },
    )

    system_prompt = dedent(
        f"""
        You are an email search agent. You are given a user query and a list
        of tools you can use to search the user's email. Use the tools to
        search the user's emails and find the answer to the user's query.
        You may take up to {MAX_TURNS} turns to find the answer, so if your
        first search doesn't find the answer, you can try with different
        keywords.

        User's email address is {scenario.inbox_address}
        Today's date is {scenario.query_date}

        When you have found the answer, use the return_final_answer_tool to
        provide your final answer along with the source message IDs.
        """
    )

    # Mutable container for the final answer (captured by tool closure)
    final_answer_container: dict[str, Optional[FinalAnswer]] = {"value": None}

    def on_final_answer(answer: FinalAnswer) -> None:
        final_answer_container["value"] = answer

    # Create scenario-specific tools
    tools = create_email_tools(
        scenario=scenario,
        db_path=db_path,
        on_final_answer=on_final_answer,
    )

    # Initialize the chat model from ART
    chat_model = init_chat_model(model.name, temperature=1.0)

    # Create the LangGraph ReAct agent
    react_agent = create_react_agent(chat_model, tools)

    try:
        # Run the agent
        config = {
            "configurable": {"thread_id": str(uuid.uuid4())},
            "recursion_limit": MAX_TURNS,
        }

        await react_agent.ainvoke(
            {
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=scenario.question),
                ]
            },
            config=config,
        )

        # Check if we got a final answer
        if final_answer_container["value"]:
            traj.final_answer = final_answer_container["value"]
            # Score the trajectory using the judge
            correctness_response = await judge_correctness(
                scenario,
                traj.final_answer.answer,
                judge_model=judge_model,
            )
            traj.metrics["correct"] = float(correctness_response.accept)

    except Exception as e:
        print(f"Error running LangGraph agent: {e}")
        traj.messages_and_choices.append(
            {"role": "assistant", "content": f"Error: {str(e)}"}
        )

    return traj
