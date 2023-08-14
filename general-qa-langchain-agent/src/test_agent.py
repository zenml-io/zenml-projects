from zenml.post_execution import get_pipeline

if __name__ == "__main__":
    pipeline = get_pipeline(pipeline="zenml_agent")

    run = pipeline.versions[0].runs[0]

    agent = run.get_step("agent_creator").output.read()

    print(agent.run("What is ZenML?"))