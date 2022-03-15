from zenml.pipelines import pipeline


@pipeline
def train_pipeline(
    game_wrap, build_dqn, replay_buffer, agent, get_information_meta, train
):
    """
    It trains the agent.
    Args:
        game_wrap: This is a function that returns a GameWrapper object. The GameWrapper object wraps
        over the game that you want to train on. It has functions that can be used to get the available
        actions, get the current state, etc
        build_dqn: This is a function that returns a DQN. The parameters are the game_wrapper, which
        is the game wrapper object that we created earlier
        replay_buffer: The replay buffer is a class that holds all the experiences a DQN has seen,
        and samples from it randomly to train the DQN
        agent: The agent that will be used to play the game
        get_information_meta: This is a function that returns the frame number, rewards, and loss
        list
        train: The function that will be called inside the train_pipeline function. This is the
        function that will be called
    """
    game_wrapper = game_wrap()
    main_dqn, target_dqn = build_dqn(game_wrapper)
    replay_buffer = replay_buffer()
    agent = agent(game_wrapper, replay_buffer, main_dqn, target_dqn)
    frame_number, rewards, loss_list = get_information_meta(agent)
    train(game_wrapper, loss_list, rewards, frame_number, agent)

