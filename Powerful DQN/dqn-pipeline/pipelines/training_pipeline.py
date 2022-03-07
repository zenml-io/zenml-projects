from zenml.pipelines import pipeline


@pipeline
def train_pipeline(GameWrap, build_dqn, replay_buffer, agent, get_information_meta, train):
    game_wrapper = GameWrap()
    MAIN_DQN, target_dqn = build_dqn(game_wrapper)
    replay_buffer = replay_buffer()
    agent = agent(game_wrapper, replay_buffer, MAIN_DQN, target_dqn)
    frame_number, rewards, loss_list = get_information_meta(agent)
    train(game_wrapper, loss_list, rewards, frame_number, agent)

    