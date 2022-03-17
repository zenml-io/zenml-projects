from zenml.steps import BaseStepConfig


class PreTrainingConfigs(BaseStepConfig):
    # The configuration for the pre-training of the agent
    ENV_NAME: str = "BreakoutDeterministic-v4"

    WRITE_TENSORBOARD: bool = True
    TENSORBOARD_DIR: str = "tensorboard/"

    LEARNING_RATE: float = 0.00001
    INPUT_SHAPE: tuple = (84, 84)
    BATCH_SIZE: int = 32
    SAVE_PATH = "breakout-saves"

    USE_PER: bool = False
    MEM_SIZE: int = 100

    LOAD_FROM: str = None
    LOAD_REPLAY_BUFFER: bool = True

    MAX_NOOP_STEPS: int = 2000

    TOTAL_FRAMES: int = 3000
    FRAMES_BETWEEN_EVAL: int = 100000
    MAX_EPISODE_LENGTH: int = 18000
    EVAL_LENGTH: int = 10000
    UPDATE_FREQ: int = 10000

    PRIORITY_SCALE: float = 0.7  # How much the replay buffer should sample based on priorities. 0 = complete random samples, 1 = completely aligned with priorities
    CLIP_REWARD: bool = True  # Any positive reward is +1, and negative reward is -1, 0 is unchanged

    UPDATE_FREQ: int = 4  # Number of actions between gradient descent steps
    DISCOUNT_FACTOR: float = 0.99  # Gamma, how much to discount future rewards

    BATCH_SIZE: int = 32  # Batch size for training
    MIN_REPLAY_BUFFER_SIZE = 50000  # The minimum size the replay buffer must be before we start to update the agent

    WRITE_TENSORBOARD: bool = True
    EVAL_LENGTH: int = 10000  # Number of frames to evaluate for
