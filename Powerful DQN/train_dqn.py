import numpy as np
import cv2

import random
import os
import json
import time

import gym
env = gym.make('BreakoutDeterministic-v4')


import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input,
                                     Lambda, Subtract)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop

from zenml.pipelines import pipeline
from zenml.repository import Repository
from zenml.steps import Output, step, BaseStepConfig
from zenml.materializers.base_materializer import BaseMaterializer

# from GameWrap import GameWrapper
from replay_buffer import ReplayBuffer
# from agent import Agent 

from typing import Any, Type, Union, List
import pickle
DEFAULT_FILENAME = "PyEnvironment"
from zenml.io import fileio


# import keras
from tensorflow import keras

def build_q_network(n_actions, learning_rate=0.00001, input_shape=(84, 84), history_length=4):
    """Builds a dueling DQN as a Keras model
    Arguments:
        n_actions: Number of possible action the agent can take
        learning_rate: Learning rate
        input_shape: Shape of the preprocessed frame the model sees
        history_length: Number of historical frames the agent can see
    Returns:
        A compiled Keras model
    """
    model_input = Input(shape=(input_shape[0], input_shape[1], history_length))
    x = Lambda(lambda layer: layer / 255)(model_input)  # normalize by 255

    x = Conv2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    x = Conv2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    x = Conv2D(64, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    x = Conv2D(1024, (7, 7), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)

    # Split into value and advantage streams
    val_stream, adv_stream = Lambda(lambda w: tf.split(w, 2, 3))(x)  # custom splitting layer

    val_stream = Flatten()(val_stream)
    val = Dense(1, kernel_initializer=VarianceScaling(scale=2.))(val_stream)

    adv_stream = Flatten()(adv_stream)
    adv = Dense(n_actions, kernel_initializer=VarianceScaling(scale=2.))(adv_stream)

    # Combine streams into Q-Values
    reduce_mean = Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))  # custom layer for reduce mean

    q_vals = Add()([val, Subtract()([adv, reduce_mean(adv)])])

    # Build model
    model = Model(model_input, q_vals)
    model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())

    return model

class GameWrapper:
    """Wrapper for the environment provided by Gym"""
    def __init__(self, env_name, no_op_steps=10, history_length=4):
        self.env = gym.make(env_name)
        self.no_op_steps = no_op_steps
        self.history_length = 4

        self.state = None
        self.last_lives = 0

    def reset(self, evaluation=False):
        """Resets the environment
        Arguments:
            evaluation: Set to True when the agent is being evaluated. Takes a random number of no-op steps if True.
        """

        self.frame = self.env.reset()
        self.last_lives = 0

        # If evaluating, take a random number of no-op steps.
        # This adds an element of randomness, so that the each
        # evaluation is slightly different.
        if evaluation:
            for _ in range(random.randint(0, self.no_op_steps)):
                self.env.step(1)

        # For the initial state, we stack the first frame four times
        self.state = np.repeat(process_frame(self.frame), self.history_length, axis=2)

    def step(self, action, render_mode=None):
        """Performs an action and observes the result
        Arguments:
            action: An integer describe action the agent chose
            render_mode: None doesn't render anything, 'human' renders the screen in a new window, 'rgb_array' returns an np.array with rgb values
        Returns:
            processed_frame: The processed new frame as a result of that action
            reward: The reward for taking that action
            terminal: Whether the game has ended
            life_lost: Whether a life has been lost
            new_frame: The raw new frame as a result of that action
            If render_mode is set to 'rgb_array' this also returns the rendered rgb_array
        """
        new_frame, reward, terminal, info = self.env.step(action)

        # In the commonly ignored 'info' or 'meta' data returned by env.step
        # we can get information such as the number of lives the agent has.

        # We use this here to find out when the agent loses a life, and
        # if so, we set life_lost to True.

        # We use life_lost to force the agent to start the game
        # and not sit around doing nothing.
        if info['ale.lives'] < self.last_lives:
            life_lost = True
        else:
            life_lost = terminal
        self.last_lives = info['ale.lives']

        processed_frame = process_frame(new_frame)
        self.state = np.append(self.state[:, :, 1:], processed_frame, axis=2)

        if render_mode == 'rgb_array':
            return processed_frame, reward, terminal, life_lost, self.env.render(render_mode)
        elif render_mode == 'human':
            self.env.render()

        return processed_frame, reward, terminal, life_lost
class Agent(object):
    """Implements a standard DDDQN agent"""
    def __init__(self,
                 dqn,
                 target_dqn,
                 replay_buffer,
                 n_actions,
                 input_shape=(84, 84),
                 batch_size=32,
                 history_length=4,
                 eps_initial=1,
                 eps_final=0.1,
                 eps_final_frame=0.01,
                 eps_evaluation=0.0,
                 eps_annealing_frames=1000000,
                 replay_buffer_start_size=50000,
                 max_frames=25000000,
                 use_per=True):
        """
        Arguments:
            dqn: A DQN (returned by the DQN function) to predict moves
            target_dqn: A DQN (returned by the DQN function) to predict target-q values.  This can be initialized in the same way as the dqn argument
            replay_buffer: A ReplayBuffer object for holding all previous experiences
            n_actions: Number of possible actions for the given environment
            input_shape: Tuple/list describing the shape of the pre-processed environment
            batch_size: Number of samples to draw from the replay memory every updating session
            history_length: Number of historical frames available to the agent
            eps_initial: Initial epsilon value.
            eps_final: The "half-way" epsilon value.  The epsilon value decreases more slowly after this
            eps_final_frame: The final epsilon value
            eps_evaluation: The epsilon value used during evaluation
            eps_annealing_frames: Number of frames during which epsilon will be annealed to eps_final, then eps_final_frame
            replay_buffer_start_size: Size of replay buffer before beginning to learn (after this many frames, epsilon is decreased more slowly)
            max_frames: Number of total frames the agent will be trained for
            use_per: Use PER instead of classic experience replay
        """

        self.n_actions = n_actions
        self.input_shape = input_shape
        self.history_length = history_length

        # Memory information
        self.replay_buffer_start_size = replay_buffer_start_size
        self.max_frames = max_frames
        self.batch_size = batch_size

        self.replay_buffer = replay_buffer
        self.use_per = use_per

        # Epsilon information
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_frame = eps_final_frame
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_frames = eps_annealing_frames

        # Slopes and intercepts for exploration decrease
        # (Credit to Fabio M. Graetz for this and calculating epsilon based on frame number)
        self.slope = -(self.eps_initial - self.eps_final) / self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope*self.replay_buffer_start_size
        self.slope_2 = -(self.eps_final - self.eps_final_frame) / (self.max_frames - self.eps_annealing_frames - self.replay_buffer_start_size)
        self.intercept_2 = self.eps_final_frame - self.slope_2*self.max_frames

        # DQN
        self.DQN = dqn
        self.target_dqn = target_dqn

    def calc_epsilon(self, frame_number, evaluation=False):
        """Get the appropriate epsilon value from a given frame number
        Arguments:
            frame_number: Global frame number (used for epsilon)
            evaluation: True if the model is evaluating, False otherwise (uses eps_evaluation instead of default epsilon value)
        Returns:
            The appropriate epsilon value
        """
        if evaluation:
            return self.eps_evaluation
        elif frame_number < self.replay_buffer_start_size:
            return self.eps_initial
        elif frame_number >= self.replay_buffer_start_size and frame_number < self.replay_buffer_start_size + self.eps_annealing_frames:
            return self.slope*frame_number + self.intercept
        elif frame_number >= self.replay_buffer_start_size + self.eps_annealing_frames:
            return self.slope_2*frame_number + self.intercept_2

    def get_action(self, frame_number, state, evaluation=False):
        """Query the DQN for an action given a state
        Arguments:
            frame_number: Global frame number (used for epsilon)
            state: State to give an action for
            evaluation: True if the model is evaluating, False otherwise (uses eps_evaluation instead of default epsilon value)
        Returns:
            An integer as the predicted move
        """

        # Calculate epsilon based on the frame number
        eps = self.calc_epsilon(frame_number, evaluation)

        # With chance epsilon, take a random action
        if np.random.rand(1) < eps:
            return np.random.randint(0, self.n_actions)

        # Otherwise, query the DQN for an action
        q_vals = self.DQN.predict(state.reshape((-1, self.input_shape[0], self.input_shape[1], self.history_length)))[0]
        return q_vals.argmax()

    def get_intermediate_representation(self, state, layer_names=None, stack_state=True):
        """
        Get the output of a hidden layer inside the model.  This will be/is used for visualizing model
        Arguments:
            state: The input to the model to get outputs for hidden layers from
            layer_names: Names of the layers to get outputs from.  This can be a list of multiple names, or a single name
            stack_state: Stack `state` four times so the model can take input on a single (84, 84, 1) frame
        Returns:
            Outputs to the hidden layers specified, in the order they were specified.
        """
        # Prepare list of layers
        if isinstance(layer_names, list) or isinstance(layer_names, tuple):
            layers = [self.DQN.get_layer(name=layer_name).output for layer_name in layer_names]
        else:
            layers = self.DQN.get_layer(name=layer_names).output

        # Model for getting intermediate output
        temp_model = tf.keras.Model(self.DQN.inputs, layers)

        # Stack state 4 times
        if stack_state:
            if len(state.shape) == 2:
                state = state[:, :, np.newaxis]
            state = np.repeat(state, self.history_length, axis=2)

        # Put it all together
        return temp_model.predict(state.reshape((-1, self.input_shape[0], self.input_shape[1], self.history_length)))

    def update_target_network(self):
        """Update the target Q network"""
        self.target_dqn.set_weights(self.DQN.get_weights())

    def add_experience(self, action, frame, reward, terminal, clip_reward=True):
        """Wrapper function for adding an experience to the Agent's replay buffer"""
        self.replay_buffer.add_experience(action, frame, reward, terminal, clip_reward)

    def learn(self, batch_size, gamma, frame_number, priority_scale=1.0):
        """Sample a batch and use it to improve the DQN
        Arguments:
            batch_size: How many samples to draw for an update
            gamma: Reward discount
            frame_number: Global frame number (used for calculating importances)
            priority_scale: How much to weight priorities when sampling the replay buffer. 0 = completely random, 1 = completely based on priority
        Returns:
            The loss between the predicted and target Q as a float
        """

        if self.use_per:
            (states, actions, rewards, new_states, terminal_flags), importance, indices = self.replay_buffer.get_minibatch(batch_size=self.batch_size, priority_scale=priority_scale)
            importance = importance ** (1-self.calc_epsilon(frame_number))
        else:
            states, actions, rewards, new_states, terminal_flags = self.replay_buffer.get_minibatch(batch_size=self.batch_size, priority_scale=priority_scale)

        # Main DQN estimates best action in new states
        arg_q_max = self.DQN.predict(new_states).argmax(axis=1)

        # Target DQN estimates q-vals for new states
        future_q_vals = self.target_dqn.predict(new_states)
        double_q = future_q_vals[range(batch_size), arg_q_max]

        # Calculate targets (bellman equation)
        target_q = rewards + (gamma*double_q * (1-terminal_flags))

        # Use targets to calculate loss (and use loss to calculate gradients)
        with tf.GradientTape() as tape:
            q_values = self.DQN(states)

            one_hot_actions = tf.keras.utils.to_categorical(actions, self.n_actions, dtype=np.float32)  # using tf.one_hot causes strange errors
            Q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)

            error = Q - target_q
            loss = tf.keras.losses.Huber()(target_q, Q)

            if self.use_per:
                # Multiply the loss by importance, so that the gradient is also scaled.
                # The importance scale reduces bias against situataions that are sampled
                # more frequently.
                loss = tf.reduce_mean(loss * importance)

        model_gradients = tape.gradient(loss, self.DQN.trainable_variables)
        self.DQN.optimizer.apply_gradients(zip(model_gradients, self.DQN.trainable_variables))

        if self.use_per:
            self.replay_buffer.set_priorities(indices, error)

        return float(loss.numpy()), error

    def save(self, folder_name, **kwargs):
        """Saves the Agent and all corresponding properties into a folder
        Arguments:
            folder_name: Folder in which to save the Agent
            **kwargs: Agent.save will also save any keyword arguments passed.  This is used for saving the frame_number
        """

        # Create the folder for saving the agent
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        # Save DQN and target DQN
        self.DQN.save(folder_name + '/dqn.h5')
        self.target_dqn.save(folder_name + '/target_dqn.h5')

        # Save replay buffer
        self.replay_buffer.save(folder_name + '/replay-buffer')

        # Save meta
        with open(folder_name + '/meta.json', 'w+') as f:
            f.write(json.dumps({**{'buff_count': self.replay_buffer.count, 'buff_curr': self.replay_buffer.current}, **kwargs}))  # save replay_buffer information and any other information

    def load(self, folder_name, load_replay_buffer=True):
        """Load a previously saved Agent from a folder
        Arguments:
            folder_name: Folder from which to load the Agent
        Returns:
            All other saved attributes, e.g., frame number
        """

        if not os.path.isdir(folder_name):
            raise ValueError(f'{folder_name} is not a valid directory')

        # Load DQNs
        self.DQN = tf.keras.models.load_model(folder_name + '/dqn.h5')
        self.target_dqn = tf.keras.models.load_model(folder_name + '/target_dqn.h5')
        self.optimizer = self.DQN.optimizer

        # Load replay buffer
        if load_replay_buffer:
            self.replay_buffer.load(folder_name + '/replay-buffer')

        # Load meta
        with open(folder_name + '/meta.json', 'r') as f:
            meta = json.load(f)

        if load_replay_buffer:
            self.replay_buffer.count = meta['buff_count']
            self.replay_buffer.current = meta['buff_curr']

        del meta['buff_count'], meta['buff_curr']  # we don't want to return this information
        return meta

class PreTrainingConfigs(BaseStepConfig): 
    ENV_NAME : str = "BreakoutDeterministic-v4" 
    MAX_NOOP_STEPS : int = 20 
    
    WRITE_TENSORBOARD : bool = True
    TENSORBOARD_DIR : str = 'tensorboard/'

    LEARNING_RATE : float = 0.00001
    INPUT_SHAPE : tuple = (84, 84)
    BATCH_SIZE : int = 32 

    USE_PER : bool = False
    MEM_SIZE : int  = 1000000

    LOAD_FROM : str = None 
    LOAD_REPLAY_BUFFER : bool = True

    MAX_NOOP_STEPS : int = 20              

    TOTAL_FAMES : int = 30000000 
    FRAMES_BETWEEN_EVAL : int = 100000 
    MAX_EPISODE_LENGTH : int = 18000      
    EVAL_LENGTH : int  = 10000             
    UPDATE_FREQ : int = 10000              

    PRIORITY_SCALE : float  = 0.7              # How much the replay buffer should sample based on priorities. 0 = complete random samples, 1 = completely aligned with priorities
    CLIP_REWARD : bool  = True                # Any positive reward is +1, and negative reward is -1, 0 is unchanged

    UPDATE_FREQ : int = 4                   # Number of actions between gradient descent steps
    DISCOUNT_FACTOR : float  = 0.99            # Gamma, how much to discount future rewards

    BATCH_SIZE : int = 32                   # Batch size for training 
    MIN_REPLAY_BUFFER_SIZE = 50000    # The minimum size the replay buffer must be before we start to update the agent

    WRITE_TENSORBOARD : bool = True
    EVAL_LENGTH : int  = 10000               # Number of frames to evaluate for


class GameWrapperMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = [GameWrapper]

    def handle_input(
        self, data_type: Type[Any]
    ) -> Union[GameWrapper, GameWrapper]:
        """Reads a base sklearn label encoder from a pickle file."""
        super().handle_input(data_type)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "rb") as fid:
            clf = pickle.load(fid)
        return clf

    def handle_return(
        self, clf: Union[GameWrapper, GameWrapper],
    ) -> None:
        """Creates a pickle for a sklearn label encoder.
        Args:
            clf: A sklearn label encoder.
        """
        super().handle_return(clf)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "wb") as fid:
            pickle.dump(clf, fid)

class DQNMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = [tf.keras.Model]

    def handle_input(
        self, data_type: Type[Any]
    ) -> Union[tf.keras.Model, tf.keras.Model]:
        """Reads a base sklearn label encoder from a pickle file."""
        super().handle_input(data_type)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "rb") as fid:
            clf = pickle.load(fid)
        return clf

    def handle_return(
        self, clf: Union[tf.keras.Model, tf.keras.Model],
    ) -> None:
        """Creates a pickle for a sklearn label encoder.
        Args:
            clf: A sklearn label encoder.
        """
        super().handle_return(clf)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "wb") as fid:
            pickle.dump(clf, fid)

class ReplayBufferMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = [ReplayBuffer]

    def handle_input(
        self, data_type: Type[Any]
    ) -> Union[ReplayBuffer, ReplayBuffer]:
        """Reads a base sklearn label encoder from a pickle file."""
        super().handle_input(data_type)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "rb") as fid:
            clf = pickle.load(fid)
        return clf

    def handle_return(
        self, clf: Union[ReplayBuffer, ReplayBuffer],
    ) -> None:
        """Creates a pickle for a sklearn label encoder.
        Args:
            clf: A sklearn label encoder.
        """
        super().handle_return(clf)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "wb") as fid:
            pickle.dump(clf, fid)
    
class AgentMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = [Agent]

    def handle_input(
        self, data_type: Type[Any]
    ) -> Union[Agent, Agent]:
        """Reads a base sklearn label encoder from a pickle file."""
        super().handle_input(data_type)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "rb") as fid:
            clf = pickle.load(fid)
        return clf

    def handle_return(
        self, clf: Union[Agent, Agent],
    ) -> None:
        """Creates a pickle for a sklearn label encoder.
        Args:
            clf: A sklearn label encoder.
        """
        super().handle_return(clf)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "wb") as fid:
            pickle.dump(clf, fid)

def process_frame(frame, shape=(84, 84)):
    """Preprocesses a 210x160x3 frame to 84x84x1 grayscale

    Arguments:
        frame: The frame to process.  Must have values ranging from 0-255
    Returns:
        The processed frame
    """
    frame = frame.astype(np.uint8)  # cv2 requires np.uint8, other dtypes will not work

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = frame[34:34+160, :160]  # crop image
    frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
    frame = frame.reshape((*shape, 1))

    return frame


@step 
def GameWrap( 
    config : PreTrainingConfigs,
) -> GameWrapper:  
    GameWrapper_obj = GameWrapper(config.ENV_NAME, config.MAX_NOOP_STEPS) 
    return GameWrapper_obj 

@step 
def build_dqn(
    config : PreTrainingConfigs,  
    game_wrapper : GameWrapper
) -> Output(MAIN_DQN = tf.keras.Model
, target_dqn = tf.keras.Model): 


    MAIN_DQN = build_q_network(game_wrapper.env.action_space.n, config.LEARNING_RATE, input_shape=config.INPUT_SHAPE)
    TARGET_DQN = build_q_network(game_wrapper.env.action_space.n, input_shape=config.INPUT_SHAPE)

    return MAIN_DQN, TARGET_DQN

@step 
def replay_buffer( 
    config : PreTrainingConfigs, 
) -> ReplayBuffer:
    replay_buffer = ReplayBuffer(size=config.MEM_SIZE, input_shape=config.INPUT_SHAPE, use_per=config.USE_PER)
    return replay_buffer 

@step 
def agent( 
    config : PreTrainingConfigs, 
    game_wrapper : GameWrapper,
    replay_buffer : ReplayBuffer, 
    MAIN_DQN : tf.keras.Model,
    TARGET_DQN : tf.keras.Model
) -> Agent:
    agent = Agent(MAIN_DQN, TARGET_DQN, replay_buffer, game_wrapper.env.action_space.n, input_shape=config.INPUT_SHAPE, batch_size=config.BATCH_SIZE, use_per=config.USE_PER)
    return agent

@step  
def train_and_eval( 
    config : PreTrainingConfigs,
    agent : Agent,
) -> Output(frame_number = int, rewards = list, loss_list = list): 
    if config.LOAD_FROM is None:
        frame_number = 0
        rewards = []
        loss_list = []  
        return frame_number, rewards, loss_list
    else:
        print('Loading from', config.LOAD_FROM)
        meta = agent.load(config.LOAD_FROM, config.LOAD_REPLAY_BUFFER)

        # Apply information loaded from meta
        frame_number = meta['frame_number']
        rewards = meta['rewards']
        loss_list = meta['loss_list']
        return frame_number, rewards, loss_list 


class PreTrainingConfiguration(BaseStepConfig): 
    TOTAL_FAMES : int = 30000000 
    FRAMES_BETWEEN_EVAL : int = 100000 
    MAX_EPISODE_LENGTH : int = 18000      
    EVAL_LENGTH : int  = 10000             
    UPDATE_FREQ : int = 10000              

    PRIORITY_SCALE : float  = 0.7              # How much the replay buffer should sample based on priorities. 0 = complete random samples, 1 = completely aligned with priorities
    CLIP_REWARD : bool  = True                # Any positive reward is +1, and negative reward is -1, 0 is unchanged

    UPDATE_FREQ : int = 4                   # Number of actions between gradient descent steps
    DISCOUNT_FACTOR : float  = 0.99            # Gamma, how much to discount future rewards

    BATCH_SIZE : int = 32                   # Batch size for training 
    MIN_REPLAY_BUFFER_SIZE = 50000    # The minimum size the replay buffer must be before we start to update the agent

    WRITE_TENSORBOARD : bool = True
    EVAL_LENGTH : int  = 10000               # Number of frames to evaluate for

@step 
def train(
    config : PreTrainingConfigs, 
    game_wrapper : GameWrapper,
    loss_list : list, 
    rewards : list, 
    frame_number : int,
): 
    try: 
        writer = tf.summary.create_file_writer(config.TENSORBOARD_DIR)
        with writer.as_default():
            while frame_number < config.TOTAL_FRAMES:
                # Training

                epoch_frame = 0
                while epoch_frame < config.FRAMES_BETWEEN_EVAL:
                    start_time = time.time()
                    game_wrapper.reset()
                    life_lost = True
                    episode_reward_sum = 0
                    for _ in range(config.MAX_EPISODE_LENGTH):
                        # Get action
                        action = agent.get_action(frame_number, game_wrapper.state)

                        # Take step
                        processed_frame, reward, terminal, life_lost = game_wrapper.step(action)
                        frame_number += 1
                        epoch_frame += 1
                        episode_reward_sum += reward

                        # Add experience to replay memory
                        agent.add_experience(action=action,
                                            frame=processed_frame[:, :, 0],
                                            reward=reward, clip_reward=config.CLIP_REWARD,
                                            terminal=life_lost)

                        # Update agent
                        if frame_number % config.UPDATE_FREQ == 0 and agent.replay_buffer.count > config.MIN_REPLAY_BUFFER_SIZE:
                            loss, _ = agent.learn(config.BATCH_SIZE, gamma=config.DISCOUNT_FACTOR, frame_number=frame_number, priority_scale=config.PRIORITY_SCALE)
                            loss_list.append(loss)

                        # Update target network
                        if frame_number % config.UPDATE_FREQ == 0 and frame_number > config.MIN_REPLAY_BUFFER_SIZE:
                            agent.update_target_network()

                        # Break the loop when the game is over
                        if terminal:
                            terminal = False
                            break

                    rewards.append(episode_reward_sum)

                    # Output the progress every 10 games
                    if len(rewards) % 10 == 0:
                        # Write to TensorBoard
                        if config.WRITE_TENSORBOARD:
                            tf.summary.scalar('Reward', np.mean(rewards[-10:]), frame_number)
                            tf.summary.scalar('Loss', np.mean(loss_list[-100:]), frame_number)
                            writer.flush()

                        print(f'Game number: {str(len(rewards)).zfill(6)}  Frame number: {str(frame_number).zfill(8)}  Average reward: {np.mean(rewards[-10:]):0.1f}  Time taken: {(time.time() - start_time):.1f}s')

                # Evaluation every `FRAMES_BETWEEN_EVAL` frames
                terminal = True
                eval_rewards = []
                evaluate_frame_number = 0

                for _ in range(config.EVAL_LENGTH):
                    if terminal:
                        game_wrapper.reset(evaluation=True)
                        life_lost = True
                        episode_reward_sum = 0
                        terminal = False

                    # Breakout requires a "fire" action (action #1) to start the
                    # game each time a life is lost.
                    # Otherwise, the agent would sit around doing nothing.
                    action = 1 if life_lost else agent.get_action(frame_number, game_wrapper.state, evaluation=True)

                    # Step action
                    _, reward, terminal, life_lost = game_wrapper.step(action)
                    evaluate_frame_number += 1
                    episode_reward_sum += reward

                    # On game-over
                    if terminal:
                        eval_rewards.append(episode_reward_sum)

                if len(eval_rewards) > 0:
                    final_score = np.mean(eval_rewards)
                else:
                    # In case the game is longer than the number of frames allowed
                    final_score = episode_reward_sum
                # Print score and write to tensorboard
                print('Evaluation score:', final_score)
                if config.WRITE_TENSORBOARD:
                    tf.summary.scalar('Evaluation score', final_score, frame_number)
                    writer.flush()

                # Save model
                if len(rewards) > 300 and SAVE_PATH is not None:
                    agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number, rewards=rewards, loss_list=loss_list)
    except KeyboardInterrupt:
        print('\nTraining exited early.')
        writer.close()

        if SAVE_PATH is None:
            try:
                SAVE_PATH = input('Would you like to save the trained model? If so, type in a save path, otherwise, interrupt with ctrl+c. ')
            except KeyboardInterrupt:
                print('\nExiting...')

        if SAVE_PATH is not None:
            print('Saving...')
            agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number, rewards=rewards, loss_list=loss_list)
            print('Saved.')

@pipeline 
def train_pipeline( 
    GameWrap,
    build_dqn,
    replay_buffer,
    agent,
    train_and_eval,
    train
):  
    game_wrapper = GameWrap() 
    MAIN_DQN, target_dqn = build_dqn(game_wrapper) 
    replay_buffer = replay_buffer()
    agent = agent(game_wrapper,replay_buffer,MAIN_DQN, target_dqn)  
    frame_number, rewards, loss_list = train_and_eval(agent)
    train(game_wrapper,loss_list, rewards, frame_number) 


train_pipeline( 
    GameWrap = GameWrap().with_return_materializers(GameWrapperMaterializer), 
    build_dqn = build_dqn(),
    replay_buffer = replay_buffer(), 
    agent = agent(), 
    train_and_eval = train_and_eval(),
    train = train()).run() 

