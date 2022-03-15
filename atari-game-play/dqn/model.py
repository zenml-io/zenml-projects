import random
import numpy as np
import numpy as np
import cv2

import random
import os
import json
import time

import gym

env = gym.envs.make("BreakoutDeterministic-v4")

import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import (
    Add,
    Conv2D,
    Dense,
    Flatten,
    Input,
    Lambda,
    Subtract,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop

from typing import Any, Type, Union, List, Tuple
import pickle
from tensorflow import keras


def process_frame(
    frame: np.ndarray, shape: Tuple[int, int] = (84, 84)
) -> np.ndarray:
    """
    Preprocesses a 210x160x3 frame to 84x84x1 grayscale

    Args:
        frame: The frame to process, It's the frame of our environment that we are using.  Must have values ranging from 0-255
        shape: The shape of the frame to return.  Defaults to 84x84x1
    Returns:
        The processed frame
    """
    frame = frame.astype(
        np.uint8
    )  # cv2 requires np.uint8, other dtypes will not work

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = frame[34 : 34 + 160, :160]  # crop image
    frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
    frame = frame.reshape((*shape, 1))

    return frame


def build_q_network(
    n_actions: int,
    learning_rate: float = 0.00001,
    input_shape: Tuple[int, int] = (84, 84),
    history_length: int = 4,
):
    """Builds a dueling DQN as a Keras model
    Args:
        n_actions: Number of possible action the agent can take
        learning_rate: Learning rate
        input_shape: Shape of the preprocessed frame the model sees
        history_length: Number of historical frames the agent can see
    Returns:
        A compiled Keras model
    """

    model_input = Input(shape=(input_shape[0], input_shape[1], history_length))
    x = Lambda(lambda layer: layer / 255)(model_input)  # normalize by 255

    x = Conv2D(
        32,
        (8, 8),
        strides=4,
        kernel_initializer=VarianceScaling(scale=2.0),
        activation="relu",
        use_bias=False,
    )(x)
    x = Conv2D(
        64,
        (4, 4),
        strides=2,
        kernel_initializer=VarianceScaling(scale=2.0),
        activation="relu",
        use_bias=False,
    )(x)
    x = Conv2D(
        64,
        (3, 3),
        strides=1,
        kernel_initializer=VarianceScaling(scale=2.0),
        activation="relu",
        use_bias=False,
    )(x)
    x = Conv2D(
        1024,
        (7, 7),
        strides=1,
        kernel_initializer=VarianceScaling(scale=2.0),
        activation="relu",
        use_bias=False,
    )(x)

    # Split into value and advantage streams
    val_stream, adv_stream = Lambda(lambda w: tf.split(w, 2, 3))(
        x
    )  # custom splitting layer

    val_stream = Flatten()(val_stream)
    val = Dense(1, kernel_initializer=VarianceScaling(scale=2.0))(val_stream)

    adv_stream = Flatten()(adv_stream)
    adv = Dense(n_actions, kernel_initializer=VarianceScaling(scale=2.0))(
        adv_stream
    )

    # Combine streams into Q-Values
    reduce_mean = Lambda(
        lambda w: tf.reduce_mean(w, axis=1, keepdims=True)
    )  # custom layer for reduce mean

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

    def reset(self, evaluation: bool = False):
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
        self.state = np.repeat(
            process_frame(self.frame), self.history_length, axis=2
        )

    def step(
        self, action: int, render_mode=None
    ) -> Union[bool, np.ndarray, bool, Any]:
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
        if info["lives"] < self.last_lives:
            life_lost = True
        else:
            life_lost = terminal
        self.last_lives = info["lives"]

        processed_frame = process_frame(new_frame)
        self.state = np.append(self.state[:, :, 1:], processed_frame, axis=2)

        if render_mode == "rgb_array":
            return (
                processed_frame,
                reward,
                terminal,
                life_lost,
                self.env.render(render_mode),
            )
        elif render_mode == "human":
            self.env.render()

        # print(type(reward))
        # print(type(processed_frame))
        # print(type(terminal))
        # print(type(life_lost))

        return processed_frame, reward, terminal, life_lost


class Agent(object):
    """Implements a standard DDDQN agent"""

    def __init__(
        self,
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
        use_per=True,
    ):
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
        self.slope = (
            -(self.eps_initial - self.eps_final) / self.eps_annealing_frames
        )
        self.intercept = (
            self.eps_initial - self.slope * self.replay_buffer_start_size
        )
        self.slope_2 = -(self.eps_final - self.eps_final_frame) / (
            self.max_frames
            - self.eps_annealing_frames
            - self.replay_buffer_start_size
        )
        self.intercept_2 = (
            self.eps_final_frame - self.slope_2 * self.max_frames
        )

        # DQN
        self.DQN = dqn
        self.target_dqn = target_dqn

    def calc_epsilon(
        self, frame_number: np.ndarray, evaluation: bool = False
    ) -> float:
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
        elif (
            frame_number >= self.replay_buffer_start_size
            and frame_number
            < self.replay_buffer_start_size + self.eps_annealing_frames
        ):
            return self.slope * frame_number + self.intercept
        elif (
            frame_number
            >= self.replay_buffer_start_size + self.eps_annealing_frames
        ):
            return self.slope_2 * frame_number + self.intercept_2

    def get_action(
        self,
        frame_number: np.ndarray,
        state: np.ndarray,
        evaluation: bool = False,
    ) -> int:
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
        q_vals = self.DQN.predict(
            state.reshape(
                (
                    -1,
                    self.input_shape[0],
                    self.input_shape[1],
                    self.history_length,
                )
            )
        )[0]
        return q_vals.argmax()

    def get_intermediate_representation(
        self,
        state: np.ndarray,
        layer_names: str = None,
        stack_state: bool = True,
    ) -> np.ndarray:
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
            layers = [
                self.DQN.get_layer(name=layer_name).output
                for layer_name in layer_names
            ]
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
        return temp_model.predict(
            state.reshape(
                (
                    -1,
                    self.input_shape[0],
                    self.input_shape[1],
                    self.history_length,
                )
            )
        )

    def update_target_network(self) -> None:
        """Update the target Q network"""
        self.target_dqn.set_weights(self.DQN.get_weights())

    def add_experience(
        self,
        action: int,
        frame: np.ndarray,
        reward: np.ndarray,
        terminal: bool,
        clip_reward: bool = True,
    ):
        """Wrapper function for adding an experience to the Agent's replay buffer"""
        self.replay_buffer.add_experience(
            action, frame, reward, terminal, clip_reward
        )

    def learn(
        self,
        batch_size: int,
        gamma: float,
        frame_number: np.ndarray,
        priority_scale: float = 1.0,
    ):
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
            (
                (states, actions, rewards, new_states, terminal_flags),
                importance,
                indices,
            ) = self.replay_buffer.get_minibatch(
                batch_size=self.batch_size, priority_scale=priority_scale
            )
            importance = importance ** (1 - self.calc_epsilon(frame_number))
        else:
            (
                states,
                actions,
                rewards,
                new_states,
                terminal_flags,
            ) = self.replay_buffer.get_minibatch(
                batch_size=self.batch_size, priority_scale=priority_scale
            )

        # Main DQN estimates best action in new states
        arg_q_max = self.DQN.predict(new_states).argmax(axis=1)

        # Target DQN estimates q-vals for new states
        future_q_vals = self.target_dqn.predict(new_states)
        double_q = future_q_vals[range(batch_size), arg_q_max]

        # Calculate targets (bellman equation)
        target_q = rewards + (gamma * double_q * (1 - terminal_flags))

        # Use targets to calculate loss (and use loss to calculate gradients)
        with tf.GradientTape() as tape:
            q_values = self.DQN(states)

            one_hot_actions = tf.keras.utils.to_categorical(
                actions, self.n_actions, dtype=np.float32
            )  # using tf.one_hot causes strange errors
            Q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)

            error = Q - target_q
            loss = tf.keras.losses.Huber()(target_q, Q)

            if self.use_per:
                # Multiply the loss by importance, so that the gradient is also scaled.
                # The importance scale reduces bias against situataions that are sampled
                # more frequently.
                loss = tf.reduce_mean(loss * importance)

        model_gradients = tape.gradient(loss, self.DQN.trainable_variables)
        self.DQN.optimizer.apply_gradients(
            zip(model_gradients, self.DQN.trainable_variables)
        )

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
        self.DQN.save(folder_name + "/dqn.h5")
        self.target_dqn.save(folder_name + "/target_dqn.h5")

        # Save replay buffer
        self.replay_buffer.save(folder_name + "/replay-buffer")

        # Save meta
        with open(folder_name + "/meta.json", "w+") as f:
            f.write(
                json.dumps(
                    {
                        **{
                            "buff_count": self.replay_buffer.count,
                            "buff_curr": self.replay_buffer.current,
                        },
                        **kwargs,
                    }
                )
            )  # save replay_buffer information and any other information

    def load(self, folder_name, load_replay_buffer=True):
        """Load a previously saved Agent from a folder
        Arguments:
            folder_name: Folder from which to load the Agent
        Returns:
            All other saved attributes, e.g., frame number
        """

        if not os.path.isdir(folder_name):
            raise ValueError(f"{folder_name} is not a valid directory")

        # Load DQNs
        self.DQN = tf.keras.models.load_model(folder_name + "/dqn.h5")
        self.target_dqn = tf.keras.models.load_model(
            folder_name + "/target_dqn.h5"
        )
        self.optimizer = self.DQN.optimizer

        # Load replay buffer
        if load_replay_buffer:
            self.replay_buffer.load(folder_name + "/replay-buffer")

        # Load meta
        with open(folder_name + "/meta.json", "r") as f:
            meta = json.load(f)

        if load_replay_buffer:
            self.replay_buffer.count = meta["buff_count"]
            self.replay_buffer.current = meta["buff_curr"]

        del (
            meta["buff_count"],
            meta["buff_curr"],
        )  # we don't want to return this information
        return meta


class ReplayBuffer:
    """Replay Buffer to store transitions.
    This implementation was heavily inspired by Fabio M. Graetz's replay buffer
    here: https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb"""

    def __init__(
        self,
        size=1000000,
        input_shape=(84, 84),
        history_length=4,
        use_per=True,
    ):
        """
        Arguments:
            size: Integer, Number of stored transitions
            input_shape: Shape of the preprocessed frame
            history_length: Integer, Number of frames stacked together to create a state for the agent
            use_per: Use PER instead of classic experience replay
        """
        self.size = size
        self.input_shape = input_shape
        self.history_length = history_length
        self.count = (
            0  # total index of memory written to, always less than self.size
        )
        self.current = 0  # index to write to

        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty(
            (self.size, self.input_shape[0], self.input_shape[1]),
            dtype=np.uint8,
        )
        self.terminal_flags = np.empty(self.size, dtype=np.bool)
        self.priorities = np.zeros(self.size, dtype=np.float32)

        self.use_per = use_per

    def add_experience(
        self,
        action: int,
        frame: np.ndarray,
        reward: np.ndarray,
        terminal: bool,
        clip_reward: bool = True,
    ):
        """Saves a transition to the replay buffer
        Arguments:
            action: An integer between 0 and env.action_space.n - 1
                determining the action the agent perfomed
            frame: A (84, 84, 1) frame of the game in grayscale
            reward: A float determining the reward the agend received for performing an action
            terminal: A bool stating whether the episode terminated
        """
        if frame.shape != self.input_shape:
            raise ValueError("Dimension of frame is wrong!")

        if clip_reward:
            reward = np.sign(reward)

        # Write memory
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.priorities[self.current] = max(
            self.priorities.max(), 1
        )  # make the most recent experience important
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size

    def get_minibatch(self, batch_size: int = 32, priority_scale: float = 0.0):
        """Returns a minibatch of self.batch_size = 32 transitions
        Arguments:
            batch_size: How many samples to return
            priority_scale: How much to weight priorities. 0 = completely random, 1 = completely based on priority
        Returns:
            A tuple of states, actions, rewards, new_states, and terminals
            If use_per is True:
                An array describing the importance of transition. Used for scaling gradient steps.
                An array of each index that was sampled
        """

        if self.count < self.history_length:
            raise ValueError("Not enough memories to get a minibatch")

        # Get sampling probabilities from priority list
        if self.use_per:
            scaled_priorities = (
                self.priorities[self.history_length : self.count - 1]
                ** priority_scale
            )
            sample_probabilities = scaled_priorities / sum(scaled_priorities)

        # Get a list of valid indices
        indices = []
        for i in range(batch_size):
            while True:
                # Get a random number from history_length to maximum frame written with probabilities based on priority weights
                if self.use_per:
                    index = np.random.choice(
                        np.arange(self.history_length, self.count - 1),
                        p=sample_probabilities,
                    )
                else:
                    index = random.randint(self.history_length, self.count - 1)

                # We check that all frames are from same episode with the two following if statements.  If either are True, the index is invalid.
                if (
                    index >= self.current
                    and index - self.history_length <= self.current
                ):
                    continue
                if self.terminal_flags[
                    index - self.history_length : index
                ].any():
                    continue
                break
            indices.append(index)

        # Retrieve states from memory
        states = []
        new_states = []
        for idx in indices:
            states.append(self.frames[idx - self.history_length : idx, ...])
            new_states.append(
                self.frames[idx - self.history_length + 1 : idx + 1, ...]
            )

        states = np.transpose(np.asarray(states), axes=(0, 2, 3, 1))
        new_states = np.transpose(np.asarray(new_states), axes=(0, 2, 3, 1))

        if self.use_per:
            # Get importance weights from probabilities calculated earlier
            importance = (
                1
                / self.count
                * 1
                / sample_probabilities[
                    [index - self.history_length for index in indices]
                ]
            )
            importance = importance / importance.max()

            return (
                (
                    states,
                    self.actions[indices],
                    self.rewards[indices],
                    new_states,
                    self.terminal_flags[indices],
                ),
                importance,
                indices,
            )
        else:
            return (
                states,
                self.actions[indices],
                self.rewards[indices],
                new_states,
                self.terminal_flags[indices],
            )

    def set_priorities(self, indices, errors, offset=0.1):
        """Update priorities for PER
        Arguments:
            indices: Indices to update
            errors: For each index, the error between the target Q-vals and the predicted Q-vals
        """
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset

    def save(self, folder_name):
        """Save the replay buffer to a folder"""

        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        np.save(folder_name + "/actions.npy", self.actions)
        np.save(folder_name + "/frames.npy", self.frames)
        np.save(folder_name + "/rewards.npy", self.rewards)
        np.save(folder_name + "/terminal_flags.npy", self.terminal_flags)

    def load(self, folder_name):
        """Loads the replay buffer from a folder"""
        self.actions = np.load(folder_name + "/actions.npy")
        self.frames = np.load(folder_name + "/frames.npy")
        self.rewards = np.load(folder_name + "/rewards.npy")
        self.terminal_flags = np.load(folder_name + "/terminal_flags.npy")