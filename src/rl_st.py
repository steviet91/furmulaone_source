import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
import os
from collections import deque
import random
import time
import numpy as np
import sys

# #########
# Q TABLE #
# #########
class QTableAgent(object):
    """
        ST's implementation of a Q-Table
    """

    _LEARNING_RATE = 0.1
    _DISCOUNT = 0.95
    _DISCRETE_OS_LEN = 20

    def __init__(self, num_actions=3, observation_space_low=[], observation_space_high=[]):
        """
            Initialise the agent
        """
        self.observation_space_low = observation_space_low
        self.observation_space_high = observation_space_high

        self._DISCRETE_OS_SIZE = [self._DISCRETE_OS_LEN] * len(self.observation_space_high)
        self.discrete_os_win_size = (self.observation_space_high - self.observation_space_low) / self._DISCRETE_OS_SIZE

        self.table = np.random.uniform(low=-2, high=0, size=(self._DISCRETE_OS_SIZE + [num_actions]))

    def get_discrete_state(self, state):
        """
            Convert the state into a discrete state
        """
        discrete_state = (state - self.observation_space_low) / self.discrete_os_win_size
        return tuple(discrete_state.astype(np.int))

    def update_q_table(self, discrete_state, new_discrete_state, action, reward):
        """
            Update the q table
        """
        # find the max possible q for the new discrete state
        max_future_q = np.max(self.table[new_discrete_state])

        # get the current q
        current_q = self.table[discrete_state + (action,)]

        new_q = (1 - self._LEARNING_RATE) * current_q + self._LEARNING_RATE * (reward + self._DISCOUNT * max_future_q)

        self.table[discrete_state + (action, )] = new_q


# ###########
# DQN AGENT #
# ###########
class DQNAgent(object):
    """
        ST's implementation of a DQN agent
    """
    _REPLAY_MEMORY_SIZE = 10_000
    _MIN_REPLAY_MEMORY_SIZE = 1000
    _MINIBATCH_SIZE = 500
    _DISCOUNT = 0.95
    _UPDATE_TARGET_THRESH = 5

    def __init__(self, name='st_dqn', num_inputs=1, num_actions=3, hidden_layer_lens=[], activation='relu'):
        """
            Initialise the object
        """
        # set some path information
        self.module_path = os.path.dirname(os.path.abspath(__file__))
        self.save_path = self.module_path + '/../data/rl_models/'
        self.log_path = self.module_path + '/../data/rl_logs/'

        # save the arguments
        if len(hidden_layer_lens) == 0:
            self.h_layer_lens = [ (num_inputs + num_actions) //2 ] # default to singel layer, with NNeurons equal to the average of inputs vs outputs
        else:
            self.h_layer_lens = hidden_layer_lens
        self.num_inputs = num_inputs
        self.activation = activation
        self.num_actions = num_actions
        self.name = name

        # initialise the main network
        self.model = self.create_model()

        # initialise the tart network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # initialise an array to store the last n steps for training
        self.replay_memory = deque(maxlen=self._REPLAY_MEMORY_SIZE)

        # counter to determine when to update the target model with the main model weights
        self.target_update_counter = 0

        # tensorboard
        if False:
            self.tensorboard = FurmulaTensorBoard(log_dir=self.log_path + f'{int(time.time())}-{self.name}')
        else:
            self.tensorboard = None

    def create_model(self):
        """
            Creates a keras NN model
        """
        model = keras.Sequential()

        # add the hidden layers
        for i,h in enumerate(self.h_layer_lens):
            if i == 0:
                model.add(Dense(h, input_shape=(self.num_inputs, ), activation=self.activation))
            else:
                model.add(Dense(h))

        # add the output layer
        model.add(Dense(self.num_actions, activation='linear'))

        # complile the model
        model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])

        # pass back the model
        return model

    def update_replay_memory(self, transition):
        """
            Updates the replay memory with the current transition
        """
        self.replay_memory.append(transition)
        if len(self.replay_memory) > self._REPLAY_MEMORY_SIZE:
            print('MEMORY TOO BIG')

    def get_qs(self, state):
        """
            Queries the main network for Q values given current observations space
        """
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def train(self, terminal_state, step):
        """
            Trains the main network every step during episode
        """
        # only start training if a vertain nmbers of samples is already saved
        if len(self.replay_memory) < self._MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from the memory replay table
        minibatch = random.sample(self.replay_memory, self._MINIBATCH_SIZE)

        # Get the current states from the minimbatch, query the main model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from the minibatch, then query NN model for Q values
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for idx, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # if not a terminal state, get new q from future states, otherwise set to 0
            if not done:
                max_future_q = np.max(future_qs_list[idx])
                new_q = reward + self._DISCOUNT * max_future_q
            else:
                new_q = reward

            # update Q value for given state
            current_qs = current_qs_list[idx]
            current_qs[action] = new_q

            # add to the training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        if self.tensorboard:
            self.model.fit(np.array(X), np.array(y), batch_size=self._MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        else:
            self.model.fit(np.array(X), np.array(y), batch_size=self._MINIBATCH_SIZE, verbose=0, shuffle=False)
        # Update the target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # if the counter reaches a set value, update the target network with the weights of the main
        if self.target_update_counter > self._UPDATE_TARGET_THRESH:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

# ####################
# CUSTOM TensorBoard #
# ####################
class FurmulaTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir
        self.writer.set_as_default()
        self.step = 0

    def update_stats(self,**stats):
        for k,v in stats.items():
            tf.summary.scalar(k, data=v, step=self.step)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass


if __name__ == "__main__":
    tb = FurmulaTensorBoard(log_dir='C:/furmulaone_source/data/rl_logs/test')
    tb.update_states(reward_avg=0.5, reward_min=0.5, reward_max=0.5)
