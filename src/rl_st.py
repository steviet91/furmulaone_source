import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from src.nn import NeuralNetwork
import os
from collections import deque
import random
import time
from time import sleep
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
    _TRAIN_PROC_SLEEP_DUR = 0.01
    _NUM_EPOCHS = 1

    def __init__(self, name='st_dqn', num_inputs=1, num_actions=3, hidden_layer_lens=[], activation='relu', load_file=None, custom_sim_model=False):
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
        self.custom_sim_model = custom_sim_model

        # initialise an array to store the last n steps for training
        self.replay_memory = deque(maxlen=self._REPLAY_MEMORY_SIZE)

        if not load_file:
            # initialise the main network
            self.train_model = self.create_model()
        else:
            self.train_model = keras.models.load_model(self.save_path + load_file)
            print(f'Model loaded from file: {load_file}')

        # initialise the tart network
        self.sim_model = self.create_model(use_custom_nn=custom_sim_model)
        self.sim_model.set_weights(self.train_model.get_weights())


    def create_model(self, use_custom_nn=False):
        """
            Creates a keras NN model or a custom NN (should only be used for inference)
        """
        if not use_custom_nn:
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
        else:
            model = NeuralNetwork(self.num_inputs, self.num_actions, self.h_layer_lens, [self.activation]*len(self.h_layer_lens), 'linear', dtype=np.float32)
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
        if self.custom_sim_model:
            return self.sim_model.predict(state)
        else:
            return self.sim_model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def train(self):
        """
            Trains the main network every step during episode
        """
        # only start training if a vertain nmbers of samples is already saved
        if len(self.replay_memory) < self._MIN_REPLAY_MEMORY_SIZE:
            return
        else:
            # Get the current states from the minimbatch, query the main model for Q values
            current_states = np.array([transition[0] for transition in self.replay_memory])
            current_qs_list = self.train_model.predict(current_states)

            # Get future states, then query NN model for Q values
            new_current_states = np.array([transition[3] for transition in self.replay_memory])
            future_qs_list = self.train_model.predict(new_current_states)

            X = []
            y = []

            for idx, (current_state, action, reward, new_current_state, done) in enumerate(self.replay_memory):

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

            self.train_model.fit(np.array(X), np.array(y), epochs=self._NUM_EPOCHS, batch_size=self._MINIBATCH_SIZE, verbose=1, shuffle=True)
            self.sim_model.set_weights(self.train_model.get_weights())


if __name__ == "__main__":
    a = DQNAgent(num_inputs=12, num_actions=4, use_custom_inference_model=True)

    for i in range(20):
        inputs = np.random.rand(12)
        print(a.train_model.predict(np.array(inputs).reshape(-1, *inputs.shape))[0]-a.sim_model.predict(inputs))
    n = 1000
    t_k = []
    t_m = []
    for i in range(n):
        inputs = np.random.rand(12)
        t = time.time()
        a.train_model.predict(np.array(inputs).reshape(-1, *inputs.shape))[0]
        t_k.append(time.time()-t)
        t = time.time()
        a.sim_model.predict(inputs)
        t_m.append(time.time()-t)
    print(f'Keras: {np.mean(np.array(t_k)):.5f}s, Custom: {np.mean(np.array(t_m)):.5f}s')
