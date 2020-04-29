import gym
from gym.utils import seeding
import numpy as np
import random
from collections import OrderedDict, deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class FOneAgent():
    
    MEMORY_SIZE = 10000
    BATCH_SIZE = 500
    EPOCHS = 10

    NORMALISATION = {
        'distance': 200,
        'lidar_angle': 30,
        'vVehicle': 50,
    }

    def __init__(self, env, epsilon_start = 1., epsilon_min=0.1, epsilon_decay_rate=0.9, gamma=0.95, seed=0):
        self.env = env
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_rate = epsilon_decay_rate
        self.gamma = gamma
        self.np_random = seeding.np_random(seed)
        self.memory = deque(maxlen=self.MEMORY_SIZE)
        self.model = self._init_model()

    def choose_action(self, state, episode):
        """
            Selects an action from the action space
            Decides whether to explore (random action) or exploit (best action) based on epsilon value
            Epsilon decays with time (i.e. episode) to a min value to increase exploitation over exploration as time goes on
        """
        if np.random.rand() < self.epsilon:
            print("Exploring...")
            # Then take random action
            action = self.env.action_space.sample()
            if np.random.rand() < self.epsilon/2:
                # 50/50ish chance of overwriting random sample and giving it gas to get us moving...
                action = 1
        else:
            # Process the lidar data into simpler format for now (in future can unleash the NN on the full lidar input)
            simplified_state = self._process_state(state)
            network_inputs = np.array(simplified_state).reshape((1,10))
            # Predict the best outcome
            q_value_pedals = self.model.predict(network_inputs)
            # print("Clear distance ahead: {}, road position: {},  angle: {}, vVehicle: {}, \npedal action: {}".format(simplified_state[0], 
            #                                                                                                         simplified_state[1], 
            #                                                                                                         simplified_state[2],
            #                                                                                                         simplified_state[3],
            #                                                                                                         q_value_pedals))
            # Return the action corresponding to the best selected option
            action = np.argmax(q_value_pedals)
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """ Add state, action and result of the action to the memory, for use in experience replay later """
        # print("Remember: {}\n{}\n{}\n{}\n{}".format(state, action, reward, self._process_state(next_state), done))
        self.memory.append((self._process_state(state), action, reward, self._process_state(next_state), done))
    
    def experience_replay(self):
        """ Train the model using experience replay to improve data efficiency """
        print("Training model...")
        if len(self.memory) < self.BATCH_SIZE:
            return
        batch = random.sample(self.memory, self.BATCH_SIZE)
        states = np.zeros((self.BATCH_SIZE, 10))
        q_values_batch = np.zeros((self.BATCH_SIZE, 5))
        for i_mem, memory in enumerate(batch):
        # for state, action, reward, state_next, terminal in batch:
            state, action, reward, state_next, terminal = memory
            q_update = reward
            if not terminal:
                # state_next_reshaped = state_next.reshape((1,4))
                state_next_reshaped = state_next.reshape((1,10))
                # print("State_next_reshaped shape: {}, value: {}".format(np.shape(state_next_reshaped),state_next_reshaped))
                q_update = (reward + self.gamma * np.amax(self.model.predict(state_next_reshaped)[0]))
            q_values = self.model.predict(state.reshape((1,10)))
            q_values[0][action] = q_update
            # Update the arrays
            q_values_batch[i_mem,:] = q_values[0]
            states[i_mem,:] = state
            # print("(Training) State: {}, q_values: {}".format(state, q_values))
            # self.model.fit(state.reshape((1,4)), q_values, verbose=1)
        self.model.fit(states, q_values_batch, batch_size=self.BATCH_SIZE, epochs=self.EPOCHS,verbose=1)
        # Reduce epsilon
        self._update_epsilon()

    def _init_model(self):
        #state_shape = self.env.observation_space.shape
        state_shape = (10,) # State shape 
        # Define the model
        model = Sequential([
            Dense(2, input_shape=state_shape, activation='relu'),
            # Dense(4, activation='relu'),
            # Dense(5, activation='softmax'),
            Dense(5, activation='linear'),
        ])
        # Compile the model
        model.compile(loss='mean_squared_error', optimizer='adam', metrics='mean_squared_error')
        # Return the model
        return model

    def _update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay_rate)

    def _process_state(self, state):
        """ 
            Convert the state to a reduced order description, and normalise the outputs.

            Outputs: 
                    Distance ahead - Average distance to the first obstacle that's directly ahead of the vehicle
                                    - Normalised between 0:1, where 1 = >=200m
                    Road position - % of road position from L/R, where 0 = touching left wall, 1= touching right wall
                                    - Normalised between 0:1
                    Angle to largest distance - The angle, relative to the vehicle's heading, of the front lidar with the longest distance
                                    - Normalised between -1:1, where +/-1 = +/-30 deg (i.e. full field of view)
                                    - Normalised between 0-1, where 1 = 50m/s
        """
        # distance_ahead = self._get_distance_clear_ahead(state['lidar']['front'])
        # road_position = self._get_position_on_road(state['lidar']['left'], state['lidar']['right'])
        # angle_to_longest_distance = self._get_angle_to_longest_distance(state['lidar']['front'])
        lidars = np.array([*state['lidar']['front'],*state['lidar']['left'],*state['lidar']['right']])
        vVehicle = self._get_vVehicle(state['velocity'])
        # Normalise the values
        # n_distance_ahead = distance_ahead / self.NORMALISATION['distance']
        # n_angle_to_longest_distance = angle_to_longest_distance / self.NORMALISATION['lidar_angle']
        normLidars = lidars / self.NORMALISATION['distance']
        nVVehicle = vVehicle / self.NORMALISATION['vVehicle']
        # return np.array([n_distance_ahead, road_position, n_angle_to_longest_distance, nVVehicle])
        return np.append(normLidars, nVVehicle)
    
    def _get_distance_clear_ahead(self, lidars_front):
        """ Find middle lidar ray(s)' distance straight ahead """
        return self._get_central_distances(lidars_front)
    
    def _get_position_on_road(self, lidars_left, lidars_right):
        """ Find distance to left & right walls, to find % position on road (0 is left, 1 is right) """
        distance_clear_left = self._get_central_distances(lidars_left)
        distance_clear_right = self._get_central_distances(lidars_right)
        return distance_clear_left / (distance_clear_left + distance_clear_right)

    def _get_angle_to_longest_distance(self, lidars_front):
        """
            Calculate the angle of the lidar which has the highest distance
            TODO: Move angle information into Lidar data so that angle calculation isn't hard coded here - allows for changing lidar settings
        """
        i_lidar_max = np.argmax(lidars_front)
        a0 = 0.0
        aFov = 60 * np.pi / 180
        NRays = 20
        aRad = a0 - aFov / 2 + i_lidar_max * aFov / NRays
        aDeg = aRad * 180 / np.pi
        return aDeg

    def get_road_alignment(self, state):
        """ Find the angle of the vehicle relative to the track limits """
        # TODO: fill this in - could start by just finding the minimum in the plot of lidar distance vs angle
        # Use the difference between 

        pass

    def _get_vVehicle(self, v):
        return np.sqrt(v[0]**2 + v[1]**2)

    def _get_central_distances(self, lidars, min_lidar_count = 3):
        """ 
            Find the average distance of the central lidars in the set passed in
            Will use either the min_lidar_count or min_lidar_count + 1 depending on whether there are odd or even lidar rays in the set respectively
        """
        # Find middle lidar ray(s)
        i_lidar_middle = (len(lidars) - 1) / 2
        # Find threshold for which rays to use
        i_lidar_threshold_delta = min_lidar_count / 2 + 0.1
        # Pick the central 3 rays (if there's an odd number), or 4 if it's even - use the 1.6 to filter appropriately
        central_lidars = [lidar for (i, lidar) in enumerate(lidars) if i > (i_lidar_middle - i_lidar_threshold_delta) and i < (i_lidar_middle + i_lidar_threshold_delta)]
        return np.mean(central_lidars)
   
