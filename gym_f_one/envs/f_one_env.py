import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from collections import OrderedDict

class FOneEnv(gym.Env):
    """ Furmula One Environment that follows OpenAI Gym Interface """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(FOneEnv, self).__init__()
        # Rewards
        self.rewards = {
            'time': -1,
            'distance': 1,
            'speed':2,
            'collision': -10
        }
        # Limits of our spaces
        self.action_limits = {
            'pedals': {
                'min':-1,
                'max':1
            },
            'steering': {
                'min':-360,
                'max':360
            }
        }
        self.obs_limits = {
            'position': {
                'min':-300,
                'max':300
            },
            'velocity': {
                'min':-100,
                'max':100
            },
            'lidar': {
                'min':0,
                'max':200,
                'count':3
            }
        }
        # Define action and observation space
        # They must be gym.spaces objects
        # Discrete, single option at a time (either steering or pedals)
        self.action_space = spaces.Discrete(5)
        self.actions = [{
            'value': 0,
            'description': 'Doing nothing',
            'demands': {'rThrottlePedalDemanded':  0,
                        'rBrakePedalDemanded': 0,
                        'aSteeringWheelDemanded': 0,
                        'aLidarFront': 0,
                        'aLidarLeft': 0,
                        'aLidarRight': 0
                    },
            }, {
            'value': 1,
            'description': 'Accelerating',
            'demands': {'rThrottlePedalDemanded':  1,
                        'rBrakePedalDemanded': 0,
                        'aSteeringWheelDemanded': 0,
                        'aLidarFront': 0,
                        'aLidarLeft': 0,
                        'aLidarRight': 0
                    },
            }, {
            'value': 2,
            'description': 'Braking',
            'demands': {'rThrottlePedalDemanded':  0,
                        'rBrakePedalDemanded': 1,
                        'aSteeringWheelDemanded': 0,
                        'aLidarFront': 0,
                        'aLidarLeft': 0,
                        'aLidarRight': 0
                    },
            }, {
            'value': 3,
            'description': 'Turning Left',
            'demands': {'rThrottlePedalDemanded':  0,
                        'rBrakePedalDemanded': 0,
                        'aSteeringWheelDemanded': -360,
                        'aLidarFront': 0,
                        'aLidarLeft': 0,
                        'aLidarRight': 0
                    },
            }, {
            'value': 4,
            'description': 'Turning Right',
            'demands': {'rThrottlePedalDemanded':  0,
                        'rBrakePedalDemanded': 0,
                        'aSteeringWheelDemanded': 360,
                        'aLidarFront': 0,
                        'aLidarLeft': 0,
                        'aLidarRight': 0
                    },
            }, 
        ]
        # Continuous action spaces - not for use in Q style structure
        # self.action_space = spaces.Dict({"pedals":spaces.Box(
        #                                                 low=self.action_limits['pedals']['min'],
        #                                                 high=self.action_limits['pedals']['max'],
        #                                                 shape=(1,),
        #                                                 # dtype=np.float32
        #                                                 ),
        #                                 "steering":spaces.Box(
        #                                                 low=self.action_limits['steering']['min'],
        #                                                 high=self.action_limits['steering']['max'],
        #                                                 shape=(1,),
        #                                                 # dtype=np.uint8samples = [ped.sample for i in range(200)]
        #                                                 ),
        # })
        # Observation space
        self.observation_space = spaces.Dict({"position": spaces.Box(low=self.obs_limits['position']['min'], high=self.obs_limits['position']['max'], shape=(2,)),
                                                "velocity":spaces.Box(low=self.obs_limits['velocity']['min'], high=self.obs_limits['velocity']['max'], shape=(2,)),
                                                "lidar": spaces.Dict({
                                                    "front":spaces.Box(low=self.obs_limits['lidar']['min'], high=self.obs_limits['lidar']['max'], shape=(self.obs_limits['lidar']['count'],)),
                                                    "left":spaces.Box(low=self.obs_limits['lidar']['min'], high=self.obs_limits['lidar']['max'], shape=(self.obs_limits['lidar']['count'],)),
                                                    "right":spaces.Box(low=self.obs_limits['lidar']['min'], high=self.obs_limits['lidar']['max'], shape=(self.obs_limits['lidar']['count'],)),
                                                })
                                            })

        self.set_timestep()
        self.seed()

    def set_vehicle(self, vehicle):
        self.veh = vehicle

    def set_timestep(self, timestep=0.01):
        self.timestep = timestep

    def set_random_seed(self, random_seed):
        self.seed(random_seed)

    def set_visualisation(self, vis):
        self.vis = vis

    def step(self, action):
        # Take the action
        self._take_action(action)

        # Get the reward
        reward, info = self._get_reward()

        # Return the jazz
        return self.state, reward, self.veh.bHasCollided, info

    def reset(self):
        self.veh.reset_states()
        self.veh.reset_vehicle_position()
        self.vis.reset_camera()
        self.update_state()
        return self.state

    def render(self, mode='human', close=False):
        # draw the visualisation
        self.vis.draw()

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def update_state(self):
        # Get the position from the vehicle and store it in our state variable
        self.state = OrderedDict([
            ("position", self.veh.posVehicle),
            ("velocity", np.array([self.veh.vxVehicle, self.veh.vxVehicle])),
            ("lidar", OrderedDict([
                ("front", self.veh.lidar_front.collision_array),
                ("left", self.veh.lidar_left.collision_array),
                ("right", self.veh.lidar_right.collision_array),
                ])
            )
        ])

    def _take_action(self, action):
        input_data = self._get_demands(action)
        # print('Translated Action: {}'.format(input_data))

        # lidar angle
        aLidarFront = input_data['aLidarFront']
        aLidarLeft = input_data['aLidarLeft']
        aLidarRight = input_data['aLidarRight']
        
        self.veh.update(input_data['rThrottlePedalDemanded'], input_data['rBrakePedalDemanded'], input_data['aSteeringWheelDemanded'],
                    aRotFront=aLidarFront, aRotL=aLidarLeft, aRotR=aLidarRight)
        # Update the 'state'
        self.update_state()


    def _get_demands(self, action):
        # print("Action: {}".format(action))
        demand = [demand for demand in self.actions if demand['value'] == action][0]
        return demand['demands']

    def _get_reward(self):
        # Calculate the reward
        rewardTime = self.rewards['time'] * self.timestep
        rewardDistance = self.rewards['distance'] * (self.veh.vVehicle * self.timestep)
        rewardSpeed = self.rewards['speed'] * self.veh.vVehicle 
        rewardCollision = self.rewards['collision'] * self.veh.bHasCollided
        reward = rewardTime + rewardDistance + rewardCollision
        infoString = 'Rewards:\n\tTimestep: {}\n\tDistance: {}\n\tCollision: {}'.format(rewardTime,rewardDistance,rewardCollision)
        return reward, infoString
