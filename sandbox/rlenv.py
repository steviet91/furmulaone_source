"""
Environment to use the Furmula One sim with OpenAI Gym
"""
import gym
import gym_f_one
from src.network import VehicleOutputsSend, DriverInputsRecv, VehicleOutputsRecv, DriverInputsSend
from src.vehicle import Vehicle
from src.visualiser import Vis
from src.track import TrackHandler
from time import sleep
import numpy as np
from gym.utils import seeding
from collections import OrderedDict
from sandbox.rlagent import FOneAgent

timestep = 0.1
episodes = 500

def main():
    # instantiate the objects
    track = TrackHandler('dodec_track')
    veh = Vehicle(1, track, 60, 60 ,60, task_rate=timestep, auto_reset=False)
    vis = Vis(track, veh, use_camera_spring=False)

    # Create the environment
    env = gym.make('f_one-v0')
    # Set the variables it needs
    env.set_vehicle(veh)
    env.set_visualisation(vis)
    state = env.reset()

    # Agent instance
    agent = FOneAgent(env)

    run_game = True

    # Initialise the episode counter
    ep = 0
    # Initialise variables to store reward, state & actions
    states = []
    actions = []
    rewards = []

    ep_reward = 0

    while run_game:
        # Get 'state' information for the agent
        # Check user inputs
        action = agent.choose_action(state, ep)

        # Update the model
        new_state, reward, finished, info = env.step(action)
        ep_reward += reward
        # print('\tAction: {}\n\tState:{}\n\tNew State:{}\n\tReward: {}'.format(action, state, new_state, reward))
        # print('\tAction: {}\n\tReward: {}'.format(action, reward))

        # Let the agent know what happened
        agent.remember(state, action, reward, new_state, finished)

        # draw the visualisation
        env.render()

        # Make the new state the current one
        state = new_state

        # Update lists of actions, etc.
        # states.append(state)
        # actions.append(action)
        # rewards.append(reward)

        if finished:
            if ep < episodes:
                print("Episode {} completed. Total reward: {} ".format(ep, ep_reward))
                ep_reward = 0
                # Make it retrain
                if ep % 10 == 0:
                    agent.experience_replay()
                ep += 1
                veh.reset_vehicle_position()
                veh.reset_states()
            else:
                print("Game ended as wall has been hit {} times.".format(ep))
                break

        # sleep(timestep)

if __name__ == "__main__":
    main()