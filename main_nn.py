from src.vehicle import Vehicle
from src.visualiser import Vis
from src.track import TrackHandler
from src.nn import NeuralNetwork
from src.rl_st import DQNAgent
from sandbox.game_pad_inputs import GamePad
from src.environment import FurmulaOne
from src.lidar import Lidar
import numpy as np
from time import sleep
import time

def main():
    task_rate = 0.1
    env = FurmulaOne(track='nardo', is_store=False, task_rate=task_rate)
    agent = DQNAgent(num_inputs=len(env.observation_space.high), num_actions=env.action_space.n, custom_sim_model=True,load_file='st_dqn_-443.00max__-459.50avg_-476.00min__1588408179.model')
    for i in range(1):
        current_state = env.reset()
        done = False
        while not done:
            #t = time.time()
            action = np.argmax(agent.get_qs(current_state))
            current_state, reward, done, _ = env.step(action)
            env.render()
            #sleep(task_rate-(time.time()-t))

if __name__ == "__main__":
    main()
