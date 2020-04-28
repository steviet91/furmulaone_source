import gym
from src.vehicle import Vehicle
from src.lidar import Lidar
from src.track import TrackHandler
from src.visualiser import Vis
import numpy as np


class FurmulaOne(gym.Env):

    _NUM_STATES = Lidar._NRays * 3 + 6
    _NUM_ACTIONS = 15
    _MAX_T_SIM = 10 * 60 # maximum simulation time, prevents slow driving winning

    def __init__(self, track='dodec_track', task_rate=0.01):
        # initialise the objects
        self.task_rate = task_rate
        self.track = TrackHandler(track)
        self.vehicle = Vehicle(1, self.track, 60, 60, 60, task_rate=self.task_rate, auto_reset=False)
        self.vis = Vis(self.track, self.vehicle)

    def reset(self):

        self.rThrottlePedalDemand = 0.0
        self.rBrakePedalDemand = 0.0
        self.aSteeringWheelDemand = 0.0
        self.t_sim = 0.0
        self.t_sim_reward_mem = 0.0
        self.drLapProgress = 0.0
        self.r_progress_mem = 0.0

        self.vehicle.reset_states()
        self.vehicle.reset_vehicle_position()

        self.get_state()

        return self.get_state()

    def render(self):

        self.vis.draw()

    def step(self, action):
        """
            Possible actions are as follows:
                [0]     - Do nothing
                [1]     - Increase steering angle
                [2]     - Reduce steering angle
                [3]     - Increase throttle
                [4]     - Reduce throttle
                [5]     - Increase brake
                [6]     - Reduce brake
                [7]     - Increase throttle - increase steering angle
                [8]     - Increase throttle - reduce steering angle
                [9]     - Reduce throttle - increase steering angle
                [10]    - Reduce throttle - reduce steering angle
                [11]    - Increase brake - increase steering angle
                [12]    - Increase brake - reduce steering angle
                [13]    - Reduce brake - increase steering angle
                [14]    - Reduce brake - reduce steering angle
        """
        # increment simulation time
        self.t_sim += self.task_rate
        """
        if action == 1:
            self.aSteeringWheelDemand = 360.0
            self.rBrakePedalDemand = 0.0
            self.rThrottlePedalDemand = 0.0
        if action == 2:
            self.aSteeringWheelDemand = -360.0
            self.rBrakePedalDemand = 0.0
            self.rThrottlePedalDemand = 0.0
        if action == 3:
            self.aSteeringWheelDemand = 0
            self.rBrakePedalDemand = 1.0
            self.rThrottlePedalDemand = 0.0
        if action == 4:
            self.aSteeringWheelDemand = 0.0
            self.rBrakePedalDemand = 0.0
            self.rThrottlePedalDemand = 1.0
        """
        # sort out the brake
        if any(action == i for i in [5, 11, 12]):
            self.rBrakePedalDemand += 100 * self.task_rate

        elif any(action == i for i in [6, 13, 14]):
            self.rBrakePedalDemand -= 100 * self.task_rate
        self.rBrakePedalDemand = max(0.0, min(1.0, self.rBrakePedalDemand))

        # sort out the throttle
        if self.rBrakePedalDemand > 0:
            self.rThrottlePedalDemand = 0.0
        else:
            if any(action == i for i in [3,7,8]):
                self.rThrottlePedalDemand += 100 * self.task_rate
            elif any(action == i for i in [4,9,10]):
                self.rThrottlePedalDemand -= 100 * self.task_rate
        self.rThrottlePedalDemand = max(0.0, min(1.0, self.rThrottlePedalDemand))

        # sort out the steering
        if any(action == i for i in [1, 7, 9, 11, 13]):
            self.aSteeringWheelDemand += 1000 * self.task_rate
        elif any(action == i for i in [2, 8, 10, 12, 14]):
            self.aSteeringWheelDemand -= 1000 * self.task_rate
        self.aSteeringWheelDemand = max(-360.0, min(360.0, self.aSteeringWheelDemand))

        # update the vehicle
        self.vehicle.update(self.rThrottlePedalDemand, self.rBrakePedalDemand, self.aSteeringWheelDemand)
        self.r_progress = self.vehicle.rLapProgress + 1.0 * self.vehicle.NLapsComplete

        # check if done
        if self.vehicle.bHasCollided or self.vehicle.bMovingBackwards or self.vehicle.NLapsComplete >= 5 or self.t_sim > self._MAX_T_SIM:
            print(self.vehicle.bHasCollided, self.vehicle.bMovingBackwards, self.vehicle.NLapsComplete, self.t_sim)
            done = True
        else:
            done = False

        # calculate the reward
        reward = self.calc_reward(done)

        # set the return values
        return self.get_state(), reward, done, {}

    def calc_reward(self, done):
        """
            Calculate the reward value
        """
        # car has crashed or moving backwards
        if self.vehicle.bHasCollided or self.vehicle.bMovingBackwards:
            return -400

        # reward speed at which the vehicle is navigating the lap
        if self.r_progress != self.r_progress_mem:
            dt_reward = self.t_sim - self.t_sim_reward_mem
            self.drLapProgress = max(0.0, min(1.0, (self.r_progress - self.r_progress_mem) / dt_reward))
            self.t_sim_reward_mem = self.t_sim
            self.r_progress_mem = self.r_progress

        # if vehicle speed is below a thresh then penalise all non +ve accelerations
        if self.vehicle.vVehicle < 10.0:
            if self.vehicle.gxVehicle <= 0:
                return -1
            else:
                return 1
        else:
            return self.drLapProgress

    def get_state(self):
        """
            Return the normalised values
        """

        # lidar data, normalised
        coll_arr = np.hstack((self.vehicle.lidar_front.collision_array,
                            self.vehicle.lidar_left.collision_array,
                            self.vehicle.lidar_right.collision_array))
        coll_arr[np.where(coll_arr < 0)[0]] = Lidar._xLidarRange
        coll_arr = coll_arr / Lidar._xLidarRange

        # vehicle info
        veh_info = np.array([max(0.0, min(1.0, self.vehicle.vVehicle / 100)),
                            max(0.0, min(1.0, (self.vehicle.gxVehicle + 10) / 20)),
                            max(0.0, min(1.0, (self.vehicle.gyVehicle + 10) / 20)),
                            self.rThrottlePedalDemand, self.rBrakePedalDemand,
                            (self.aSteeringWheelDemand + 360) / 720])

        states = np.hstack((coll_arr,veh_info))
        return states

    def close(self):
        pass


if __name__ == "__main__":
    env = FurmulaOne()
    env.reset()
    for i in range(10000):
        new_state, reward, done, _ = env.step(3)
        print(new_state)
        #print(reward)
        if done:
            break
        env.render()
    env.close()
