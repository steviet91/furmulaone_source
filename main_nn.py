from src.vehicle import Vehicle
from src.visualiser import Vis
from src.track import TrackHandler
from src.nn import NeuralNetwork
from sandbox.game_pad_inputs import GamePad
from src.lidar import Lidar
import numpy as np
from time import sleep
import time

# ####################
# NN OUTPUT HANDLING #
# ####################
def update_nn_outputs( outputs, nn):
    """
        Assumes outputs are [Thr/Brake, Steering]
    """

    thr_brake = max(0.0, min(1.0, nn.outputs[0])) * 2 - 1
    if thr_brake >= 0:
        outputs[0] = thr_brake
        outputs[1] = 0.0
    else:
        outputs[0] = 0.0
        outputs[1] = thr_brake
    outputs[2] = (max(0.0, min(1.0, nn.outputs[1])) * 2 - 1) * 360.0

# ###################
# NN INPUT HANDLING #
# ###################
def update_nn_inputs(inputs, veh):
    """
        Fixes the order of the inputs
        Lidar is scaled 0-1 - out of range lidar is set to max distance.
    """
    collArr = np.hstack((veh.lidar_front.collision_array, veh.lidar_left.collision_array, veh.lidar_right.collision_array))
    collArr[np.where(collArr < 0)[0]] = Lidar._xLidarRange
    collArr = collArr / Lidar._xLidarRange
    inputs[:] = collArr

def main():
    # instantiate the objects
    task_rate = 0.1
    track = TrackHandler('dodec_track')
    nn = NeuralNetwork.loader('20200425_064103_id_4')
    veh = Vehicle(1, track, 60, 60 ,60, task_rate=task_rate)
    nn_inputs = np.zeros(9)
    nn_outputs = np.zeros(3)
    vis = Vis(track, veh)
    run_game = True



    while run_game:
        t = time.time()
        """
        if gp.quit_requested:
            print("Quitting...")
            gp.exit_thread()
            break
        """
        nn.update_network(nn_inputs)
        update_nn_outputs(nn_outputs, nn)
        # Run a vehicle update
        veh.update(nn_outputs[0], nn_outputs[1], nn_outputs[2])
        update_nn_inputs(nn_inputs, veh)

        # draw the visualisation
        vis.reset_image()
        vis.draw_car()
        vis.draw_track()
        vis.draw_all_lidars()
        vis.draw_demands()
        vis.render_image()
        vis.update_camera_position()
        tSleep = task_rate - (time.time()-t)
        if tSleep < 0:
            tSleep = 0.00001
        sleep(tSleep)



if __name__ == "__main__":
    main()
    sleep(5)
