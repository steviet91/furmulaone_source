from src.vehicle import Vehicle
from src.visualiser import Vis
from src.track import TrackHandler
from src.network import DriverInputsRecv
from src.network import VehicleOutputsSend
from time import sleep
import time


def main():
    # instantiate the objects
    drv_in = DriverInputsRecv()
    veh_out = VehicleOutputsSend()
    track = TrackHandler('dodec_track')
    veh = Vehicle(1, track, 60, 60 ,60)
    vis = Vis(track, veh)
    run_game = True

    aLidarFront = 0.0 # angle from their nominal
    aLidarLeft = 0.0 # angle from their nominal
    aLidarRight = 0.0 # angle from their nominal
    rThrottlePedalDemand = 0.0
    rBrakePedalDemand = 0.0
    aSteeringWheelDemand = 0.0
    t = time.time()

    while run_game:
        # Check user inputs
        input_data = drv_in.check_network_data()
        if input_data is not None:
            if input_data['bQuit']:
                print("Quitting...")
                break
            if input_data['bResetCar']:
                veh.reset_states()
                veh.reset_vehicle_position()
                vis.reset_camera()

            # set the car controls
            rThrottlePedalDemand = input_data['rThrottlePedalDemanded']
            rBrakePedalDemand = input_data['rBrakePedalDemanded']
            aSteeringWheelDemand = input_data['aSteeringWheelDemanded']

            # lidar angle
            aLidarFront = input_data['aLidarFront']
            aLidarLeft = input_data['aLidarLeft']
            aLidarRight = input_data['aLidarRight']

        # Run a vehicle update
        veh.update(rThrottlePedalDemand, rBrakePedalDemand, aSteeringWheelDemand,
                    aRotFront=aLidarFront, aRotL=aLidarLeft, aRotR=aLidarRight, task_rate=time.time()-t)
        t = time.time()

        # draw the visualisation
        vis.draw_car()
        vis.draw_track()
        vis.draw_all_lidars()
        vis.draw_demands()
        vis.render_image()
        vis.update_camera_position()

        # send the vehicle output dat aacross the network
        veh_out.set_vehicle_sensors(veh.get_vehicle_sensors())
        veh_out.set_lidar_data(veh.lidar_front.collision_array,
                                veh.lidar_left.collision_array,
                                veh.lidar_right.collision_array)
        veh_out.send_data()
        sleep(0.001)



if __name__ == "__main__":
    main()
    sleep(5)
