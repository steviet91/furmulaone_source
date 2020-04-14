from src.vehicle import Vehicle
from src.visualiser import Vis
from src.track import TrackHandler
from src.network import DriverInputsRecv
from time import sleep
import time


def main():
    # instantiate the objects
    drv_in = DriverInputsRecv()
    track = TrackHandler('octo_track')
    veh = Vehicle(1, track, 60, 60 ,60)
    vis = Vis(track, veh)
    run_game = True

    aLidarFront = 0.0 # angle from their nominal
    aLidarLeft = 0.0 # angle from their nominal
    aLidarRight = 0.0 # angle from their nominal

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
            veh.set_driver_inputs(input_data['rThrottlePedalDemanded'],
                                    input_data['rBrakePedalDemanded'],
                                    input_data['aSteeringWheelDemanded'])

            # lidar angle
            aLidarFront = input_data['aLidarFront']
            aLidarLeft = input_data['aLidarLeft']
            aLidarRight = input_data['aLidarRight']

        # Update the dynamics
        veh.update_long_dynamics()
        veh.update_lat_dynamics()

        # update vehicle position
        veh.update_position()

        # check for collision with track model
        veh.check_for_vehicle_collision()

        # fire the lidars
        veh.update_lidars(aRotFront=aLidarFront, aRotL=aLidarLeft, aRotR=aLidarRight)

        # draw the visualisation
        vis.draw_car()
        vis.draw_track()
        vis.draw_all_lidars()
        vis.render_image()
        vis.update_camera_position()

        # check the lap data
        track.check_new_lap(veh.posVehicle[0], veh.posVehicle[1])


if __name__ == "__main__":
    main()
    sleep(5)
