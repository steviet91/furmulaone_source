from src.vehicle import Vehicle
from src.visualiser import Vis
from src.track import TrackHandler
import keyboard
from time import sleep
from threading import Thread
from inputs import get_gamepad


class GamePad(object):
    def __init__(self):
        self.aSteeringWheelDemand = 0.0
        self.rBrakeDemand = 0.0
        self.rThrottleDemand = 0.0
        self.run_thread = True

    def get_gamepad_inputs(self):
        while self.run_thread:
            events = get_gamepad()
            for event in events:
                if event.code == "ABS_X":
                    self.aSteeringWheelDemand = max(-360, min(360, event.state / 12000 * 360))  # assume 12000 is max
                    # print('aSteeringWheelDemand: ', event.state,self.aSteeringWheelDemand)
                if event.code == "ABS_Z":
                    self.rBrakeDemand = min(1, event.state / 255)  # max is 255
                    # print('rBrakeDemand: ', event.state,self.rBrakeDemand)
                if event.code == "ABS_RZ":
                    self.rThrottleDemand = min(1, event.state / 255)  # max is 255
                    # print('rThrottleDemand: ', event.state,self.rThrottleDemand)
            sleep(0.0001)

    def exit_thread(self):
        self.run_thread = False


def main():
    track = TrackHandler('octo_track')
    veh = Vehicle(1, track)
    vis = Vis(track, veh)
    gp = GamePad()
    in_thread = Thread(target=gp.get_gamepad_inputs, daemon=True)
    in_thread.start()
    while True:

        # Get user inputs
        if keyboard.is_pressed('q'):
            break
        if keyboard.is_pressed('r'):
            veh.reset_states()
            veh.reset_vehicle_position()
        """
            if keyboard.is_pressed('w'):
                rThrottleDemand = 1.0
            else:
                rThrottleDemand = 0.0
            if keyboard.is_pressed('s'):
                rBrakeDemand = 1.0
            else:
                rBrakeDemand = 0.0
            if keyboard.is_pressed('a'):
                aSteeringWheelDemand = -360.0
            elif keyboard.is_pressed('d'):
                aSteeringWheelDemand = 360.0
            else:
                aSteeringWheelDemand = 0.0
        """
        # set the driver inputs
        veh.set_driver_inputs(gp.rThrottleDemand, gp.rBrakeDemand, gp.aSteeringWheelDemand)

        # Update the dynamics
        veh.update_long_dynamics()
        veh.update_lat_dynamics()
        veh.update_position()
        veh.check_for_vehicle_collision()
        veh.update_lidars()
        vis.draw_car()
        vis.draw_data()
        vis.draw_track()
        vis.draw_all_lidars()
        vis.render_image()
        vis.update_camera_position()
        sleep(0.05)


if __name__ == "__main__":
    main()
    sleep(5)
