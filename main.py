from src.vehicle import Vehicle
from src.visualiser import Vis
from src.track import TrackHandler
# import keyboard
from time import sleep
import time
from threading import Thread
from inputs import get_gamepad, get_key, devices


class GamePad(object):

    # The amount by which to vary the throttle/brake/steering demand on each key event (keyboard controls only)
    _throttle_brake_delta = 0.05
    _steering_delta = 45

    def __init__(self):
        # Tells the main thread that the user has asked to quit or reset
        self.quit_requested = False
        self.reset_requested = False
        # Init the demans all to 0
        self.aSteeringWheelDemand = 0.0
        self.rBrakeDemand = 0.0
        self.rThrottleDemand = 0.0
        self.run_thread = True
        # Check if a gamepad is present
        self.use_gamepad = len(devices.gamepads) > 0

    def get_gamepad_inputs(self):
        while self.run_thread:

            # Check if a gamepad is present
            if self.use_gamepad:
                try:
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
                except:
                    self.aSteeringWheelDemand = 0.0
                    self.rBrakeDemand = 0.0
                    self.rThrottleDemand = 0.0
                    # pass # We still need to check keyboard events in case of q/r being pressed
            # Check the keyboard regardless (so we can listen for reset / quit requests)
            events = get_key()
            if events:
                for event in events:
                    if event.code == "KEY_Q":
                        self.quit_requested = True
                    if event.code == "KEY_R":
                        self.reset_requested = True
                    if event.code == "KEY_W":
                        if event.state > 0:
                            # Then the key has been pressed (1 ==> key down) or is being pressed (2 ==> held down)
                            self.rThrottleDemand += self._throttle_brake_delta
                            self.rBrakeDemand = 0
                        else:
                            # Key up
                            self.rThrottleDemand = 0
                    if event.code == "KEY_S":
                        if event.state > 0:
                            self.rThrottleDemand = 0
                            self.rBrakeDemand += self._throttle_brake_delta
                        else:
                            # Key up
                            self.rBrakeDemand = 0
                    if event.code == "KEY_A":
                        if event.state > 0:
                            self.aSteeringWheelDemand -= self._steering_delta
                        else:
                            # Key up
                            self.aSteeringWheelDemand = 0
                    if event.code == "KEY_D":
                        if event.state > 0:
                            self.aSteeringWheelDemand += self._steering_delta
                        else:
                            # Key up
                            self.aSteeringWheelDemand = 0
                    # Make sure the demands are sensible
                    self.rThrottleDemand = max(0, min(1, self.rThrottleDemand))
                    self.rBrakeDemand = max(0, min(1, self.rBrakeDemand))
                    self.aSteeringWheelDemand = max(-360, min(360, self.aSteeringWheelDemand))
                    # print('rThrottleDemand: ', self.rThrottleDemand)
                    # print('rBrakeDemand: ', self.rBrakeDemand)
                    # print('aSteeringWheelDemand: ', self.aSteeringWheelDemand)

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
    run_game = True
    while run_game:
        # Get user inputs
        if gp.quit_requested:
            print("Quitting...")
            break
        if gp.reset_requested:
            veh.reset_states()
            veh.reset_vehicle_position()

        veh.set_driver_inputs(gp.rThrottleDemand, gp.rBrakeDemand, gp.aSteeringWheelDemand)

        # Update the dynamics
        veh.update_long_dynamics()
        veh.update_lat_dynamics()

        # update vehicle position
        veh.update_position()

        # check for collision with track model
        veh.check_for_vehicle_collision()

        # fire the lidars
        veh.update_lidars()

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
