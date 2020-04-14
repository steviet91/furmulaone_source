from src.vehicle import Vehicle
from src.visualiser import Vis
from src.track import TrackHandler
# import keyboard
from time import sleep
import time
from threading import Thread
from inputs import get_gamepad, get_key, devices


class GamePad(object):
    def __init__(self):
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
                    pass
            else:
                # Check the keyboard - TODO: ramp demands with time
                events = get_key()
                if events:
                    for event in events:
                        print("event: " + event.code)
                        if event.state > 0:
                            # Then the key has been pressed (1 ==> key down) or is being pressed (2 ==> held down)
                            # if keyboard.is_pressed('w'):
                            if event.code == "KEY_W":
                                self.rThrottleDemand = 1.0
                                self.rBrakeDemand = 0.0
                                print('rThrottleDemand: ', event.state,self.rThrottleDemand)
                            # if keyboard.is_pressed('s'):
                            if event.code == "KEY_S":
                                self.rThrottleDemand = 0.0
                                self.rBrakeDemand = 1.0
                                print('rBrakeDemand: ', event.state,self.rThrottleDemand)
                            # if keyboard.is_pressed('a'):
                            if event.code == "KEY_A":
                                self.aSteeringWheelDemand = -360.0
                            # elif keyboard.is_pressed('d'):
                            if event.code == "KEY_D":
                                self.aSteeringWheelDemand = 360.0
            sleep(0.0001)

    def exit_thread(self):
        self.run_thread = False


def main():
    print("Initialising game...")
    track = TrackHandler('octo_track')
    veh = Vehicle(1, track)
    vis = Vis(track, veh)
    gp = GamePad()
    in_thread = Thread(target=gp.get_gamepad_inputs, daemon=True)
    in_thread.start()
    run_game = True
    print("Starting game...")
    while run_game:
        # print("Press enter to begin ...")
        # # Get user inputs
        # events = get_key()
        # if events:
        #     for event in events:
        #         if event.state == 1:
        #             # Key down
        #             if event.code == 'KEY_ESC':
        #                 print("Esc pressed")
        #             elif event.code == 'KEY_Q':
        #                 run_game = False
        #             elif event.code == 'KEY_R':
        #                 veh.reset_states()
        #                 veh.reset_vehicle_position()
        # # Get user inputs
        # if keyboard.is_pressed('q'):
        #     break
        # if keyboard.is_pressed('r'):
        #     veh.reset_states()
        #     veh.reset_vehicle_position()

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
