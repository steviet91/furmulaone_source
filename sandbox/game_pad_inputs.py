from time import sleep
from src.network import DriverInputsSend
from inputs import get_gamepad, get_key, devices
from threading import Thread

class GamePad(object):

    # The amount by which to vary the throttle/brake/steering demand on each key event (keyboard controls only)
    _throttle_brake_delta = 0.05
    _steering_delta = 45
    _lidar_rot_delta = 5

    def __init__(self):
        # Tells the main thread that the user has asked to quit or reset
        self.quit_requested = False
        self.reset_requested = False
        # Init the demans all to 0
        self.aSteeringWheelDemand = 0.0
        self.rBrakeDemand = 0.0
        self.rThrottleDemand = 0.0
        self.lidar_rot = 0.0
        self.run_thread = True
        # Check if a gamepad is present
        self.use_gamepad = len(devices.gamepads) > 0

        self.input_thread = Thread(target=self.get_gamepad_inputs, daemon=True)
        self.input_thread.start()

    def get_gamepad_inputs(self):
        while self.run_thread:
            # Check if a gamepad is present
            if self.use_gamepad:
                try:
                    events = get_gamepad()
                    for event in events:
                        if event.code == "BTN_NORTH":
                            self.quit_requested = True
                        if event.code == "BTN_EAST":
                            self.reset_requested = True
                        if event.code == "ABS_X":
                            self.aSteeringWheelDemand = max(-360, min(360, event.state / 12000 * 360))  # assume 12000 is max
                            # print('aSteeringWheelDemand: ', event.state,self.aSteeringWheelDemand)
                        if event.code == "ABS_Z":
                            self.rBrakeDemand = min(1, event.state / 255)  # max is 255
                            # print('rBrakeDemand: ', event.state,self.rBrakeDemand)
                        if event.code == "ABS_RZ":
                            self.rThrottleDemand = min(1, event.state / 255)  # max is 255
                        if event.code == 'ABS_HAT0X':
                            if event.state < 0:
                                self.lidar_rot -= self._lidar_rot_delta
                            elif event.state > 0:
                                self.lidar_rot += self._lidar_rot_delta
                            # print('rThrottleDemand: ', event.state,self.rThrottleDemand)
                except:
                    self.aSteeringWheelDemand = 0.0
                    self.rBrakeDemand = 0.0
                    self.rThrottleDemand = 0.0
                    pass # We still need to check keyboard events in case of q/r being pressed
            else:
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
    gp = GamePad()
    s = DriverInputsSend()

    while True:
        # check quit
        if gp.quit_requested:
            s.set_quit(True)
            s.send_data()
            gp.exit_thread()
            break

        # check reset
        if gp.reset_requested:
            s.set_reset(True)
            # reset the request
            gp.reset_requested = False
        else:
            s.set_reset(False)

        # update driver inputs
        s.set_throttle(gp.rThrottleDemand)
        s.set_brake(gp.rBrakeDemand)
        s.set_steering(gp.aSteeringWheelDemand)
        s.set_lidar_angle_front(gp.lidar_rot)
        s.set_lidar_angle_left(gp.lidar_rot)
        s.set_lidar_angle_right(gp.lidar_rot)

        s.send_data()

        sleep(0.01)

if __name__ == "__main__":
    main()
