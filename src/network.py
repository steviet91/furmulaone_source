import socket
from multiprocessing import Process
from multiprocessing import Pipe
from time import sleep
import time
import numpy as np
import struct

class DriverInputsRecv(object):
    """
        Object to read in the driver inputs, the game is the server in this instance
    """

    def __init__(self):
        """
            Initialise the object
        """
        # find the scripts path
        import os
        self.module_path = os.path.dirname(os.path.abspath(__file__))

        # read in the config
        import json
        with open(self.module_path + '/../setup/di_network_config.json','r') as f:
            self.config = json.load(f)

        # initialise the network
        self.initialise_network()

        # initialise some variables
        self.NMessageCounter = 0
        self.tMessageTimeStamp = None
        self.buff_len = self.config['buff_len']

        # run the socket process
        self.set_pipe, self.get_pipe = Pipe()
        self.run_proc = True
        self.sock_proc = Process(target='self.run_recv',
                                    name='DriverInputRecvProc',
                                    args=(self.set_pipe,),
                                    daemon=True)
        self.sock_proc.start()


    def initialise_network(self):
        """
            Initalise the network for the driver inputs to be recieved onto
        """

        self.sock = socket.socket(socket.AF_INET, # Internet
                                    socket.SOCK_DGRAM) # UDP

        self.sock.bind((self.config['ip_addr'], self.config['ip_port']))
        self.sock.settimeout(self.config['sock_timeout'])

    def run_recv(self, set_pipe):
        """
            Run a loop to get the latest information from the driver demands
        """
        while self.run_proc:
            # check the pipe for receive
            if set_pipe.poll():
                # the only information we should get here is a boolean to
                # stop running the process
                if set_pipe.recv(self.buff_len):
                    self.run_proc = False

            # now check the network socket
            try:
                data, addr = sock.recvfrom()
                if len(data) == self.buff_len:
                    self.process_recv_data(data, set_pipe)
            except socket.timeout:
                pass

            sleep(0.001)

    def exit(self):
        """
            Call to close down the object properly
        """
        # terminate the recv process
        self.get_pipe.send(True)

        # close down the socket
        self.sock.close()

    def process_recv_data(self, data, set_pipe):
        """
            Extract the necessary information from the message
        """
        # upack the message - note looking for a static message type
        data = struct.unpack('@I@d@d@d@d@d@d@d@c@c', data)

        # check the message counter
        NMessageCounter = data[0]
        if NMessageCounter < self.NMessageCounter:
            # reach the limit of uint32
            self.NMessageCounter = NMessageCounter
        else:
            if (NMessageCounter - self.NMessageCounter) > 1:
                print('Warning: Driver inputs has droped packets')
            self.NMessageCounter = NMessageCounter

        # check the time stamps
        if self.tMessageTimeStamp is None:
            self.tMessageTimeStamp = data[1]
        else:
            if (data[1] - self.tMessageTimeStamp) > 0.3:
                print('Warning: Delay of {:.3f} s since last driver input message'.format(data[1] - self.tMessageTimeStamp))
            self.tMessageTimeStamp = data[1]

        # update the input dict and send throug the pipe
        data_dict = {'rThrottlePedalDemand': data[2],
                        'rBrakePedalDemand': data[3],
                        'aSteeringWheelDemanded': data[4],
                        'aLidarFront': data[5],
                        'aLidarLeft': data[6],
                        'aLidarRight': data[7],
                        'bResetCar': bool(data[8]),
                        'bQuit': bool(data[9])}
        set_pipe.send(data_dict)

    def check_network_data(self):
        """
            Check the network data and return the latest data set, if no new data then set to None
        """
        data_dict = None
        while self.get_pipe.poll():
            data_dict = self.get_pipe.recv()

        return data_dict


class DriverInputsSend(object):
    """
        Object to store and send the driver inputs to the game
    """
    def __init__(self):
        """
            Initialise the object
        """
        # find the scripts path
        import os
        self.module_path = os.path.dirname(os.path.abspath(__file__))

        # read in the config
        import json
        with open(self.module_path + '/../setup/di_network_config.json','r') as f:
            self.config = json.load(f)

        # set up the variables
        self.data_dict = {'rThrottlePedalDemand': 0.0,
                        'rBrakePedalDemand': 0.0,
                        'aSteeringWheelDemanded': 0.0,
                        'aLidarFront': 0.0,
                        'aLidarLeft': 0.0,
                        'aLidarRight': 0.0,
                        'bResetCar': False,
                        'bQuit': False}
        self.NMessageCounter = np.array([0], dtype=np.uint32) # np will auto rollover when limit of uin32 is reached
        self.tMessageTimeStamp = None

        # initialise the network
        self.initialise_network()

    def initialise_network(self):
        """
            Initalise the network for the driver inputs to be sent
        """
        self.sock = socket.socket(socket.AF_INET, # Internet
                                    socket.SOCK_DGRAM) # UDP

    def set_throttle(self, rThrottleDemanded: float):
        """
            Set the value of the throttle demand
        """
        self.data_dict['rThrottlePedalDemand'] = rThrottleDemanded

    def set_brake(self, rBrakePedalDemanded: float):
        """
            Set the value of the brake demand
        """
        self.data_dict['rBrakePedalDemand'] = rBrakePedalDemanded

    def set_steering(self, aSteeringWheelDemanded: float):
        """
            Set the value of the steering wheel angle
        """
        self.data_dict['aSteeringWheelDemanded'] = aSteeringWheelDemanded

    def set_reset(self, bReset: bool):
        """
            Set the value of the reset
        """
        self.data_dict['bResetCar'] = bReset

    def set_quit(self, bQuit: bool):
        """
            Set the value of the quit
        """
        self.data_dict['bQuit'] = bQuit

    def set_lidar_angle_front(self, aLidar: float):
        """
            Set the nominal angle of the front lidar
        """
        self.data_dict['aLidarFront'] = aLidar

    def set_lidar_angle_left(self, aLidar: float):
        """
            Set the nominal angle of the left lidar
        """
        self.data_dict['aLidarLeft'] = aLidar

    def set_lidar_angle_right(self, aLidar: float):
        """
            Set the nominal angle of the right lidar
        """
        self.data_dict['aLidarRight'] = aLidar

    def send_data(self):
        """
            Send the data over the network
        """
        # pack up the data
        raw_data = [ v for v in  self.data_dict.values() ]
        self.tMessageTimeStamp = time.time()
        data = [self.NMessageCounter, self.tMessageTimeStamp] + raw_data
        data = struct.pack('@I@d@d@d@d@d@d@d@c@c', data)

        # send the data
        self.sock.sendto(data, (self.config['ip_addr'], self.config['ip_port']))

        # increment the send counter
        self.NMessageCounter += 1
