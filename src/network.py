import socket
from multiprocessing import Process
from multiprocessing import Pipe
from time import sleep
import time
import numpy as np
import struct




# ########################
# #### DRIVER INPUTS #####
# ########################
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
        self.msg_len = self.config['msg_len']

        # run the socket process
        self.set_pipe, self.get_pipe = Pipe()
        self.run_proc = True
        self.sock_proc = Process(target=self.run_recv,
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
        self.sock.setblocking(1)

    def run_recv(self, set_pipe):
        """
            Run a loop to get the latest information from the driver demands
        """
        while self.run_proc:
            # check the pipe for receive
            if set_pipe.poll():
                # the only information we should get here is a boolean to
                # stop running the process
                if set_pipe.recv():
                    self.run_proc = False

            # now check the network socket
            try:
                data = self.sock.recv(1024)
                if len(data) == self.msg_len:
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
        data = struct.unpack('@Qddddddd??', data)

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
        data_dict = {'rThrottlePedalDemanded': data[2],
                        'rBrakePedalDemanded': data[3],
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
        self.data_dict = {'rThrottlePedalDemanded': 0.0,
                        'rBrakePedalDemanded': 0.0,
                        'aSteeringWheelDemanded': 0.0,
                        'aLidarFront': 0.0,
                        'aLidarLeft': 0.0,
                        'aLidarRight': 0.0,
                        'bResetCar': False,
                        'bQuit': False}
        self.NMessageCounter = np.array([0], dtype=np.uint64) # np will auto rollover when limit of uin32 is reached
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
        self.data_dict['rThrottlePedalDemanded'] = rThrottleDemanded

    def set_brake(self, rBrakePedalDemandeded: float):
        """
            Set the value of the brake demand
        """
        self.data_dict['rBrakePedalDemanded'] = rBrakePedalDemandeded

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
        data = [self.NMessageCounter[0], self.tMessageTimeStamp] + raw_data
        data = struct.pack('@Qddddddd??', *data)

        # send the data
        self.sock.sendto(data, (self.config['ip_addr'], self.config['ip_port']))

        # increment the send counter
        self.NMessageCounter += 1


# ##########################
# #### VEHICLE OUTPUTS #####
# ##########################
class VehicleOutputsRecv(object):
    """
        Recieves all the vehicle data from the udp and makes it available to the
        user as a data dict or through get functions
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
        with open(self.module_path + '/../setup/vo_network_config.json','r') as f:
            self.config = json.load(f)

        # initialise the network
        self.initialise_network()

        # initialise some variables
        self.NMessageCounter = 0
        self.tMessageTimeStamp = None
        self.NMessageHeader = self.config['NMessageHeader']

        # initialise the data dict
        self.initialise_data_dict()

        # run the socket process
        self.set_pipe, self.get_pipe = Pipe()
        self.run_proc = True
        self.sock_proc = Process(target=self.run_recv,
                                    name='VehicleOuputsRecvProc',
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
        self.sock.setblocking(1)

    def initialise_data_dict(self):
        """
            Set up the data dictionary
        """
        self.data_dict = {'tMessageTimeStamp': None, 'xLidarCollisionFront': [],
                            'xLidarCollisionL': [], 'xLidarCollisionR': [],
                            'gxVehicle': 0.0, 'gyVehicle': 0.0,
                            'vxVehicle': 0.0, 'vyVehicle': 0.0, 'aSlipF': 0.0,
                            'aSlipR': 0.0, 'aBodySlip': 0.0, 'aYaw': 0.0,
                            'rThrottlePedal': 0.0, 'rBrakePedal': 0.0, 'aSteeringWheel': 0.0,
                            'nYaw': 0.0, 'xVehicle': 0.0, 'yVehicle': 0.0,
                            'aLidarRotL': 0.0, 'aLidarRotR': 0.0, 'aLidarRotFront': 0.0}

    def run_recv(self, set_pipe):
        """
            Method to run the recv command from the socket and send the data through
            a pipe to the main process for consumption
        """
        while self.run_proc:
            # check the pipe for receive
            if set_pipe.poll():
                # the only information we should get here is a boolean to
                # stop running the process
                if set_pipe.recv():
                    self.run_proc = False

            # now check the network socket
            try:
                data = self.sock.recv(1024)
                NMessageHeader =struct.unpack('@Q',data[:8])[0]
                if NMessageHeader == self.NMessageHeader:
                    self.process_recv_data(data[8:], set_pipe)
            except socket.timeout:
                pass

            sleep(0.001)

    def process_recv_data(self, data, set_pipe):
        """
            Process the data and place into correct data structures
        """
        # general info
        NMessageCounter = struct.unpack('@Q',data[:8])[0]
        if NMessageCounter < self.NMessageCounter:
            # message counter has hit the data type limit and reset to 0
            self.NMessageCounter = NMessageCounter
        else:
            if (NMessageCounter -  self.NMessageCounter) > 1:
                # can probably do something more useful than this
                print('! WARNING ! Packets have been lost')
            self.NMessageCounter = NMessageCounter
        data = data[8:]
        tMessageTimeStamp = struct.unpack('@d',data[:8])[0]
        if self.tMessageTimeStamp is None:
            self.tMessageTimeStamp = tMessageTimeStamp
        else:
            # check the time since the last message
            if (tMessageTimeStamp - self.tMessageTimeStamp) > 0.1:
                print('! WARNING ! Elapsed time between the last two recv messages was {:.3f} s'.format(tMessageTimeStamp-self.tMessageTimeStamp))
            self.tMessageTimeStamp = tMessageTimeStamp
        data = data[8:]

        # lidar info
        NLidarRays = struct.unpack('@Q',data[:8])[0]
        lidar_fmt = '@' + 'd' * NLidarRays
        data = data[8:]
        # front
        lidarfront_data = list(struct.unpack(lidar_fmt, data[:8*NLidarRays]))
        data = data[8*NLidarRays:]
        # left
        lidarleft_data = list(struct.unpack(lidar_fmt, data[:8*NLidarRays]))
        data = data[8*NLidarRays:]
        # right
        lidarright_data = list(struct.unpack(lidar_fmt, data[:8*NLidarRays]))
        data = data[8*NLidarRays:]

        # vehicle sensors - assumed all to be doubles
        sens_fmt = '@' + 'd' * (len(data) // 8)
        data = struct.unpack(sens_fmt, data)

        # package up the information into a dictionary (note: this is need maintaining
        # if more information is added to the socket)
        self.data_dict['tMessageTimeStamp'] = self.tMessageTimeStamp
        self.data_dict['xLidarCollisionFront'] = lidarfront_data
        self.data_dict['xLidarCollisionL'] = lidarleft_data
        self.data_dict['xLidarCollisionR'] = lidarright_data

        self.data_dict['gxVehicle'] = data[0]
        self.data_dict['gyVehicle'] = data[1]
        self.data_dict['vxVehicle'] = data[2]
        self.data_dict['vyVehicle'] = data[3]
        self.data_dict['aSlipF'] = data[4]
        self.data_dict['aSlipR'] = data[5]
        self.data_dict['aBodySlip'] = data[6]
        self.data_dict['aYaw'] = data[7]
        self.data_dict['rThrottlePedal'] = data[8]
        self.data_dict['rBrakePedal'] = data[9]
        self.data_dict['aSteeringWheel'] = data[10]
        self.data_dict['nYaw'] = data[11]
        self.data_dict['xVehicle'] = data[12]
        self.data_dict['yVehicle'] = data[13]
        self.data_dict['aLidarRotL'] = data[14]
        self.data_dict['aLidarRotR'] = data[15]
        self.data_dict['aLidarRotFront'] = data[16]

        # send the data down the pipe
        set_pipe.send(self.data_dict)

    def check_network_data(self):
        """
            Check the network for new data, will poll until the buffer is empty
        """
        while self.get_pipe.poll():
            self.data_dict = self.get_pipe.recv()

    def exit(self):
        """
            Call to close down the object properly
        """
        # terminate the recv process
        self.get_pipe.send(True)

        # close down the socket
        self.sock.close()

    def get_all_data(self):
        """
            Return all data as a dictionary
        """
        return self.data_dict

class VehicleOutputsSend(object):
    """
        Sends all vehicle data onto udp for the reciever (the driver) to pick up
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
        with open(self.module_path + '/../setup/vo_network_config.json','r') as f:
            self.config = json.load(f)

        self.NMessageCounter = np.array([0], dtype=np.uint64) # np will auto rollover when limit of uin32 is reached
        self.tMessageTimeStamp = None
        self.NMessageHeader = np.array([self.config['NMessageHeader']], dtype=np.uint64)

        # initialise the network
        self.initialise_network()

    def initialise_network(self):
        """
            Initialise the network socket
        """
        self.sock = socket.socket(socket.AF_INET, # Internet
                                    socket.SOCK_DGRAM) # UDP

    def set_vehicle_sensors(self, veh_sensors: dict):
        """
            Set the data for the vehicle sensors from the provided dict (this won't
            include the lidar)
        """
        self.sensors = [ v for v in veh_sensors.values() ]
        self.sensors_dtype = 'd' * len(self.sensors)

    def set_lidar_data(self, collFront: np.ndarray, collL: np.ndarray, collR: np.ndarray):
        """
            Set the lidar collision information
        """
        self.lidar_colls = list(collFront) + list(collL) + list(collR)
        self.NLidarRays = int(len(collFront))
        self.lidar_dtype = 'd'* (self.NLidarRays * 3)

    def send_data(self):
        """
            Send the data out through the socket
        """
        # get the time stamp
        self.tMessageTimeStamp = time.time()

        # determine the message pack format
        fmt = '@QQdQ' + self.lidar_dtype + self.sensors_dtype

        # pack up the data into a bytes like object
        data = [self.NMessageHeader[0], self.NMessageCounter[0], self.tMessageTimeStamp,
                self.NLidarRays] + self.lidar_colls + self.sensors

        data = struct.pack(fmt, *data)

        # send the data
        self.sock.sendto(data, (self.config['ip_addr'], self.config['ip_port']))

        # increament the counter
        self.NMessageCounter += 1
