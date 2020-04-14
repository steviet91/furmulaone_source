import socket
from multiprocessing import Process
from multiprocessing import Pipe
from time import sleep


class DriverInputs(object):
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
                                    socket.SOCK_DGRAM # UDP)

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
        data = unpack('Idddddddcc', data)

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
