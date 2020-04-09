import json
import os
import numpy as np
import time
from .track import Track

class Vehicle(object):
    def __init__(self,id: int):
        self.id = 1;

        # Get the module path
        self.module_path = os.path.dirname(os.path.abspath(__file__))

        # load the config file
        with open(self.module_path+'/../setup/vehicle_config.json','r') as f:
            self.config = json.load(f)

        # create addition vehicle properties
        self.initialise_vehicle_properties()

        # initialise the vehicle states
        self.reset_states()

        # initialise the vehicle position
        self.reset_vehicle_position()

        # create timers
        self.tLastInputUpdate = None
        self.tLastLongUpdate = None
        self.tLastLatUpdate = None
        self.tLastPosUpdate = None

        # create lidar object
        self.lidars = None


    def initialise_lidars(self, track: Track, aFOVL: float, aFOVR: float, aFOVFront: float, aRotL: float=None, aRotR: float=None, aRotFront: float=None):
        """
            Set up the LIDAR objects
        """


    def reset_states(self):
        """
            Reset the vehicle states
        """
        self.rThrottlePedal = 0.0
        self.rBrakePedal = 0.0
        self.aSteeringWheel = 0.0
        self.rSlipF = 0.0
        self.rSlipR = 0.0
        self.aSlipF = 0.0
        self.aSlipR = 0.0
        self.aSteer = 0.0
        self.vVehicle = self.config['vVehicleStationary']
        self.vxVehicle = self.config['vVehicleStationary']
        self.vyVehicle = self.config['vVehicleStationary']
        self.gxVehicle = 0.0
        self.gyVehicle = 0.0
        self.nWheelR = 0.0
        self.nWheelF = 0.0
        self.ndotWheelR = 0.0
        self.ndotWheelF = 0.0
        self.FTyreXTotalF = 0.0
        self.FTyreYTotalF = 0.0
        self.FTyreXTotalR = 0.0
        self.FTyreYTotalR = 0.0
        self.aBodySlip = 0.0
        self.aYaw = 3.14/2
        self.nYaw = 0.0
        self.ndotYaw = 0.0


    def reset_vehicle_position(self):
        """
            Reset the vehicle position to (0,0)
        """
        self.posVehicle = np.array([0.0,0.0],dtype=np.float64)

    def initialise_vehicle_properties(self):
        """
            Initialise some more properties using the config
        """
        self.xCOGF = self.config['xWheelBase'] * self.config['rCOGLongR']
        self.xCOGR = self.config['xWheelBase'] * (1 - self.config['rCOGLongR'])
        self.FBrakingMax = self.config['mVehicle'] * self.config['gGravity'] * 0.8 # leave some capacity for lat

    def set_driver_inputs(self,rThrottlePedalDemand: float,rBrakePedalDemand: float,aSteeringWheelDemand: float):
        """
            Update the steering inputs
        """
        #
        # Check the limits of inputs
        #
        rThrottlePedalDemand = max(0.0,min(1.0,rThrottlePedalDemand))
        rBrakePedalDemand = max(0.0,min(1.0,rBrakePedalDemand))
        aSteeringWheelDemand = max(self.config['aSteeringWheelMin'],min(self.config['aSteeringWheelMax'],aSteeringWheelDemand))
        #
        # Prevent combined throttle/braking
        #
        if rBrakePedalDemand > 0.0:
            rThrottlePedalDemand = 0.0
        #
        # Check the input rates
        #
        if self.tLastInputUpdate is None:
            self.tLastInputUpdate = time.time()
            tElapsed = 0.0
            drThrottlePedal = 0.0
            drBrakePedal = 0.0
            nSteeringWheel = 0.0
        else:
            tNow = time.time()
            tElapsed = tNow - self.tLastInputUpdate
            self.tLastInputUpdate = tNow
            drThrottlePedal = (rThrottlePedalDemand - self.rThrottlePedal) / tElapsed
            drBrakePedal = (rBrakePedalDemand - self.rBrakePedal) / tElapsed
            nSteeringWheel = (aSteeringWheelDemand - self.aSteeringWheel) / tElapsed
        # update throttle
        if drThrottlePedal < 0:
            self.rThrottlePedal -= min(abs(drThrottlePedal), self.config['drThrottlePedalMax']) * tElapsed
        else:
            self.rThrottlePedal += min(drThrottlePedal, self.config['drThrottlePedalMax']) * tElapsed
        # update brake
        if drBrakePedal < 0:
            self.rBrakePedal -= min(abs(drBrakePedal), self.config['drBrakePedalMax']) * tElapsed
        else:
            self.rBrakePedal += min(drBrakePedal, self.config['drBrakePedalMax']) * tElapsed
        # update steering
        if nSteeringWheel < 0:
            self.aSteeringWheel -= min(abs(nSteeringWheel), self.config['nSteeringWheelMax']) * tElapsed
        else:
            self.aSteeringWheel += min(nSteeringWheel, self.config['nSteeringWheelMax']) * tElapsed

    def update_long_dynamics(self):
        """
            Update the vehicle longitudinal dynamics
        """
        #
        # calculate the torque demand
        #
        MMax = np.interp(self.nWheelR * 30 / np.pi,self.config['nWheel_BRP'],self.config['MPowertrainMax_LU'])
        MMin = np.interp(self.nWheelR * 30 / np.pi,self.config['nWheel_BRP'],self.config['MPowertrainMin_LU'])
        MPowertrain = (MMax - MMin) * self.rThrottlePedal + MMin
        #
        # calculate the braking demand
        #
        FBrakeTotal = self.FBrakingMax * self.rBrakePedal * np.sign(self.vxVehicle)
        FBrakeF = FBrakeTotal * (1 - self.config['rCOGLongR']) # account for COG pos, weight transfer not modelled
        FBrakeR = FBrakeTotal - FBrakeF
        MBrakeF = FBrakeF * self.config['rBrakeDisc']
        MBrakeR = FBrakeR * self.config['rBrakeDisc']
        #
        # calculate the resistive forces
        #
        FResistive = self.vxVehicle * self.config['Cb'] + 0.5 * (self.vxVehicle**2 * self.config['rhoAir'] + self.config['CdA'])
        #
        # calculate the tyre forces
        #
        # front
        MTotalF = -1 * MBrakeF
        if MTotalF < 0:
            self.FTyreXTotalF = max(-1 * self.config['mVehicle'] * self.config['gGravity'] * (1 - self.config['rCOGLongR']) * self.config['mu'] * 0.9, MTotalF / self.config['rRollRadF'])
        else:
            self.FTyreXTotalF = min(self.config['mVehicle'] * self.config['gGravity'] * (1 - self.config['rCOGLongR']) * self.config['mu'] * 0.9, MTotalF / self.config['rRollRadF'])
        # rear
        MTotalR = MPowertrain - MBrakeR
        if MTotalR < 0:
            self.FTyreXTotalR = max(-1 * self.config['mVehicle'] * self.config['gGravity'] * self.config['rCOGLongR'] * self.config['mu'] * 0.9, MTotalR / self.config['rRollRadR'])
        else:
            self.FTyreXTotalR = min(self.config['mVehicle'] * self.config['gGravity'] * self.config['rCOGLongR'] * self.config['mu'] * 0.9, MTotalR / self.config['rRollRadR'])
        #
        # calculate the accelation
        #
        self.gxVehicle = (self.FTyreXTotalF + self.FTyreXTotalR - FResistive) / self.config['mVehicle']
        #
        # calculate the velocity
        #
        if self.tLastLongUpdate is None:
            self.tLastLongUpdate = time.time()
        else:
            tNow = time.time()
            tElapsed = tNow - self.tLastLongUpdate
            self.tLastLongUpdate = tNow
            self.vxVehicle += self.gxVehicle * tElapsed
            # check for underflow
            if abs(self.vxVehicle) < self.config['vVehicleStationary']:
                self.vxVehicle = self.config['vVehicleStationary'] * np.sign(self.vxVehicle)
        #
        # Update the wheel speeds
        #
        self.nWheelF = self.vxVehicle / self.config['rRollRadF']
        self.nWheelR = self.vxVehicle / self.config['rRollRadR']
        #print(MPowertrain, self.gxVehicle, self.FTyreXTotalF, self.FTyreXTotalR,self.rThrottlePedal,self.vxVehicle)

    def update_lat_dynamics(self):
        """
            Update the lateral dynamics of the vehicle
        """
        # determine the driver demands
        self.aSteer = self.aSteeringWheel / self.config['rSteeringRatio'] * np.pi / 180

        # calculate the slip ratios
        #print('aSteer:',self.aSteer,'vy:',self.vyVehicle,'x',self.xCOGF,'nYaw:',self.nYaw,'vx:',self.vxVehicle)
        if abs(self.vxVehicle) <= self.config['vVehicleStationary']:
            self.aSlipF = 0.0
            self.aSlipR = 0.0
        else:
            self.aSlipF = self.aSteer - np.arctan((self.vyVehicle + self.xCOGF * self.nYaw) / abs(self.vxVehicle))
            self.aSlipR = -1 * np.arctan((self.vyVehicle - self.xCOGR * self.nYaw) / abs(self.vxVehicle))
           
        # calculate the maximum tyre forces
        FTyreYMaxF = np.sqrt((self.config['mVehicle'] * self.config['gGravity'] * (1 - self.config['rCOGLongR']) * self.config['mu'])**2 - (self.FTyreXTotalF)**2)
        FTyreYMaxR = np.sqrt((self.config['mVehicle'] * self.config['gGravity'] * self.config['rCOGLongR'] * self.config['mu'])**2 - (self.FTyreXTotalR)**2)

        # calculate the lateral tyre forces
        if self.aSlipF < 0:
            self.FTyreYTotalF = max(-1 * FTyreYMaxF, 2 * self.config['CTyreY'] * self.aSlipF)
        else:
            self.FTyreYTotalF = min(FTyreYMaxF, 2 * self.config['CTyreY'] * self.aSlipF)
        if self.aSlipR < 0:
            self.FTyreYTotalR = max(-1 * FTyreYMaxR, 2 * self.config['CTyreY'] * self.aSlipR)
        else:
            self.FTyreYTotalR = min(FTyreYMaxR, 2 * self.config['CTyreY'] * self.aSlipR)

        # calculate accelations
        self.gyVehicle = (self.FTyreYTotalF + self.FTyreYTotalR) / self.config['mVehicle'] - self.vxVehicle * self.nYaw
        self.ndotYaw = (self.xCOGF * self.FTyreYTotalF - self.xCOGF * self.FTyreYTotalR) / self.config['jVehicleZZ']

        # calculate the velocities and angles
        if self.tLastLatUpdate is None:
            self.tLastLatUpdate = time.time()
        else:
            tNow = time.time()
            tElapsed = tNow - self.tLastLatUpdate
            self.tLastLatUpdate = tNow
            self.vyVehicle += self.gyVehicle * tElapsed
           
            #if abs(self.vyVehicle) < self.config['vVehicleStationary']:
            #    self.vyVehicle = self.config['vVehicleStationary']
            if abs(self.vxVehicle) <= self.config['vVehicleStationary']:
                self.vyVehicle = 0.0
            self.nYaw += self.ndotYaw * tElapsed
            self.aYaw += self.nYaw * tElapsed

        # calculate velocity and body slip
        self.aBodySlip = np.arctan(self.vyVehicle / abs(self.vxVehicle))
        self.vVehicle = np.sqrt(self.vxVehicle**2 + self.vyVehicle**2)
        #print('vy:',self.vyVehicle,'nYaw:', self.nYaw, 'aSlipF:', self.aSlipF, 'aSlipR:', self.aSlipR)


    def get_vehicle_sensors(self):
        """
            Returns the vehicle sensors
        """
        return {'vVehicle': self.vVehicle, 'vxVehicle': self.vxVehicle, 'vyVehicle': self.vyVehicle, 'aSlipF': self.aSlipF, 'aSlipR': self.aSlipR,
               'rThrottlePedal': self.rThrottlePedal, 'rBrakePedal': self.rBrakePedal, 'aSteeringWheel': self.aSteeringWheel, 'nYaw': self.nYaw}

    def update_position(self):
        """
            Update the vehicle position in the world
        """ 
        if self.tLastPosUpdate is None:
            self.tLastPosUpdate = time.time()
        else:
            tNow = time.time()
            tElapsed = tNow - self.tLastPosUpdate
            self.tLastPosUpdate = tNow
            sVehicleMoved = self.vVehicle * tElapsed
            self.posVehicle[0] += sVehicleMoved * np.cos(self.aYaw+self.aBodySlip) # xVehicle
            self.posVehicle[1] += sVehicleMoved * np.sin(self.aYaw+self.aBodySlip) # yVehicle
