import json
import os
import numpy as np
import time
from .track import Track
from .geom import Circle
from .geom import Line
from .geom import check_for_intersection_lineseg_lineseg
from .geom import calc_angle_between_unit_vectors
from .lidar import Lidar

class Vehicle(object):
    def __init__(self, id: int, track: Track, aLidarFOVFront: float, aLidarFOVL: float, aLidarFOVR: float):
        """
            Initialise the vehicle object
        """
        # save the arguments
        self.id = id
        self.track = track
        self.aLidarFOVFront = aLidarFOVFront
        self.aLidarFOVL = aLidarFOVL
        self.aLidarFOVR = aLidarFOVR

        # Get the module path
        self.module_path = os.path.dirname(os.path.abspath(__file__))

        # load the config file
        with open(self.module_path + '/../setup/vehicle_config.json', 'r') as f:
            self.config = json.load(f)

        # create timers
        self.tLastInputUpdate = None
        self.tLastLongUpdate = None
        self.tLastLatUpdate = None
        self.tLastPosUpdate = None
        self.tLastLidarUpdate = None

        # create the car corner point offsets
        self.carFLOffset = np.array([self.config['xVehicleLength'] * self.config['rCOGLongR'], -0.5 * self.config['xVehicleWidth']])
        self.carFROffset = np.array([self.config['xVehicleLength'] * self.config['rCOGLongR'], 0.5 * self.config['xVehicleWidth']])
        self.carRLOffset = np.array([-1 * self.config['xVehicleLength'] * (1 - self.config['rCOGLongR']), -0.5 * self.config['xVehicleWidth']])
        self.carRROffset = np.array([-1 * self.config['xVehicleLength'] * (1 - self.config['rCOGLongR']), 0.5 * self.config['xVehicleWidth']])

        # create addition vehicle properties
        self.initialise_vehicle_properties()

        # create the lidars
        self.initialise_lidars()

        # initialise the vehicle states
        self.reset_states()

        # initialise the vehicle position
        self.reset_vehicle_position()

        # create lidar object
        self.lidars = None

    def initialise_lidars(self, aRotL: float = 0.0, aRotR: float = 0.0, aRotFront: float = 0.0):
        """
            Set up the LIDAR objects
        """

        # convert angles to rad
        aFOVFront = self.aLidarFOVFront * np.pi / 180
        aFOVL = self.aLidarFOVL * np.pi / 180
        aFOVR = self.aLidarFOVR * np.pi / 180

        # calculate the maximum rotation angle based on FOV
        self.aRotLimFront = self.config['aLidarRotMax'] * np.pi / 180 - aFOVFront / 2
        self.aRotLimL = self.config['aLidarRotMax'] * np.pi / 180 - aFOVL / 2
        self.aRotLimR = self.config['aLidarRotMax'] * np.pi / 180 - aFOVR / 2

        # determine the initial rotation angles
        self.aLidarRotFront = max(-1 * self.aRotLimFront, min(self.aRotLimFront, aRotFront * np.pi / 180))
        self.aLidarRotL = max(-1 * self.aRotLimL, min(self.aRotLimL, aRotL * np.pi / 180))
        self.aLidarRotR = max(-1 * self.aRotLimR, min(self.aRotLimR, aRotR * np.pi / 180))

        # set up the front lidar
        self.aLidarF = 0.0
        self.lidar_front = Lidar(self.track, self.aLidarF, self.carFLOffset[0], 0, aFOVFront)

        # set up the left lidar
        self.aLidarL = -1.0 * np.pi / 2.0
        x0 = self.carFLOffset[0] - self.config['xVehicleLength'] / 2
        y0 = self.carFLOffset[1]
        self.lidar_left = Lidar(self.track, self.aLidarL, x0, y0, aFOVL)

        # set up the right lidar
        self.aLidarR = np.pi / 2.0
        x0 = self.carFROffset[0] - self.config['xVehicleLength'] / 2
        y0 = self.carFROffset[1]
        self.lidar_right = Lidar(self.track, self.aLidarR, x0, y0, aFOVR)

        # Apply the inital rotations
        self.lidar_front.rotate_lidar_by_delta(self.aLidarRotFront, self.lidar_front.x0, self.lidar_front.y0)
        self.lidar_left.rotate_lidar_by_delta(self.aLidarRotL, self.lidar_left.x0, self.lidar_left.y0)
        self.lidar_right.rotate_lidar_by_delta(self.aLidarRotR, self.lidar_right.x0, self.lidar_right.y0)


    def update_lidars(self, aRotFront: float=0.0, aRotL: float=0.0, aRotR: float=0.0):
        """
            Apply any rotation to the lidar (relative to the vehicle) and fire the liders
        """
        # apply the rotations
        if self.tLastLidarUpdate is None:
            self.tLastLidarUpdate = time.time()
            tElapsed = 0.0
        else:
            tNow = time.time()
            tElapsed = tNow - self.tLastLidarUpdate
            self.tLastLidarUpdate = tNow

        if tElapsed > 0.0:
            # front
            daRotFrontRaw = (aRotFront * np.pi / 180 - self.aLidarRotFront) / tElapsed
            if abs(daRotFrontRaw) > 0.0:
                daRotFront = self.process_lidar_rotation(daRotFrontRaw, self.aLidarRotFront, self.aRotLimFront, tElapsed)
                if abs(daRotFront) > 0.0:
                    self.lidar_front.rotate_lidar_by_delta(daRotFront, self.lidar_front.x0, self.lidar_front.y0)
                    self.aLidarRotFront += daRotFront

            # left
            daRotLRaw = (aRotL * np.pi / 180 - self.aLidarRotL) / tElapsed
            if abs(daRotLRaw) > 0.0:
                daRotL = self.process_lidar_rotation(daRotLRaw, self.aLidarRotL, self.aRotLimL, tElapsed)
                if abs(daRotL) > 0.0:
                    self.lidar_left.rotate_lidar_by_delta(daRotL, self.lidar_left.x0, self.lidar_left.y0)
                    self.aLidarRotL += daRotL

            # right
            daRotRRaw = (aRotR * np.pi / 180 - self.aLidarRotR) / tElapsed
            if abs(daRotRRaw) > 0.0:
                daRotR = self.process_lidar_rotation(daRotRRaw, self.aLidarRotR, self.aRotLimR, tElapsed)
                if abs(daRotR) > 0.0:
                    self.lidar_right.rotate_lidar_by_delta(daRotR, self.lidar_right.x0, self.lidar_right.y0)
                    self.aLidarRotR += daRotR

        # fire the lidar rays
        self.lidar_front.fire_lidar()
        self.lidar_left.fire_lidar()
        self.lidar_right.fire_lidar()

    def process_lidar_rotation(self, daRotRaw: float, aRotCurrent: float, aRotMax: float, tElapsed: float):
        """
            Process the lidar rotation and return a delta angle
        """
        # limit the rate of change
        if abs(daRotRaw) > self.config['daLidarRotMax']:
            if daRotRaw < 0.0:
                daRotRaw = -1 * self.config['daLidarRotMax']
            else:
                daRotRaw = self.config['daLidarRotMax']

        daRotRaw = daRotRaw * tElapsed

        aRotNew = aRotCurrent + daRotRaw

        if aRotNew < -1* aRotMax:
            aRotNew = -1 * aRotMax
        elif aRotNew > aRotMax:
            aRotNew = aRotMax

        return aRotNew - aRotCurrent


    def initialise_vehicle_colliders(self):
        """
            Set up the collider objects to detect track limit collision
        """
        # colliders will be the vehicle body limits defined by the FL/FR/RL/RR offsets
        self.colliders = []
        self.colliders.append(Line(tuple(self.carRLOffset), tuple(self.carFLOffset)))
        self.colliders.append(Line(tuple(self.carFLOffset), tuple(self.carFROffset)))
        self.colliders.append(Line(tuple(self.carFROffset), tuple(self.carRROffset)))
        self.colliders.append(Line(tuple(self.carRROffset), tuple(self.carRLOffset)))

        # add a collision circle, this will by used to minimise the number of track
        # line segements that are considered for collision
        self.collisionCircle = Circle(0, 0, 5)

        # initialise the has collided flag
        self.bHasCollided = False

    def check_for_vehicle_collision(self):
        """
            Check for vehicle collision with the track, if collision detected then
            state are reset
        """
        # find the indexes of the lines to check for collision
        in_idxs, out_idxs = self.track.get_line_idxs_for_collision(self.collisionCircle)

        self.bHasCollided = False

        # check the inner track
        if len(in_idxs) > 0:
            check_lines = [self.track.data.in_lines[i] for i in in_idxs]
            for l in self.colliders:
                collision_check = [cl for cl in check_lines if self.get_collision_state(l,cl)]
                if len(collision_check) > 0:
                    # the vehicle has collided, no need to check the other colliders
                    self.bHasCollided = True
                    # calculate the angle between the last track segment we collided with and the
                    # vehicle forward direction unit vector. Use this to reset the vehicle heading
                    aCollision = calc_angle_between_unit_vectors(self.h.v_hat,collision_check[-1].v_hat)
                    break

        # check the outer track, only if the inner hasn't already collided
        if (len(out_idxs) > 0) and not self.bHasCollided:
            check_lines = [self.track.data.out_lines[i] for i in out_idxs]
            for l in self.colliders:
                collision_check = [cl for cl in check_lines if self.get_collision_state(l,cl)]
                if len(collision_check) > 0:
                    # the vehicle has collided, no need to check the other colliders
                    self.bHasCollided = True
                    # calculate the angle between the last track segment we collided with and the
                    # vehicle forward direction unit vector. Use this to reset the vehicle heading
                    # collision with the outer track needs a negative scaling on the angle
                    aCollision = -1 * calc_angle_between_unit_vectors(self.h.v_hat,collision_check[-1].v_hat)
                    break

        if self.bHasCollided:
            # move the car back by 2 * dposVehicle to give the driver a chance to recover
            self.apply_manual_translation(-2 * self.dxVehicle, -2 * self.dyVehicle)
            # realign the car so it's paralled with the track segment
            self.apply_manual_rotation(aCollision)

            # reset the vehicle states
            self.reset_states()

            # add a lap time penalty in the track
            self.track.add_lap_time_penalty(10.0)

    def apply_manual_translation(self, dX: float, dY):
        """
            Apply a manual translation to all vehicle objects - mainly used for collisions
        """
        # vehicle position
        self.posVehicle[0] += dX
        self.posVehicle[1] += dY

        # vehicle heading vector
        self.h.translate_line_by_delta(dX, dY)

        # vehicle collision objects
        for l in self.colliders:
            l.translate_line_by_delta(dX, dY)
        self.collisionCircle.update_centre_by_delta(dX, dY)

        # lidars
        self.lidar_front.translate_lidars_by_delta(dX, dY)
        self.lidar_left.translate_lidars_by_delta(dX, dY)
        self.lidar_right.translate_lidars_by_delta(dX, dY)

    def apply_manual_rotation(self, daRot: float):
        """
            Apply a manual rotation to all vehicle objects - mainly used for collisions
        """
        # vehicle yaw
        self.aYaw += daRot

        # vehicle heading vector
        self.h.rotate_line_by_delta(daRot, self.posVehicle[0], self.posVehicle[1])

        # vehicle collision objects
        for l in self.colliders:
            l.rotate_line_by_delta(daRot, self.posVehicle[0], self.posVehicle[1])

        # lidars
        self.lidar_front.rotate_lidar_by_delta(daRot, self.posVehicle[0], self.posVehicle[1])
        self.lidar_left.rotate_lidar_by_delta(daRot, self.posVehicle[0], self.posVehicle[1])
        self.lidar_right.rotate_lidar_by_delta(daRot, self.posVehicle[0], self.posVehicle[1])

    def get_collision_state(self, l1, l2):
        """
            Returns true if the vehicle collider has collided with a track line segement
        """
        do_intersect, _ = check_for_intersection_lineseg_lineseg(l1, l2)
        return do_intersect

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
        self.daYaw = 0.0
        self.nYaw = 0.0
        self.ndotYaw = 0.0
        self.dxVehicle = 0.0
        self.dyVehicle = 0.0


    def reset_vehicle_position(self):
        """
            Reset the vehicle position to (0,0)
        """
        self.posVehicle = np.array([0.0, 0.0], dtype=np.float64)
        self.aYaw = 0.0
        self.h = Line(tuple(self.posVehicle), (1.0, 0.0))  # provides a unit vector for vehicle heading
        # reinitialise the vehicle colliders
        self.initialise_vehicle_colliders()
        self.initialise_lidars(aRotL=self.aLidarRotL, aRotR=self.aLidarRotR, aRotFront=self.aLidarRotFront)

    def initialise_vehicle_properties(self):
        """
            Initialise some more properties using the config
        """
        self.xCOGF = self.config['xWheelBase'] * self.config['rCOGLongR']
        self.xCOGR = self.config['xWheelBase'] * (1 - self.config['rCOGLongR'])
        self.FBrakingMax = self.config['mVehicle'] * self.config['gGravity'] * 0.8  # leave some capacity for lat

    def set_driver_inputs(self, rThrottlePedalDemand: float, rBrakePedalDemand: float, aSteeringWheelDemand: float):
        """
            Update the steering inputs
        """
        #
        # Check the limits of inputs
        #
        rThrottlePedalDemand = max(0.0, min(1.0, rThrottlePedalDemand))
        rBrakePedalDemand = max(0.0, min(1.0, rBrakePedalDemand))
        aSteeringWheelDemand = max(self.config['aSteeringWheelMin'], min(self.config['aSteeringWheelMax'], aSteeringWheelDemand))
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
            if tElapsed != 0.0:
                drThrottlePedal = (rThrottlePedalDemand - self.rThrottlePedal) / tElapsed
                drBrakePedal = (rBrakePedalDemand - self.rBrakePedal) / tElapsed
                nSteeringWheel = (aSteeringWheelDemand - self.aSteeringWheel) / tElapsed
            else:
                drThrottlePedal = 0.0
                drBrakePedal = 0.0
                nSteeringWheel = 0.0
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
        MMax = np.interp(self.nWheelR * 30 / np.pi, self.config['nWheel_BRP'], self.config['MPowertrainMax_LU'])
        MMin = np.interp(self.nWheelR * 30 / np.pi, self.config['nWheel_BRP'], self.config['MPowertrainMin_LU'])
        MPowertrain = (MMax - MMin) * self.rThrottlePedal + MMin
        #
        # calculate the braking demand
        #
        FBrakeTotal = self.FBrakingMax * self.rBrakePedal * np.sign(self.vxVehicle)
        FBrakeF = FBrakeTotal * (1 - self.config['rCOGLongR'])  # account for COG pos, weight transfer not modelled
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
        # print(MPowertrain, self.gxVehicle, self.FTyreXTotalF, self.FTyreXTotalR,self.rThrottlePedal,self.vxVehicle)

    def update_lat_dynamics(self):
        """
            Update the lateral dynamics of the vehicle
        """
        # determine the driver demands
        self.aSteer = self.aSteeringWheel / self.config['rSteeringRatio'] * np.pi / 180

        # calculate the slip ratios
        # print('aSteer:',self.aSteer,'vy:',self.vyVehicle,'x',self.xCOGF,'nYaw:',self.nYaw,'vx:',self.vxVehicle)
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

            # if abs(self.vyVehicle) < self.config['vVehicleStationary']:
            #    self.vyVehicle = self.config['vVehicleStationary']
            if abs(self.vxVehicle) <= self.config['vVehicleStationary']:
                self.vyVehicle = 0.0
            self.nYaw += self.ndotYaw * tElapsed

        # calculate velocity and body slip
        self.aBodySlip = np.arctan(self.vyVehicle / abs(self.vxVehicle))
        self.vVehicle = np.sqrt(self.vxVehicle**2 + self.vyVehicle**2)
        # print('vy:',self.vyVehicle,'nYaw:', self.nYaw, 'aSlipF:', self.aSlipF, 'aSlipR:', self.aSlipR)

    def get_vehicle_sensors(self):
        """
            Returns the vehicle sensors for visualisation and sending throug udp to driver
        """
        return {'gxVehicle': self.gxVehicle, 'gyVehicle': self.gyVehicle,
                'vxVehicle': self.vxVehicle, 'vyVehicle': self.vyVehicle,
                'aSlipF': self.aSlipF, 'aSlipR': self.aSlipR,
                'aBodySlip': self.aBodySlip, 'aYaw': self.aYaw,
                'rThrottlePedal': self.rThrottlePedal, 'rBrakePedal': self.rBrakePedal,
                'aSteeringWheel': self.aSteeringWheel, 'nYaw': self.nYaw,
                'xVehicle': self.posVehicle[0], 'yVehicle': self.posVehicle[1],
                'aLidarRotL': self.aLidarRotL, 'aLidarRotR': self.aLidarRotR, 'aLidarRotFront': self.aLidardRotFront}

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

            # update heading
            self.daYaw = self.nYaw * tElapsed
            self.aYaw += self.daYaw

            # update position
            sVehicleMoved = self.vVehicle * tElapsed
            self.dxVehicle = sVehicleMoved * np.cos(self.aYaw + self.aBodySlip)
            self.dyVehicle = sVehicleMoved * np.sin(self.aYaw + self.aBodySlip)
            self.posVehicle[0] += self.dxVehicle  # xVehicle
            self.posVehicle[1] += self.dyVehicle  # yVehicle

            # update heading vector
            self.h.translate_line_by_delta(self.dxVehicle, self.dyVehicle)
            self.h.rotate_line_by_delta(self.daYaw, self.posVehicle[0], self.posVehicle[1])

            # update any vehicle child objects
            # collision objects
            for l in self.colliders:
                # translate the line based on vehicle movement
                l.translate_line_by_delta(self.dxVehicle, self.dyVehicle)
                # rotate the line based on change in vehicle yaw
                l.rotate_line_by_delta(self.daYaw, self.posVehicle[0], self.posVehicle[1])
            self.collisionCircle.update_centre_by_delta(self.dxVehicle, self.dyVehicle)
            # lidars
            self.lidar_front.translate_lidars_by_delta(self.dxVehicle, self.dyVehicle)
            self.lidar_left.translate_lidars_by_delta(self.dxVehicle, self.dyVehicle)
            self.lidar_right.translate_lidars_by_delta(self.dxVehicle, self.dyVehicle)
            self.lidar_front.rotate_lidar_by_delta(self.daYaw, self.posVehicle[0], self.posVehicle[1])
            self.lidar_left.rotate_lidar_by_delta(self.daYaw, self.posVehicle[0], self.posVehicle[1])
            self.lidar_right.rotate_lidar_by_delta(self.daYaw, self.posVehicle[0], self.posVehicle[1])
