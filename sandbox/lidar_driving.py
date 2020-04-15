from time import sleep
from src.network import DriverInputsSend
from src.network import VehicleOutputsRecv
from sandbox.game_pad_inputs import GamePad
import time
import numpy as np
import os
import json

def main():
    s = DriverInputsSend()
    vo = VehicleOutputsRecv()
    gp = GamePad()

    rThrP = 0.0
    rThrI = 0.0
    kThrP = 1
    kThrI = 0.5
    use_ff_thr = True # use feedforward element

    aSteerP = 0.0
    aSteerI = 0.0
    kSteerI = 0.0
    kSteerP = 25
    kSteerPBrake = 200

    vxTarg = 20  # 2 m/s vxVehicle target

    tLast = None

    brake_dist = 50
    vxBraking_thresh = 2

    # get the path to the script
    module_path = os.path.dirname(os.path.abspath(__file__))

    # load the vehicle config
    with open(module_path + '/../setup/vehicle_config.json', 'r') as f:
        veh_config = json.load(f)

    while True:
        # check quit
        if gp.quit_requested:
            s.set_quit(True)
            s.send_data()
            gp.exit_thread()
            vo.exit()
            break

        # check reset
        if gp.reset_requested:
            s.set_reset(True)
            # reset the request
            gp.reset_requested = False
            # reset the integral terms
            aSteerI = 0.0
            rThrI = 0.0
        else:
            s.set_reset(False)

        # get the vehicle data
        vo.check_network_data()
        data = vo.get_all_data()

        # get the elpsed time
        if tLast is None:
            tLast = time.time()
            tElapsed = 0.0
        else:
            t = time.time()
            tElapsed = t - tLast
            tLast = t

        # detect a corner using the middle three lidar rays from the front
        if data['tMessageTimeStamp'] is not None:
            xColF = np.array(data['xLidarCollisionFront'])
            midRayIdx = int(np.floor((len(xColF) / 2)))
            if (midRayIdx % 2) == 0:
                idxs = range(midRayIdx-1,midRayIdx+3)
            else:
                idxs = range(midRayIdx-1,midRayIdx+2)
            xColF = xColF[idxs]
            # get the average
            xColF = xColF[np.where(xColF > 0)[0]]
            if len(xColF) == 0:
                bUseBrakes = False
                s.set_brake(0.0)
            else:
                xF = np.mean(xColF)
                if xF < brake_dist and data['vxVehicle'] >= vxBraking_thresh:
                    bUseBrakes = True
                    s.set_brake(1.0)
                else:
                    bUseBrakes = False
                    s.set_brake(0.0)
        else:
            bUseBrakes = False
            s.set_brake(0.0)

        # set the throttle pedal on a PI + ff
        if not bUseBrakes:
            if data['tMessageTimeStamp'] is not None:
                vErr = vxTarg - data['vxVehicle']
                rThrP = vErr * kThrP
                rThrI += tElapsed * vErr * kThrI
                rThrI = max(0, min(1.0, rThrI))
                if use_ff_thr:
                    MResFF = veh_config['rRollRadR'] * (vxTarg * veh_config['Cb'] + 0.5 * (vxTarg**2 * veh_config['rhoAir'] + veh_config['CdA']))
                    MMax = np.interp(vxTarg / veh_config['rRollRadR'], veh_config['nWheel_BRP'], veh_config['MPowertrainMax_LU'])
                    MMin = np.interp(vxTarg / veh_config['rRollRadR'], veh_config['nWheel_BRP'], veh_config['MPowertrainMin_LU'])
                    rThr_ff = (MResFF - MMin) / (MMax - MMin)
                    rThr =  rThrP + rThrI + rThr_ff
                else:
                    rThr =  rThrP + rThrI
                s.set_throttle(rThr)
            else:
                vErr = 0.0
                s.set_throttle(0.0)
                rThrI = 0.0
        else:
            vErr = 0.0
            s.set_throttle(0.0)
            rThrI = 0.0


        # set the steering on a PI - target the middle of the track
        xColL = np.array(data['xLidarCollisionL'])
        xColL = xColL[np.where(xColL > 0)[0]]
        xColR = np.array(data['xLidarCollisionR'])
        xColR = xColR[np.where(xColR > 0)[0]]
        if len(xColL) > 0 and len(xColL) > 0:
            xToLeft = np.mean(xColL) # distance to LHS of track
            xToRight = np.mean(xColR) # distance to RHS of track
            rMiddleErr = xToRight / (xToLeft + xToRight) - 0.5 # -ve on right, + to left
            if bUseBrakes:
                aSteerP = rMiddleErr * kSteerPBrake
            else:
                aSteerP = rMiddleErr * kSteerP
            aSteerI += tElapsed * rMiddleErr * kSteerI
            aSteerI = max(-20, min(20, aSteerI))
            aSteer = aSteerI + aSteerP
            s.set_steering(aSteer)
        else:
            rMiddleErr = 0.0
            s.set_steering(0.0)
            aSteerI = 0.0

        # send the inputs
        s.send_data()
        print('vxErr:', vErr, 'rToMiddle', rMiddleErr)
        print(s.data_dict)

        sleep(0.1)

if __name__ == "__main__":
    main()
