import cv2 as cv
import numpy as np
from .track import TrackHandler
from .vehicle import Vehicle
from .geom import rotate_point
from .geom import calc_euclid_distance_2d
from .lidar import Lidar
import time


class Vis(object):

    def __init__(self, track: TrackHandler, vehicle: Vehicle):
        # save the arguements
        self.track = track
        self.vehicle = vehicle

        # Load settings in the config.json
        self.load_config()
        # get the screen resolution and set as image size
        self.img_w = self.config['img']['w']
        self.img_h = self.config['img']['h']
        self.img_x_buffer = self.config['img']['x_buffer']
        self.img_y_buffer = self.config['img']['y_buffer']
        self.img_scale = self.config['img']['scale']
        self.orig_img = np.zeros((self.img_h, self.img_w, 3), np.uint8)
        # image colours
        self.colours = self.config['colours']

        # camera
        self.kCameraSpring = self.config['camera']['kCameraSpring']
        self.cCameraDamper = self.config['camera']['cCameraDamper']
        self.mCamera = self.config['camera']['mCamera']
        self.mVehicle = self.config['camera']['mVehicle']
        self.mu_camera = self.config['camera']['mu_camera']
        self.xCameraVehicle = self.config['camera']['xCameraVehicle']
        self.xdotCameraVehicle = self.config['camera']['xdotCameraVehicle']
        self.cameraPosOrigin = np.array(self.config['camera']['cameraPosOrigin'])
        self.tLastCamUpdate = None
        self.carPos = np.array([0, 0]) + self.cameraPosOrigin
        # set the camera position equal to the vehicle (scaled properly)
        self.cameraPos = self.vehicle.posVehicle / self.img_scale
        self.reset_camera()

    def load_config(self):
        # find the scripts path
        import os
        self.module_path = os.path.dirname(os.path.abspath(__file__))
        # read in the config
        import json
        with open(self.module_path + '/../setup/vis_config.json','r') as f:
            self.config = json.load(f)

    def reset_camera(self):
        """
            Reset the camera states
        """
        self.vxCamera = 0.0
        self.vyCamera = 0.0
        self.vCamera = 0.0

    def draw_car(self):

        # create a copy of the base image
        self.show_img = np.copy(self.orig_img)

        # translate the visual based on the new position
        self.carPos = self.vehicle.posVehicle / self.img_scale + self.cameraPosOrigin - self.cameraPos

        carRL = self.vehicle.colliders[0].p1 / self.img_scale + self.cameraPosOrigin - self.cameraPos
        carFL = self.vehicle.colliders[0].p2 / self.img_scale + self.cameraPosOrigin - self.cameraPos
        carFR = self.vehicle.colliders[2].p1 / self.img_scale + self.cameraPosOrigin - self.cameraPos
        carRR = self.vehicle.colliders[2].p2 / self.img_scale + self.cameraPosOrigin - self.cameraPos

        # draw the car
        rect = cv.minAreaRect(np.array((carFL, carFR, carRR, carRL), dtype=np.float32))
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(self.show_img, [box], 0, self.colours['base'], 2)
        cv.circle(self.show_img, (int(self.carPos[0]), int(self.carPos[1])), 2, self.colours['base'], -1)
        cv.circle(self.show_img, (int(self.carPos[0]), int(self.carPos[1])), int(self.vehicle.collisionCircle.r / self.img_scale), (255, 0, 0))

    def update_camera_position(self):
        """
            Check the proximity of the car to the screen edges, update the translation
        """
        # determine the elapsed time
        if self.tLastCamUpdate is None:
            self.tLastCamUpdate = time.time()
            bElapsedTimeAvailable = False  # no elapsed time yet
        else:
            tNow = time.time()
            tElapsed = tNow - self.tLastCamUpdate
            self.tLastCamUpdate = tNow
            bElapsedTimeAvailable = True

        # mass, spring, damper method
        # Force acting on the camera
        if self.vehicle.bHasCollided:
            self.reset_camera()
        else:

            # determine current car position relative to (0, 0) in the image
            carPos = self.vehicle.posVehicle / self.img_scale

            # calculate the spring length
            xCameraVehicle = calc_euclid_distance_2d(tuple(carPos), tuple(self.cameraPos))

            # calcuate the velocity delta between the car and camera
            if bElapsedTimeAvailable:
                if tElapsed != 0.0:
                    xdotCameraVehicle = (xCameraVehicle - self.xCameraVehicle) / tElapsed
                else:
                    # div by zero protection
                    xdotCameraVehicle = 0.0
                self.xCameraVehicle = xCameraVehicle
            else:
                xdotCameraVehicle = 0.0

            # calculate the resultant force on the camera
            FCamera = max(self.kCameraSpring * xCameraVehicle + xdotCameraVehicle * self.cCameraDamper - self.mu_camera * self.mCamera * 9.81, 0.0)


            # calculate the angle between the camera and vehicle
            aCamVeh = np.arctan2(carPos[1] - self.cameraPos[1], carPos[0] - self.cameraPos[0])
            # calculate the component forces
            FxCamera = FCamera * np.cos(aCamVeh)
            FyCamera = FCamera * np.sin(aCamVeh)
            # calculate the accelerations
            gxCamera = FxCamera / self.mCamera
            gyCamera = FyCamera / self.mCamera

            # integrate the accelerations
            if bElapsedTimeAvailable:
                self.vxCamera += gxCamera * tElapsed
                self.vyCamera += gyCamera * tElapsed
                self.vCamera = np.sqrt(self.vxCamera**2 + self.vyCamera**2)
                self.cameraPos[0] += self.vxCamera * tElapsed
                self.cameraPos[1] += self.vyCamera * tElapsed

    def draw_data(self):
        """
            Draw the telemetry
        """
        t = self.vehicle.get_vehicle_sensors()
        thickness = 2
        x_start = self.img_x_buffer / 10
        y_offset = 75
        y_start = self.img_y_buffer
        i = 0
        for k, v in t.items():
            pos = (int(x_start), int(y_start + y_offset * i))
            i += 1
            s = k + ':{:.5f}'.format(v)
            cv.putText(self.show_img, s, pos, cv.FONT_HERSHEY_SIMPLEX, 0.75, self.colours['base'], thickness)

    def draw_demands(self):
        """
            Draw a visualisation for the driver demands (throttle, brake, steer)
        """
        # Get actual vehicle data (i.e. the values being applied to the vehicle model, not the driver demands)
        actual_inputs = self.vehicle.get_vehicle_sensors()
        # Position of the HUD
        hud_dims = self.config['hud']
        bar_height = hud_dims['height'] - 2*hud_dims['padding']
        # hud_border = cv.rectangle(self.show_img, (hud_dims['x'], hud_dims['y']), (hud_dims['x'] + hud_dims['width'], hud_dims['y']+ hud_dims['height']), self.colours['hud']['outline'], thickness=1)
        hud_border = cv.rectangle(self.show_img, (0, self.img_h-1), (self.img_w-1, self.img_h - hud_dims['height']), self.colours['hud']['outline'], thickness=1)
        hud_bg = cv.rectangle(self.show_img, (1, self.img_h-2), (self.img_w-2, self.img_h - hud_dims['height'] + 1), self.colours['hud']['bg'], thickness=-1)
        # Throttle
        throttle_x1 = hud_dims['x'] + hud_dims['padding']
        throttle_y1 = self.img_h-1 - hud_dims['padding']
        throttle_x2 = throttle_x1 + hud_dims['bar_width']
        throttle_y2 = throttle_y1 - (hud_dims['height'] - 2*hud_dims['padding'])
        throttle_border = cv.rectangle(self.show_img, (throttle_x1,throttle_y1), (throttle_x2,throttle_y2),self.colours['hud']['outline'], thickness=1)
        # Note: y=0 is the top of the screen, so need to calculate 'height' of the bar as y2 - value
        throttle_y2_actual = throttle_y1 - int(bar_height * actual_inputs['rThrottlePedal'])
        throttle_actual = cv.rectangle(self.show_img, (throttle_x1,throttle_y1), (throttle_x2,throttle_y2_actual),self.colours['hud']['throttle'], thickness=-1)
        # Brake
        brake_x1 = throttle_x2 + hud_dims['padding']
        brake_y1 = throttle_y1
        brake_x2 = brake_x1 + hud_dims['bar_width']
        brake_y2 = throttle_y2
        brake_border = cv.rectangle(self.show_img, (brake_x1,brake_y1), (brake_x2,brake_y2),self.colours['hud']['outline'], thickness=1)
        brake_y2_actual = brake_y1 - int(bar_height * actual_inputs['rBrakePedal'])
        brake_actual = cv.rectangle(self.show_img, (brake_x1,brake_y1), (brake_x2,brake_y2_actual),self.colours['hud']['brake'], thickness=-1)
        # Steering
        steer_x1 = brake_x2 + hud_dims['padding']
        steer_y1 = int(self.img_h - (hud_dims['height'] / 2) - hud_dims['bar_width']/2)
        steer_x2 = steer_x1 + hud_dims['steering_bar_width']
        steer_y2 = steer_y1 + hud_dims['bar_width']
        steer_xMid = int((steer_x1 + steer_x2)/2)
        steer_border = cv.rectangle(self.show_img, (steer_x1,steer_y1), (steer_x2,steer_y2),self.colours['hud']['outline'], thickness=1)
        steer_centreline = cv.line(self.show_img, (steer_xMid, steer_y1-hud_dims['steering_centreline_overshoot']),(steer_xMid, steer_y2+hud_dims['steering_centreline_overshoot']) , self.colours['hud']['outline'], thickness=1)
        steer_bar_width = steer_x2 - steer_x1
        steer_x1_actual = steer_xMid + int(steer_bar_width/2 * actual_inputs['aSteeringWheel'] / self.vehicle.config['aSteeringWheelMax'])
        steer_actual = cv.rectangle(self.show_img, (steer_x1_actual,steer_y1), (steer_xMid,steer_y2),self.colours['hud']['steer'], thickness=-1)
        # Add vCar in lieu of proper speedo
        font = getattr(cv,hud_dims['font'])
        vCar = np.sqrt(actual_inputs['vxVehicle']**2 + actual_inputs['vyVehicle']**2)
        text = 'vCar:{0:>5.1f} m/s'.format(vCar)
        text_size = cv.getTextSize(text, font, hud_dims['font_scale'], 2)[0]
        v_car_x = steer_xMid - int(text_size[0]/2)
        v_car_y = throttle_y1
        cv.putText(self.show_img, text, (v_car_x, v_car_y), font, hud_dims['font_scale'], self.colours['hud']['outline'],)

    def draw_all_lidars(self):
        """
            Draw the all lidars
        """
        self.draw_lidar(self.vehicle.lidar_front)
        self.draw_lidar(self.vehicle.lidar_left)
        self.draw_lidar(self.vehicle.lidar_right)

    def draw_lidar(self, lidar: Lidar):
        """
            Draw all rays that have hit a point and put distance on ray
        """
        thickness = 1
        for i,r in enumerate(lidar.rays):
            if lidar.collision_array[i] > 0:
                # the lidar ray scored a hit
                p1 = r.p1 / self.img_scale + self.cameraPosOrigin - self.cameraPos
                p2 = (r.p1 + r.v_hat * lidar.collision_array[i]) / self.img_scale + self.cameraPosOrigin - self.cameraPos
                cv.line(self.show_img, tuple(p1.astype(np.int32)), tuple(p2.astype(np.int32)), self.colours['lidar'])
                #pos = (r.p1 + r.v_hat * lidar.collision_array[i] / 2 ) / self.img_scale + self.cameraPosOrigin
                #cv.putText(self.show_img, '{:.2f} m'.format(lidar.collision_array[i]), tuple(pos.astype(np.int32)), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness)
        p0 = np.array([lidar.collisionCircle.x0, lidar.collisionCircle.y0]) / self.img_scale + self.cameraPosOrigin - self.cameraPos
        cv.circle(self.show_img, tuple(p0.astype(np.int32)), int(lidar.collisionCircle.r / self.img_scale), (255, 0, 0))


    def draw_track(self):
        """
            Draw the track onto the image
        """
        thickness = 1
        # render the in lines
        for i,l in enumerate(self.track.data.in_lines):
            # map the line coordinates to the image
            p1 = (l.p1 / self.img_scale) + self.cameraPosOrigin - self.cameraPos
            p2 = (l.p2 / self.img_scale) + self.cameraPosOrigin - self.cameraPos
            cv.line(self.show_img, tuple(p1.astype(np.int32)), tuple(p2.astype(np.int32)), self.colours['base'])
            for ii in range(1,20):
                pos = (l.p1 + l.v * ii / 20) / self.img_scale + self.cameraPosOrigin - self.cameraPos
                cv.putText(self.show_img, str(i + 1), tuple(pos.astype(np.int32)), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness)
        # render the out lines
        for i,l in enumerate(self.track.data.out_lines):
            # map the line coordinates to the image
            p1 = (l.p1 / self.img_scale) + self.cameraPosOrigin - self.cameraPos
            p2 = (l.p2 / self.img_scale) + self.cameraPosOrigin - self.cameraPos
            cv.line(self.show_img, tuple(p1.astype(np.int32)), tuple(p2.astype(np.int32)), self.colours['base'])
            for ii in range(1,20):
                pos = (l.p1 + l.v * ii / 20) / self.img_scale + self.cameraPosOrigin - self.cameraPos
                cv.putText(self.show_img, str(i + 1), tuple(pos.astype(np.int32)), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness)

    def render_image(self):
        cv.imshow('Vis', self.show_img)
        cv.waitKey(1)
