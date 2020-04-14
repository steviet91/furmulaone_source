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

        # get the screen resolution and set as image size
        self.img_w = 1000
        self.img_h = 1000
        self.img_x_buffer = 100
        self.img_y_buffer = 100
        self.img_scale = 0.1
        self.orig_img = np.zeros((self.img_w, self.img_h, 3), np.uint8)
        self.baseColour = (0, 255, 179)
        self.lidarColour = (29, 142, 249)

        # camera
        self.kCameraSpring = 1.0
        self.cCameraDamper = 8.0
        self.mCamera = 2.0
        self.mVehicle = 1.0
        self.mu_camera = 1.0
        self.xCameraVehicle = 0.0
        self.xdotCameraVehicle = 0.0
        self.tLastCamUpdate = None
        self.cameraPosOrigin = np.array([500.0, 500.0])
        self.carPos = np.array([0, 0]) + self.cameraPosOrigin
        # set the camera position equal to the vehicle (scaled properly)
        self.cameraPos = self.vehicle.posVehicle / self.img_scale
        self.reset_camera()

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
        cv.drawContours(self.show_img, [box], 0, self.baseColour, 2)
        cv.circle(self.show_img, (int(self.carPos[0]), int(self.carPos[1])), 2, self.baseColour, -1)
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
                xdotCameraVehicle = (xCameraVehicle - self.xCameraVehicle) / tElapsed
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
            cv.putText(self.show_img, s, pos, cv.FONT_HERSHEY_SIMPLEX, 0.75, self.baseColour, thickness)

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
                cv.line(self.show_img, tuple(p1.astype(np.int32)), tuple(p2.astype(np.int32)), self.lidarColour)
                #pos = (r.p1 + r.v_hat * lidar.collision_array[i] / 2 ) / self.img_scale + self.cameraPosOrigin
                #cv.putText(self.show_img, '{:.2f} m'.format(lidar.collision_array[i]), tuple(pos.astype(np.int32)), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness)


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
            cv.line(self.show_img, tuple(p1.astype(np.int32)), tuple(p2.astype(np.int32)), self.baseColour)
            for ii in range(1,20):
                pos = (l.p1 + l.v * ii / 20) / self.img_scale + self.cameraPosOrigin - self.cameraPos
                cv.putText(self.show_img, str(i + 1), tuple(pos.astype(np.int32)), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness)
        # render the out lines
        for i,l in enumerate(self.track.data.out_lines):
            # map the line coordinates to the image
            p1 = (l.p1 / self.img_scale) + self.cameraPosOrigin - self.cameraPos
            p2 = (l.p2 / self.img_scale) + self.cameraPosOrigin - self.cameraPos
            cv.line(self.show_img, tuple(p1.astype(np.int32)), tuple(p2.astype(np.int32)), self.baseColour)
            for ii in range(1,20):
                pos = (l.p1 + l.v * ii / 20) / self.img_scale + self.cameraPosOrigin - self.cameraPos
                cv.putText(self.show_img, str(i + 1), tuple(pos.astype(np.int32)), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness)

    def render_image(self):
        cv.imshow('Vis', self.show_img)
        cv.waitKey(1)
