import cv2 as cv
import numpy as np
from .track import TrackHandler
from .vehicle import Vehicle
from .geom import rotate_point


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
        self.rCOGLongR = self.vehicle.config['rCOGLongR']
        self.xVehicleWidth = self.vehicle.config['xVehicleWidth'] / self.img_scale
        self.xVehicleLength = self.vehicle.config['xVehicleLength'] / self.img_scale
        self.originOffset = np.array([500.0, 500.0])
        self.carPos = np.array([0, 0]) + self.originOffset
        self.carFLOffset = np.array([self.xVehicleLength * self.rCOGLongR, -0.5 * self.xVehicleWidth])
        self.carFROffset = np.array([self.xVehicleLength * self.rCOGLongR, 0.5 * self.xVehicleWidth])
        self.carRLOffset = np.array([-1 * self.xVehicleLength * (1 - self.rCOGLongR), -0.5 * self.xVehicleWidth])
        self.carRROffset = np.array([-1 * self.xVehicleLength * (1 - self.rCOGLongR), 0.5 * self.xVehicleWidth])
        self.orig_img = np.zeros((self.img_w, self.img_h, 3), np.uint8)
        self.baseColour = (0, 255, 179)

    def draw_car(self):

        # create a copy of the base image
        self.show_img = np.copy(self.orig_img)

        # translate the visual based on the new position
        """
        posConverted = self.vehicle.posVehicle / self.img_scale + self.originOffset
        deltaPos = posConverted - self.carPos
        self.carPos += deltaPos
        """
        self.carPos = self.vehicle.posVehicle / self.img_scale + self.originOffset

        carRL = self.vehicle.colliders[0].p1 / self.img_scale + self.originOffset
        carFL = self.vehicle.colliders[0].p2 / self.img_scale + self.originOffset
        carFR = self.vehicle.colliders[2].p1 / self.img_scale + self.originOffset
        carRR = self.vehicle.colliders[2].p2 / self.img_scale + self.originOffset

        # draw the car
        rect = cv.minAreaRect(np.array((carFL, carFR, carRR, carRL), dtype=np.float32))
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(self.show_img, [box], 0, self.baseColour, 2)
        cv.circle(self.show_img, (int(self.carPos[0]), int(self.carPos[1])), 2, self.baseColour, -1)
        cv.circle(self.show_img, (int(self.carPos[0]), int(self.carPos[1])), int(self.vehicle.collisionCircle.r / self.img_scale), (255, 0, 0))

        # check the proximity of the car to the screen edges, update the translation
        # x
        if self.carPos[0] < self.img_x_buffer:
            # about to disappear to the left of the screen, shift if to the right
            self.originOffset[0] += self.img_x_buffer - self.carPos[0]
            self.carPos[0] += self.img_x_buffer - self.carPos[0]
        elif (self.img_w - self.carPos[0]) < self.img_x_buffer:
            # about to disappear to the right of the screen, shift it to the left
            self.originOffset[0] -= self.img_x_buffer - (self.img_w - self.carPos[0])
            self.carPos[0] -= self.img_x_buffer - (self.img_w - self.carPos[0])
        # y
        if self.carPos[1] < self.img_y_buffer:
            # about to disappear to the top of the screen, shift it to the bottom
            self.originOffset[1] += self.img_y_buffer - self.carPos[1]
            self.carPos[1] += self.img_y_buffer - self.carPos[1]
        elif (self.img_h - self.carPos[1]) < self.img_y_buffer:
            # about to disappear to the bottom of the screen, shift it to the top
            self.originOffset[1] -= self.img_y_buffer - (self.img_h - self.carPos[1])
            self.carPos[1] -= self.img_y_buffer - (self.img_h - self.carPos[1])
        # print('Orig: ',cPosOrig,'New: ',self.carPos)

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

    def draw_track(self):
        """
            Draw the track onto the image
        """
        thickness = 1
        # render the in lines
        for i,l in enumerate(self.track.data.in_lines):
            # map the line coordinates to the image
            p1 = (l.p1 / self.img_scale) + self.originOffset
            p2 = (l.p2 / self.img_scale) + self.originOffset
            cv.line(self.show_img, tuple(p1.astype(np.int32)), tuple(p2.astype(np.int32)), self.baseColour)
            for ii in range(1,20):
                pos = (l.p1 + l.v * ii / 20) / self.img_scale + self.originOffset
                cv.putText(self.show_img, str(i + 1), tuple(pos.astype(np.int32)), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness)
        # render the out lines
        for i,l in enumerate(self.track.data.out_lines):
            # map the line coordinates to the image
            p1 = (l.p1 / self.img_scale) + self.originOffset
            p2 = (l.p2 / self.img_scale) + self.originOffset
            cv.line(self.show_img, tuple(p1.astype(np.int32)), tuple(p2.astype(np.int32)), self.baseColour)
            for ii in range(1,20):
                pos = (l.p1 + l.v * ii / 20) / self.img_scale + self.originOffset
                cv.putText(self.show_img, str(i + 1), tuple(pos.astype(np.int32)), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness)

    def render_image(self):
        cv.imshow('Vis', self.show_img)
        cv.waitKey(1)
