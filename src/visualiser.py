import cv2 as cv
import numpy as np

class Vis(object):

    def __init__(self):
        self.img_w = 1000
        self.img_h = 1000
        self.img_x_buffer = 100
        self.img_y_buffer = 100
        self.img_scale = 0.1
        self.rCOGLongR = 0.6
        self.xVehicleWidth = 1.915 / self.img_scale
        self.xVehicleLength = 4.529 / self.img_scale
        self.originOffset = np.array([500.0,500.0])
        self.carPos = np.array([0,0])+self.originOffset
        self.carFLOffset = np.array([self.xVehicleLength * self.rCOGLongR, -0.5 * self.xVehicleWidth])
        self.carFROffset = np.array([self.xVehicleLength * self.rCOGLongR, 0.5 * self.xVehicleWidth])
        self.carRLOffset = np.array([-1 * self.xVehicleLength * (1 - self.rCOGLongR), -0.5 * self.xVehicleWidth])
        self.carRROffset = np.array([-1 * self.xVehicleLength * (1 - self.rCOGLongR), 0.5 * self.xVehicleWidth])
        self.orig_img = np.zeros((self.img_w,self.img_h,3), np.uint8)

    def draw_car(self, pos, heading):
        
        # create a copy of the base image
        self.show_img = np.copy(self.orig_img)

        # translate the visual based on the new position
        posConverted = np.array([pos[0] / self.img_scale,pos[1] / self.img_scale])+self.originOffset
        deltaPos = posConverted - self.carPos
        self.carPos += deltaPos
        carFL = self.carPos + self.carFLOffset
        carFR = self.carPos + self.carFROffset
        carRL = self.carPos + self.carRLOffset
        carRR = self.carPos + self.carRROffset
         
        # apply the neccessary rotations to the image
        FLrot = self.rotate_point(self.carPos[0], self.carPos[1], heading, np.copy(carFL))
        FRrot = self.rotate_point(self.carPos[0], self.carPos[1], heading, np.copy(carFR))
        RLrot = self.rotate_point(self.carPos[0], self.carPos[1], heading, np.copy(carRL))
        RRrot = self.rotate_point(self.carPos[0], self.carPos[1], heading, np.copy(carRR))

        # draw the car
        rect = cv.minAreaRect(np.array((FLrot, FRrot, RRrot, RLrot), dtype=np.float32))
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(self.show_img, [box], 0, (0,255,179), 2)
        cv.circle(self.show_img, (int(self.carPos[0]), int(self.carPos[1])), 2, (0,255,179), -1)

        # check the proximity of the car to the screen edges, update the translation
        # x
        cPosOrig = np.copy(self.carPos)
        if self.carPos[0] < self.img_x_buffer:
            # about to disappear to the left of the screen, shift if to the right
            self.originOffset[0] += self.img_x_buffer-self.carPos[0]
            self.carPos[0] += self.img_x_buffer-self.carPos[0]
        elif (self.img_w - self.carPos[0]) < self.img_x_buffer:
            # about to disappear to the right of the screen, shift it to the left
            self.originOffset[0] -= self.img_x_buffer - (self.img_w - self.carPos[0])
            self.carPos[0] -= self.img_x_buffer - (self.img_w - self.carPos[0])
        # y
        if self.carPos[1] < self.img_y_buffer:
            # about to disappear to the top of the screen, shift it to the bottom
            self.originOffset[1] += self.img_y_buffer-self.carPos[1]
            self.carPos[1] += self.img_y_buffer-self.carPos[1]
        elif (self.img_h - self.carPos[1]) < self.img_y_buffer:
            # about to disappear to the bottom of the screen, shift it to the top
            self.originOffset[1] -= self.img_y_buffer - (self.img_h - self.carPos[1])
            self.carPos[1] -= self.img_y_buffer - (self.img_h - self.carPos[1])
        #print('Orig: ',cPosOrig,'New: ',self.carPos)

    def draw_data(self,t):
        """
            Draw the telemetry in t
        """
        color = (0,255,179)
        thickness = 2
        x_start = self.img_x_buffer / 10
        y_offset = 75
        y_start = self.img_y_buffer
        i = 0
        for k,v in t.items():
            pos = (int(x_start), int(y_start + y_offset * i))
            i += 1
            s = k + ':{:.5f}'.format(v)
            cv.putText(self.show_img, s, pos, cv.FONT_HERSHEY_SIMPLEX, 0.75, color, thickness)

    def render_image(self):
        cv.imshow('Vis',self.show_img)
        cv.waitKey(1)

    def rotate_point(self, cx: float, cy: float, a: float, p):
        """
            Rotate the point p about the center (cx,cy) by angle a (rad)
        """
        c = np.cos(a)
        s = np.sin(a)
        
        # translate the p back to the center
        p[0] -= cx
        p[1] -= cy

        # rotate the point
        x_new = p[0] * c - p[1] * s
        y_new = p[0] * s + p[1] * c

        # translate the new point back
        p[0] = x_new + cx
        p[1] = y_new + cy

        return p