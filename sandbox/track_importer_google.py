import cv2 as cv
import argparse
import numpy as np

# import a track from a google terrain image using open cv. Need to spit out the track csv files
refPt = []

def select_points(event, x, y, flags, param):
    global refPt
    if event == cv.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        print('Points selected', refPt)

def main():
    global refPt
    # set the requirements of the arguement parser
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required = True, help = "Name of image (inc ext) within data/track/images")
    ap.add_argument('-s', '--scale', required = False, help = 'Scale of the image in m/pix')
    args = vars(ap.parse_args())

    # grab the location of the script
    import os
    mod_path = os.path.dirname(os.path.abspath(__file__))
    im_path = mod_path + '/../data/track/images/' + args['image']
    print(im_path)
    # load the image to cv
    image = cv.imread(im_path)

    if args['scale'] is None:
        # get the scale - ask the user to select two points and a distance in metres
        print('Select two points on the image to use for scaling')
        print("Press 'y' when happy, press 'r' to reset the points")
        # set the callback
        cv.namedWindow('scale_image')
        cv.setMouseCallback('scale_image', select_points)
        # get the user to select points
        while True:
            cv.imshow('scale_image', image)
            key = cv.waitKey(1) & 0xFF

            # reset the points
            if key == ord('r'):
                print('Resetting points')
                refPt = []

            if key == ord('y'):
                if len(refPt) > 2:
                    print("Too many points selected, press 'r' to reset")
                if len(refPt) < 2:
                    print('Need two points, {0} selected'.format(len(refPt)))
                else:
                    print('Points {0} selected'.format(refPt))
                    break
        # get the real distance between the two points from the user
        scale_dist = float(input('Enter the distance in metres between the two points: '))
        # calculate the image scaling in m/pixel
        scale = scale_dist / np.sqrt((refPt[1][0] - refPt[0][0])**2 + (refPt[1][1] - refPt[0][1])**2)
    else:
        scale = float(args['scale'])

    print('Scale set at {:.3f} m/pix'.format(scale))

    # covert the image to greyscale
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # convert to alpha image
    for i in range(0,175):
        (thresh, alpha_img) = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY)

        cv.imshow('Alpha',alpha_img)
        cv.waitKey(100)

if __name__ == "__main__":
    main()
