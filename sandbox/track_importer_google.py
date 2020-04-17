import cv2 as cv
import numpy as np
import argparse
from msvcrt import getch
from src.geom import calc_euclid_distance_2d_sq

# import a track from a google terrain image using open cv. Need to spit out the track csv files

# need tons of globals for the mouse callback functions in cv
refPt = []
final_cnts = []
hover_cnts = []
temp_cnts = []
cent_cnts = []
points_raw = []
hover_points = []
final_points = []
last_idx_selected = None
len_points_added = None
start_idx = None


# ######################
# CV 2 MOUSE CALLBACKS #
# ######################
def select_scale_points(event, x, y, flags, param):
    """
        Select points to provide real world scaling
    """
    global refPt
    if event == cv.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        print('Points selected', refPt)


def select_contour(event, x, y, flags, param):
    """
        Select the contours to be used for the track generation
    """
    global temp_cnts, hover_cnts, cnt_thresh, final_cnts, cent_cnts
    d = 1e9
    idx = None
    for i,c in enumerate(cent_cnts):
        d_temp = calc_euclid_distance_2d_sq((x, y), c)
        if d_temp < d:
            d = d_temp
            idx = i

    # highlight the closest contour
    hover_cnts = temp_cnts[idx]

    # select the closest contour
    if event == cv.EVENT_LBUTTONDOWN:
        final_cnts.append(temp_cnts[idx])

def select_cnt_points(event, x, y, flags, param):
    """
        Select the points on the selected contours to be used for track generation
    """
    global points_raw, hover_points, final_points, last_idx_selected, len_points_added

    d = 1e9
    idx = None
    for i,p in enumerate(points_raw):
        d_temp = calc_euclid_distance_2d_sq((x, y), tuple(p))
        if d_temp < d:
            d = d_temp
            idx = i

    # select the closest point
    if event == cv.EVENT_LBUTTONDOWN and len(hover_points) > 0:
        if len(final_points) == 0:
            final_points = hover_points.copy()
        else:
            final_points = np.vstack((final_points,hover_points))
        last_idx_selected = idx
        len_points_added = len(hover_points) # used incase of an undo
    # shift will allow multiple points to be selected
    if flags == cv.EVENT_FLAG_SHIFTKEY and len(final_points) > 0:
        # multi point selection using the last point in final points as the
        # start of the bulk selection.
        if idx > last_idx_selected:
            # highlight the points
            hover_points = points_raw[last_idx_selected:idx+1,:]
        elif idx < last_idx_selected:
            hover_points = np.flip(points_raw[idx+1:last_idx_selected,:],0)
    else:
        # highlight the closest point
        hover_points = points_raw[idx,:]

def select_start_line(event, x, y, flags, param):
    """
        Select the point to be used as the start line
    """
    global final_points, start_idx, hover_points

    d = 1e9
    idx = None
    for i,p in enumerate(final_points):
        d_temp = calc_euclid_distance_2d_sq((x, y), tuple(p))
        if d_temp < d:
            d = d_temp
            idx = i
    # select the closest point
    if event == cv.EVENT_LBUTTONDOWN and len(hover_points) > 0:
        start_idx = idx
    # highlight the closest point
    hover_points = final_points[idx,:]

# ###################
# UTILITY FUNCTIONS #
# ###################
def smooth_points(x, y, s, NPoints):
    """
        Smooth the provided points with the splprep
    """
    from scipy.interpolate import splev, splprep
    tck, u = splprep([x, y], s=s)
    u_ev = np.linspace(0, 1, NPoints)
    new_points = splev(u_ev, tck)
    x_new = np.array(new_points[0])
    y_new = np.array(new_points[1])
    return np.vstack((x_new,y_new)).T

def produce_edge_image(thresh, img):
    """
        Threshold the image and return the edges
    """
    (thresh, alpha_img) = cv.threshold(img, thresh, 255, cv.THRESH_BINARY_INV)
    blur_img = cv.medianBlur(alpha_img, 9)
    blur_img = cv.morphologyEx(blur_img, cv.MORPH_OPEN, (5,5))
    # find the edged
    return cv.Canny(blur_img, 30, 200)

# ######
# MAIN #
# ######
def main():
    global refPt, temp_cnts, final_cnts, hover_cnts, cent_cnts
    # set the requirements of the arguement parser
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required = True, help = "Name of image (inc ext) within data/track/images")
    ap.add_argument('-s', '--scale', required = False, help = 'Scale of the image in m/pix')
    ap.add_argument('-ss', '--splinesmoothing', required = False, help = 'Spline smoothing value for splprep')
    ap.add_argument('-t', '--track', required = True, help = 'Name of the track')
    ap.add_argument('-th', '--threshold', required = False, help = 'Theshold value')
    args = vars(ap.parse_args())

    # grab the location of the script
    import os
    mod_path = os.path.dirname(os.path.abspath(__file__))
    im_path = mod_path + '/../data/track/images/' + args['image'] + '.png'
    print(im_path)
    # load the image to cv
    image = cv.imread(im_path)

    if args['scale'] is None:
        # get the scale - ask the user to select two points and a distance in metres
        print('Select two points on the image to use for scaling')
        print("Press 'y' when happy, press 'r' to reset the points")
        # set the callback
        cv.namedWindow('scale_image')
        cv.setMouseCallback('scale_image', select_scale_points)
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
        # close the window
        cv.destroyWindow('scale_image')
        # get the real distance between the two points from the user
        scale_dist = float(input('Enter the distance in metres between the two points: '))
        # calculate the image scaling in m/pixel
        scale = scale_dist / np.sqrt((refPt[1][0] - refPt[0][0])**2 + (refPt[1][1] - refPt[0][1])**2)
    else:
        scale = float(args['scale'])

    print('Scale set at {:.3f} m/pix'.format(scale))

    # covert the image to greyscale
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # scale it up

    # convert to alpha image
    if args['threshold'] is None:
        img_copy = image.copy()
        calc_thresh = True
        thresh = 90
        while True:
            cv.imshow('Thresh', img_copy)
            key = cv.waitKey(1) & 0xFF

            if key == ord('w'):
                thresh += 2
                thresh = min(250, thresh)
                calc_thresh = True

            if key == ord('s'):
                thresh -= 2
                thresh = max(2, thresh)
                calc_thresh = True

            if key == ord('y'):
                break

            if calc_thresh:
                img_copy = image.copy()
                edged = produce_edge_image(thresh, gray_img.copy())
                contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                cv.drawContours(img_copy, contours, -1, (0, 0, 255), 1)
                cv.putText(img_copy, 'Threshold @ '+str(thresh), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 179), 2)
                cv.putText(img_copy, 'W to increase, S to descrease, Y to continue', (50, 100), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 179), 1)
                calc_thresh = False

    else:
        edged = produce_edge_image(int(args['threshold']), gray_img)
        contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.destroyWindow('Thresh')

    for c in contours:
        if cv.contourArea(c) > 100:
            temp_cnts.append(c)
            # calc the centroid
            M = cv.moments(c)
            cent_cnts.append((M['m10'] / M['m00'], M['m01'] / M['m00']))

    # get the user to choose the contours
    cv.namedWindow('Contours')
    cv.setMouseCallback('Contours', select_contour)
    print("Select the contours to keep, when ready press 'y', 'z' will undo the last change")
    for i in range(0,2):
        img_copy = image.copy()
        if i == 0:
            print('## INNER TRACK ## First choose the contours that form the inner track edge')
        else:
            print('## OUTER TRACK ## Now select the contours that form the outer track edge')
        while True:
            key = cv.waitKey(1) & 0xFF
            cv.imshow('Contours', img_copy)
            if key == ord('y'):
                print('Selection complete')
                if i == 0:
                    in_track_cnts = final_cnts
                    final_cnts = []
                elif i == 1:
                    out_track_cnts = final_cnts
                break

            if key == ord('z'):
                if len(final_cnts) == 1:
                    final_cnts = []
                elif len(final_cnts) > 1:
                    final_cnts = final_cnts[:-1]

            img_copy = image.copy()
            cv.drawContours(img_copy, temp_cnts, -1, (0, 0, 255), 1)
            if len(hover_cnts) > 0:
                cv.drawContours(img_copy, hover_cnts, -1, (255, 0, 0), 2)
            if len(final_cnts) > 0:
                cv.drawContours(img_copy, final_cnts, -1, (0 ,255, 0), 1)
            if i == 0:
                cv.putText(img_copy, 'Select contours to use for INNER track', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 179), 2)
            else:
                cv.putText(img_copy, 'Select contours to use for OUTER track', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 179), 2)
            cv.putText(img_copy, 'Left-Click to select, Y to continue, Z to undo (uses contour centroid to locate closest)', (50, 100), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 179), 1)
    cv.destroyWindow('Contours')

    # get the user to select the points
    global points_raw, hover_points, final_points, len_points_added, last_idx_selected
    for i in range(0,2):
        if i == 0:
            temp_cnts = in_track_cnts
        else:
            temp_cnts = out_track_cnts

        # stack up the contour points into a single 2D array
        for ii,c in enumerate(temp_cnts):
            if ii == 0:
                points_raw = np.array(c[:,0,:])
            else:
                points_raw = np.vstack((points_raw,c[:,0,:]))

        # allow the user to select the points
        cv.namedWindow('Points')
        cv.setMouseCallback('Points', select_cnt_points)
        img_copy = image.copy()
        while True:
            key = cv.waitKey(1) & 0xFF
            cv.imshow('Points', img_copy)
            if key == ord('y'):
                print('Selection complete')
                if i == 0:
                    in_track_points = final_points.astype(dtype=np.float64)
                    final_points = []
                elif i == 1:
                    out_track_points = final_points.astype(dtype=np.float64)
                break

            if key == ord('z'):
                if len_points_added is not None and len(final_points) > 0:
                    if len_points_added == len(final_points):
                        final_points = []
                        last_idx_selected = None
                    else:
                        final_points = final_points[:-1 * len_points_added]
                        last_idx_selected -= len_points_added

            # plot the latest data
            img_copy = image.copy()
            for p in points_raw:
                cv.circle(img_copy, tuple(p), 1, (0, 0, 255))
            if len(hover_points) > 0:
                if len(hover_points.shape) == 1:
                    cv.circle(img_copy, tuple(hover_points), 2, (255, 0, 0),-1)
                elif hover_points.shape[0] > 1:
                    for p in hover_points:
                        cv.circle(img_copy, tuple(p), 2, (255, 0, 0),-1)
            if len(final_points) > 0:
                if len(final_points.shape) == 1:
                    cv.circle(img_copy, tuple(final_points), 2, (255, 0, 0),-1)
                elif final_points.shape[0] > 1:
                    for p in final_points:
                        cv.circle(img_copy, tuple(p), 2, (0, 255, 0),-1)
            if i == 0:
                cv.putText(img_copy, 'Select points to use for INNER track', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 179), 2)
            else:
                cv.putText(img_copy, 'Select points to use for OUTER track', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 179), 2)
            cv.putText(img_copy, 'Left-Click to select, hold shift to multi-select, Y to continue, Z to undo', (50, 100), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 179), 1)
            #print('H',len(hover_points),'F',final_points)
    cv.destroyWindow('Points')

    # fit a splines to the data sets in order to smooth

    for i in range(0,2):
        # choose the point set
        if i == 0:
            points = in_track_points.copy()
        else:
            points = out_track_points.copy()

        # need to remove duplicates for splprep to function
        for ii,p in enumerate(points):
            if ii == 0:
                points_proc = p
            else:
                if len(points_proc.shape) == 1:
                    if p[0] != points_proc[0] and p[1] != points_proc[1]:
                        points_proc = np.vstack((points_proc, p))
                elif p[0] != points_proc[-1,0] and p[1] != points_proc[-1,1]:
                    points_proc = np.vstack((points_proc, p))
        x = points_proc[:,0]
        y = points_proc[:,1]
        if args['splinesmoothing'] is None:
            s = 100
            N = 2000
            calc_spline = True
            while True:
                if calc_spline:
                    smoothed_points = smooth_points(x, y, s, N)
                    plot_points = smoothed_points.astype(np.int32).T
                    plot_points = np.array([list(zip(plot_points[0], plot_points[1]))])
                    calc_spline = False
                img_copy = image.copy()
                if i == 0:
                    cv.drawContours(img_copy, in_track_cnts, -1, (0 ,0, 255), 1)
                else:
                    cv.drawContours(img_copy, out_track_cnts, -1, (0 ,0, 255), 1)
                cv.drawContours(img_copy, plot_points, -1, (0 ,255, 0), 1)
                for p in smoothed_points.astype(np.int32):
                    cv.circle(img_copy, tuple(p), 2, (0, 255, 0),-1)
                cv.putText(img_copy, 'BSpline Smoothing @ '+str(s) + ' with ' + str(N) + ' points', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 179), 2)
                cv.putText(img_copy, 'W to incr smoothing, S to dcr smoothing,  D to incr points, A to drc points, Y to continue', (50, 100), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 179), 1)
                cv.imshow('Smoothed',img_copy)
                key = cv.waitKey(100) & 0xFF

                if key == ord('y'):
                    if i == 0:
                        print('Inner Smoothing Completed')
                        in_points_final = smoothed_points
                    else:
                        print('Outer Smoothing Completed')
                        out_points_final = smoothed_points
                    break
                if key == ord('s'):
                    s -= 200
                    s = max(100, s)
                    calc_spline = True
                if key == ord('w'):
                    s += 200
                    calc_spline = True
                if key == ord('a'):
                    N -= 200
                    N = max(200, N)
                    calc_spline = True
                if key == ord('d'):
                    N += 200
                    calc_spline = True
        else:
            spline_smoothing = float(args['splinesmoothing'])
            smoothed_points = smooth_points(x, y, spline_smoothing, 2000)
            if i == 0:
                print('Inner Smoothing Completed')
                in_points_final = smoothed_points
            else:
                print('Outer Smoothing Completed')
                out_points_final = smoothed_points
    cv.destroyWindow('Smoothed')

    # set the start line
    global start_idx
    for i in range(0,2):
        hover_points = []
        start_idx = None
        if i == 0:
            final_points = in_points_final.copy()
        else:
            final_points = out_points_final.copy()
        cv.namedWindow('StartLine')
        cv.setMouseCallback('StartLine', select_start_line)
        img_copy = image.copy()
        plot_points = in_points_final.astype(np.int32).T
        in_cnt = np.array([list(zip(plot_points[0], plot_points[1]))])
        plot_points = out_points_final.astype(np.int32).T
        out_cnt = np.array([list(zip(plot_points[0], plot_points[1]))])
        while True:
            key = cv.waitKey(1) & 0xFF
            cv.imshow('StartLine', img_copy)

            if key == ord('y') and start_idx is not None:
                if i == 0:
                    if start_idx > 0:
                        print(in_points_final)
                        in_points_final = np.vstack((in_points_final[start_idx:,:], in_points_final[:start_idx,:]))
                        print(in_points_final)
                        in_start = (int(in_points_final[0,0]), int(in_points_final[0,1]))
                    print('Inner start line set')
                else:
                    if start_idx > 0:
                        out_points_final = np.vstack((out_points_final[start_idx:,:], out_points_final[:start_idx,:]))
                    print('Outer start line set')
                break

            img_copy = image.copy()
            if i == 0:
                cv.drawContours(img_copy, in_cnt, -1, (0 , 255, 0), 1)
                cv.drawContours(img_copy, out_cnt, -1, (0 , 0, 255), 1)
                for p in in_points_final.astype(np.int32):
                    cv.circle(img_copy, tuple(p), 2, (0, 255, 0),-1)
                if start_idx is not None:
                    p = (int(in_points_final[start_idx,0]), int(in_points_final[start_idx,1]))
                    cv.circle(img_copy, p, 2, (0, 0, 255),-1)
            else:
                cv.drawContours(img_copy, in_cnt, -1, (0 , 0, 255), 1)
                cv.drawContours(img_copy, out_cnt, -1, (0 , 255, 0), 1)
                for p in out_points_final.astype(np.int32):
                    cv.circle(img_copy, tuple(p), 2, (0, 255, 0),-1)
                cv.circle(img_copy, in_start, 2, (0, 255, 0), 1)
                if start_idx is not None:
                    p = (int(out_points_final[start_idx,0]), int(out_points_final[start_idx,1]))
                    cv.circle(img_copy, p, 2, (0, 0, 255),-1)
            if len(hover_points) > 0:
                cv.circle(img_copy, tuple(hover_points.astype(np.int32)), 2, (255, 0, 0),-1)
            if i == 0:
                cv.putText(img_copy, 'set INNER start line', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 179), 2)
            else:
                cv.putText(img_copy, 'Set OUTER start line', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 179), 2)
            cv.putText(img_copy, 'Left-Click to select, Y to continue', (50, 100), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 179), 1)

    cv.destroyWindow('StartLine')

    # convert the points to real world values
    in_points_final = in_points_final * scale
    out_points_final = out_points_final * scale

    # save as csv files
    import csv
    import os
    module_path = os.path.dirname(os.path.abspath(__file__))
    with open(module_path + '/../data/track/' + args['track'] + '_IN.csv', 'w', newline = '') as f:
        writer = csv.writer(f, delimiter = ',')
        for i in range(0,len(in_points_final)):
            writer.writerow([in_points_final[i,0], in_points_final[i,1]])

    with open(module_path + '/../data/track/' + args['track'] + '_OUT.csv', 'w', newline = '') as f:
        writer = csv.writer(f, delimiter = ',')
        for i in range(0,len(out_points_final)):
            writer.writerow([out_points_final[i,0], out_points_final[i,1]])

    # call the track module to convert into a playable .track asset
    from src.track import Track
    t = Track()
    t.load_from_csv(args['track'])
    t.pickle_track()


if __name__ == "__main__":
    main()
