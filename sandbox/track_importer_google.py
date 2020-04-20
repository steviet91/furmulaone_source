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
pix_selected = None
calc_gray = False

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

def select_target_pixel(event, x, y, flags, param):
    """
        Select a pixel to increase or reduce its intensity
    """
    global pix_selected, calc_gray

    if event == cv.EVENT_LBUTTONDOWN:
        pix_selected = (x,y)
        calc_gray = True
        print('User has selected pixel:', pix_selected)

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
    return cv.Canny(blur_img, 30, 200), alpha_img

def set_text_help(img, text, is_gray=False, pos=0):
    """
        Set the set onto the image with a background
    """
    text_x = 20
    text_y = 30 + 30 * pos
    font = cv.FONT_HERSHEY_SIMPLEX
    font_size = 0.5
    if is_gray:
        font_colour = (255, 255, 255)
    else:
        font_colour = (0, 255, 179)
    background = (0, 0, 0)
    font_thickness = 1
    (text_w, text_h),_ = cv.getTextSize(text, font, fontScale=font_size, thickness=font_thickness)
    box_coords = ((text_x, text_y+5), (text_x + text_w + 10 , text_y - text_h - 5))
    cv.rectangle(img, box_coords[0], box_coords[1], background, cv.FILLED)
    cv.putText(img, text, (text_x, text_y), font, fontScale=font_size, color=font_colour, thickness=font_thickness)


# ######
# MAIN #
# ######
def main():
    global refPt, temp_cnts, final_cnts, hover_cnts, cent_cnts
    # set the requirements of the arguement parser
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required = True, help = "Name of .png image file (excluding ext) within data/track/images")
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
        img_copy = image.copy()
        set_text_help(img_copy, 'Define the image scale by selecting two points of known distance, press "Y" when chosen, "R" to reset points')
        while True:
            cv.imshow('scale_image', img_copy)
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
        # create a new master image for thresholding - we may want to change base properties to re grayscale
        img_copy = image.copy()
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV) # grab the hsv version of the image
        masked_img = None
        calc_thresh = True
        NImage = 0
        thresh = 90
        h_tol_up = 1 # set a tolerance for hue
        s_tol_up = 1 # set a tolerance for saturation
        v_tol_up = 1 # set a tolerance for value
        h_tol_down = -1 # set a tolerance for hue
        s_tol_down = -1 # set a tolerance for saturation
        v_tol_down = -1 # set a tolerance for value
        cv.namedWindow('Thresh')
        cv.setMouseCallback('Thresh', select_target_pixel)
        help_text = '..."W" to increase, "S" to descrease, "Y" to continue, "T" to switch between alpha image, click a pixel to modify its intensity, "D" to increase, "A" to reduce'
        help_mask1 = '(d/f to change h_tol_up, g/h to change s_tol_up, j/k to change v_tol_up)'
        help_mask2 = '(x/c to change h_tol_down, v/b to change s_tol_down, n/m to change v_tol_down)'
        global pix_selected, calc_gray
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

            # HVS Upper tolerances
            if pix_selected is not None:
                if key == ord('f'):
                    h_tol_up += 1
                    calc_gray = True

                if key == ord('d'):
                    h_tol_up -= 1
                    h_tol_up = max(1, h_tol_up)
                    calc_gray = True

                if key == ord('h'):
                    s_tol_up += 1
                    calc_gray = True

                if key == ord('g'):
                    s_tol_up -= 1
                    s_tol_up = max(1, s_tol_up)
                    calc_gray = True

                if key == ord('k'):
                    v_tol_up += 1
                    calc_gray = True

                if key == ord('j'):
                    v_tol_up -= 1
                    v_tol_up = max(1, v_tol_up)
                    calc_gray = True

                # HSV Lower tolerances
                if key == ord('x'):
                    h_tol_down -= 1
                    calc_gray = True

                if key == ord('c'):
                    h_tol_down += 1
                    h_tol_down = min(-1, h_tol_down)
                    calc_gray = True

                if key == ord('v'):
                    s_tol_down -= 1
                    calc_gray = True

                if key == ord('b'):
                    s_tol_down += 1
                    s_tol_down = min(-1, s_tol_down)
                    calc_gray = True

                if key == ord('n'):
                    v_tol_down -= 1
                    calc_gray = True

                if key == ord('m'):
                    v_tol_down += 1
                    v_tol_down = min(-1, v_tol_down)
                    calc_gray = True

            # Change image
            if key == ord('t'):
                if NImage == 2:
                    NImage = 0
                else:
                    NImage += 1
                if NImage == 2 and  masked_img is None:
                    NImage = 0
                calc_thresh = True

            if key == ord('y'):
                break

            if calc_gray:
                pix_channel = hsv[pix_selected[1], pix_selected[0]]
                low_b = pix_channel + np.array([h_tol_down, s_tol_down, v_tol_down])
                up_b = pix_channel + np.array([h_tol_up, s_tol_up, v_tol_up])
                mask = cv.inRange(hsv, low_b, up_b) # take a mask of the image
                masked_img = cv.bitwise_and(image, image, mask=mask)
                gray_img = cv.cvtColor(masked_img, cv.COLOR_BGR2GRAY) # scale it up
                calc_gray = False
                calc_thresh = True

            if calc_thresh:
                edged,alpha_img = produce_edge_image(thresh, gray_img.copy())
                contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                if NImage == 0:
                    img_copy = image.copy()
                elif NImage == 1:
                    img_copy = alpha_img.copy()
                elif NImage == 2:
                    img_copy = masked_img.copy()
                cv.drawContours(img_copy, contours, -1, (0, 0, 255), 1)
                if NImage == 0:
                    set_text_help(img_copy, ' RAW - Threshold @ '+str(thresh)+ help_text)
                elif NImage == 1:
                    set_text_help(img_copy, 'ALPHA - Threshold @ '+str(thresh)+ help_text, is_gray=True)
                elif NImage == 2:
                    set_text_help(img_copy, 'MASK - Threshold @ '+str(thresh)+ help_text)
                if masked_img is not None:
                    mask_text = 'HSV Tolerances for Mask- H('+str(h_tol_down)+'/+'+str(h_tol_up) +') S('+str(s_tol_down)+'/+'+str(s_tol_up) +') V('+str(v_tol_down)+'/+'+str(v_tol_up) +')  '
                    if NImage == 1:
                        set_text_help(img_copy, mask_text, pos=1, is_gray=True)
                        set_text_help(img_copy, help_mask1, pos=2, is_gray=True)
                        set_text_help(img_copy, help_mask2,pos=3, is_gray=True)
                    else:
                        set_text_help(img_copy, mask_text, pos=1)
                        set_text_help(img_copy, help_mask1, pos=2)
                        set_text_help(img_copy, help_mask2,pos=3)
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
    help_text = 'Left-Click to select, Y to continue, Z to undo (uses contour centroid to locate closest)'
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
            for c in cent_cnts:
                cv.circle(img_copy, (int(c[0]), int(c[1])), 1, (0, 255, 179), -1)
                cv.putText(img_copy, 'c',(int(c[0]), int(c[1])), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 179), thickness=1)
            if len(hover_cnts) > 0:
                cv.drawContours(img_copy, hover_cnts, -1, (255, 0, 0), 2)
            if len(final_cnts) > 0:
                cv.drawContours(img_copy, final_cnts, -1, (0 ,255, 0), 1)
            if i == 0:
                set_text_help(img_copy, 'Select contours to use for INNER track' + help_text)
            else:
                set_text_help(img_copy, 'Select contours to use for OUTER track' + help_text)
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
        help_text = 'Left-Click to select, hold shift to multi-select, Y to continue, Z to undo'
        img_copy = image.copy()
        last_idx_selected = None
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
                set_text_help(img_copy, 'Select points to use for INNER track' + help_text)
            else:
                set_text_help(img_copy, 'Select points to use for OUTER track' + help_text)
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
        help_text = 'W to incr smoothing, S to dcr smoothing,  D to incr points, A to drc points, Y to continue'
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
                    cv.drawContours(img_copy, in_points_plot, -1, (0, 255, 0), 1)
                cv.drawContours(img_copy, plot_points, -1, (0 ,255, 0), 1)
                for p in smoothed_points.astype(np.int32):
                    cv.circle(img_copy, tuple(p), 2, (0, 255, 0),-1)
                set_text_help(img_copy, 'BSpline Smoothing @ '+str(s) + ' with ' + str(N) + ' points...' + help_text)
                cv.imshow('Smoothed',img_copy)
                key = cv.waitKey(100) & 0xFF

                if key == ord('y'):
                    if i == 0:
                        print('Inner Smoothing Completed')
                        in_points_final = smoothed_points
                        in_points_plot = smoothed_points.astype(np.int32).T
                        in_points_plot = np.array([list(zip(in_points_plot[0], in_points_plot[1]))])
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
        help_text = 'Left-Click to select, Y to continue'
        while True:
            key = cv.waitKey(1) & 0xFF
            cv.imshow('StartLine', img_copy)

            if key == ord('y') and start_idx is not None:
                if i == 0:
                    if start_idx > 0:
                        in_points_final = np.vstack((in_points_final[start_idx:,:], in_points_final[:start_idx,:]))
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
                set_text_help(img_copy, 'Set INNER start line, ' + help_text)
                cv.putText(img_copy, 'set INNER start line', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 179), 2)
            else:
                set_text_help(img_copy, 'Set OUTER start line, ' + help_text)

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
