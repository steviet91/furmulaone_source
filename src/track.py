import numpy as np
from .geom import Line
from .geom import rotate_point
from .geom import Circle
from .geom import calc_euclid_distance_2d_sq
from .geom import check_for_intersection_lineseg_circle
from .geom import calc_angle_between_unit_vectors
import time
import os


class TrackHandler(object):
    def __init__(self, track_name: str):
        # get the module path
        self.module_path = os.path.dirname(os.path.abspath(__file__))

        # save the arguments
        self.track_name = track_name

        # load the track data
        self.data = Track.loader(track_name)

        self.NLapIdxMax = len(self.data.cent_lines)-1
        self.NProgessSearchPointsBack = min(2, self.NLapIdxMax) # Pos progress search points
        self.NProgessSearchPointsForward = min(3, self.NLapIdxMax) # Pos progress search points

    # ##################
    # VEHICLE PROGRESS #
    # ##################
    def check_new_lap(self, xC: float, yC: float, bCarNearStartLine: bool):
        """
            Check for a new lap by looking for a rising edge on the distance of the car to
            the start line vertices. This logic should also work for the first lap.
        """
        # calculate the distance between the car and each vertex
        d1 = np.sqrt((self.data.startLine.x1 - xC)**2 + (self.data.startLine.y1 - yC)**2)
        d2 = np.sqrt((self.data.startLine.x2 - xC)**2 + (self.data.startLine.y2 - yC)**2)

        # check the distance compared to the length of the start line + a tolerance
        if (d1 + d2) < (self.data.startLine.v_mag + self.data.startLine.v_mag * 2):
            # we're new the start line
            bCarNearStartLine = True
            bNewLap = False
        else:
            if bCarNearStartLine:
                # this is a falling edge, and therefore a new lap has begun
                bNewLap = True
                bCarNearStartLine = False
            else:
                bNewLap = False

        return bNewLap, bCarNearStartLine

    def get_veh_pos_progress(self, NLapIdx: int, xC: float, yC: float):
        """
            Feed back the closest inner track point and the percentage of the lap
            completed based on total number of points
        """
        search_start_idx = NLapIdx - self.NProgessSearchPointsBack
        search_end_idx = NLapIdx + self.NProgessSearchPointsForward

        # check for the lap start/end and get the list of indexes to check
        if search_start_idx < 0:
            idx_s = np.arange(self.NLapIdxMax + search_start_idx + 1, self.NLapIdxMax + 1)
            if NLapIdx > 0:
                idx_s = np.append(idx_s, np.arange(0, NLapIdx))
        else:
            idx_s = np.arange(search_start_idx, NLapIdx)
        if search_end_idx > self.NLapIdxMax:
            if NLapIdx == self.NLapIdxMax:
                idx_e = np.array([NLapIdx])
            else:
                idx_e = np.arange(NLapIdx, self.NLapIdxMax + 1)
            idx_e = np.append(idx_e, np.arange(0, search_end_idx - self.NLapIdxMax + 1))
        else:
            idx_e = np.arange(NLapIdx, search_end_idx + 1)
        idxs = np.append(idx_s, idx_e)

        # find the closest point to the current vehicle position
        d = 1e12
        idx = NLapIdx
        pC = (xC, yC)
        a1 = None
        for i in idxs:
            d_temp = calc_euclid_distance_2d_sq(pC, self.data.cent_lines[i].p1)
            if d_temp < d :
                d = d_temp
                idx = i
        return idx/self.NLapIdxMax, idx

    # ###########
    # COLLISION #
    # ###########
    def get_line_idxs_for_collision(self, c: Circle):
        """
            This return a list of indexes of lines that should be checked for collision
        """
        in_idxs = [i for i,l in enumerate(self.data.in_lines) if self.check_line_for_collision(l, c)]
        out_idxs = [i for i,l in enumerate(self.data.out_lines) if self.check_line_for_collision(l, c)]

        return in_idxs, out_idxs

    def check_line_for_collision(self, l: Line, c: Circle):
        """
            Returns true if either:
                - Has at least one vertex that lies within the circles radius
                - That intersects at least once with the circle
        """
        r2 = c.r**2
        # check the distance of the points to the circle's centre
        if calc_euclid_distance_2d_sq(l.p1, (c.x0, c.y0)) <= r2:
            # p1 lies within the circumference of the circle
            return True
        elif calc_euclid_distance_2d_sq(l.p2, (c.x0, c.y0)) <= r2:
            # p2 lies within the circumference of the circle
            return True
        else:
            # check if the line intersects the circle
            do_intersect, _ = check_for_intersection_lineseg_circle(l, c)
            if do_intersect:
                # l intersects with c
                return True
            else:
                # line shouldn't be check for collision
                return False

class Track(object):
    def __init__(self):
        # get the module path
        self.module_path = os.path.dirname(os.path.abspath(__file__))

        # initialise the variables
        self.track_name = None
        self.in_lines = []
        self.out_lines = []
        self.startLine = None
        self.startPos = None
        self.aTrackRotation0 = None

    @classmethod
    def loader(cls, track_name):
        import pickle
        # load the track
        module_path = os.path.dirname(os.path.abspath(__file__))
        return pickle.load(open(module_path + '/../data/track/' + track_name + '.track', 'rb'))

    def load_from_csv(self, track_name: str):
        """
            Load raw points from the csv files and turn into two sets of line objects

            Assumes the csv file follows rows of points [X, Y] and the in/out datasets
            take a _IN / _OUT suffix
        """
        from .geom import get_intersection_point_lineseg_lineseg
        # save the track name
        self.track_name = track_name

        # load the data from a csv
        in_raw_points = np.genfromtxt(self.module_path + '/../data/track/' + track_name + '_IN.csv', delimiter=',')
        out_raw_points = np.genfromtxt(self.module_path + '/../data/track/' + track_name + '_OUT.csv', delimiter=',')

        # set the starting line using the first two points off the in and out datasets
        # set need to run a translation of all points to ensure the startPos is (0, 0)
        startLineRaw = Line(tuple(in_raw_points[0, :]), tuple(out_raw_points[0, :]))

        # set the starting position
        startPosRaw = startLineRaw.p1 + 0.5 * startLineRaw.v  # this is the translation required

        # calculate the starting orientation to apply to the track
        # this is based on the vector of the first line in the inner set
        rotLine = Line(tuple(in_raw_points[0, :]), tuple(in_raw_points[1, :]))
        aTrackRotation0 = -1* np.arctan2(rotLine.v[1], rotLine.v[0])

        # convert the in points to line segments
        self.in_lines = []
        for i in range(0, in_raw_points.shape[0]):
            if i == in_raw_points.shape[0] - 1:
                # last point, join to the first
                p2 = in_raw_points[0, :] - startPosRaw
            else:
                p2 = in_raw_points[i + 1, :] - startPosRaw
            p1 = in_raw_points[i, :] - startPosRaw
            # rotate the points, they should now be translated to the origin
            p1 = rotate_point(0.0, 0.0, aTrackRotation0, p1)
            p2 = rotate_point(0.0, 0.0, aTrackRotation0, p2)
            self.in_lines.append(Line(tuple(p1), tuple(p2)))

        # convert the out points to lines
        self.out_lines = []
        for i in range(0, out_raw_points.shape[0]):
            if i == out_raw_points.shape[0] - 1:
                # last point, join to the first
                p2 = out_raw_points[0, :] - startPosRaw
            else:
                p2 = out_raw_points[i + 1, :] - startPosRaw
            p1 = out_raw_points[i, :] - startPosRaw
            # rotate the points, they should now be translated to the origin
            p1 = rotate_point(0.0, 0.0, aTrackRotation0, p1)
            p2 = rotate_point(0.0, 0.0, aTrackRotation0, p2)
            self.out_lines.append(Line(tuple(p1), tuple(p2)))

        # set up a centre line with equally spaced points - fire rays from the inner track
        # out to the outer track and take the halfway point of the intersection
        # ray will be fired out such that the angle between it and the current and previous lines
        # are equal
        cent_pts_raw = np.zeros((len(self.in_lines), 2))
        for i,l1 in enumerate(self.in_lines):
            # get the preceeding inner line
            if i == 0:
                l0 = self.in_lines[-1]
            else:
                l0 = self.in_lines[i - 1]

            # calculate the angle between the two lines find the equi - angle (can't remember the correct terminology!!!)
            a0 = calc_angle_between_unit_vectors(l0.v_hat, l1.v_hat) / 2

            # generate a unit vector at this angle
            # [1] cos(a0) = l1.x * x + l1.y * y
            # => x = (cos(a0) - l1.y * y) / l1.x
            # [2] cos(a0) = l0.x * x + l0.y * y
            #
            # [1] -> [2]
            # y = (cos(a0) - l0.x * cos(a0) / l1.x) / (l0.y - l0.x * l1.y / l1.x)
            # the proceeding vector needs to be flipped
            x0 = l0.v_hat[0] * -1
            y0 = l0.v_hat[1] * -1
            x1 = l1.v_hat[0]  # give him some div by zero protection!
            if abs(x1) < 1e-15:
                if x1 >= 0:
                    x1 = 1e-15
                else:
                    x1 = -1e-15
            y1 = l1.v_hat[1]

            y = (np.cos(a0) - x0 * np.cos(a0) / x1) / (y0 - x0 * y1 / x1)
            x = (np.cos(a0) - y1 * y) / x1
            # generate line objects firing in opposite directions
            lt1 = Line(l1.p1, (l1.x1 + x, l1.y1 + y))
            lt2 = Line(l1.p1, (l1.x1 - x, l1.y1 - y))

            # fire the rays at the inner track first - this ensures we don't
            # find a point on a different part of the track (i.e. if the track loops back on itself)
            din_lt1 = 1e12
            din_lt2 = 1e12
            for lin in self.in_lines:
                pInt = get_intersection_point_lineseg_lineseg(lin, lt1, l2_is_ray=True)
                if pInt is not None:
                    d_temp = calc_euclid_distance_2d_sq(lt1.p1, pInt)
                    if d_temp < din_lt1 and d_temp > 1:
                        din_lt1 = d_temp
                # now fire in the opposite direction
                pInt = get_intersection_point_lineseg_lineseg(lin, lt2, l2_is_ray=True)
                if pInt is not None:
                    d_temp = calc_euclid_distance_2d_sq(lt2.p1, pInt)
                    if d_temp < din_lt2 and d_temp > 1:
                        din_lt2 = d_temp
            # now fire the ray out at outer track - note we don't know which side the track it
            # so we fire in both directions. Only take the point if it lies closer than the
            # closes inner intersection for the specific ray (i'm not explaining this very well...)
            # essentialyl we want the outer track to be the first intersection of the ray, we don't want
            # data where the intersect happens after the ray has already intersected with the inner track
            d = 1e12
            lmid = None
            for lout in self.out_lines:
                pInt = get_intersection_point_lineseg_lineseg(lout, lt1, l2_is_ray=True)
                if pInt is not None:
                    d_temp = calc_euclid_distance_2d_sq(lt1.p1, pInt)
                    if d_temp < d and d_temp < din_lt1:
                        d = d_temp
                        lmid = lt1
                # now fire in the opposite direction
                pInt = get_intersection_point_lineseg_lineseg(lout, lt2, l2_is_ray=True)
                if pInt is not None:
                    d_temp = calc_euclid_distance_2d_sq(lt2.p1, pInt)
                    if d_temp < d and d_temp < din_lt2:
                        d = d_temp
                        lmid = lt2

            # set the point at half the distance
            cent_pts_raw[i,:] = np.array(lmid.p1) + lmid.v_hat * (np.sqrt(d) / 2)
            if i == 0:
                # set the start line if the first point
                self.startLine = Line(lmid.p1, tuple(np.array(lmid.p1) + lmid.v_hat * (np.sqrt(d))))

        # now set points every metre along the centre points
        pt_spc = 1.0 # [m]
        d = 0.0
        cent_points = np.array([])
        for i,p0 in enumerate(cent_pts_raw):
            if i == 0:
                cent_points = np.array([p0])
            if i == (cent_pts_raw.shape[0] - 1):
                p1 = cent_pts_raw[0, :]
            else:
                p1 = cent_pts_raw[i + 1, :]
            lt = Line(tuple(p0), tuple(p1))
            d += lt.v_mag # add the distance to the next point
            if d > pt_spc:
                # the distance covered is now greater than the point spacing.
                # therefore place a point at on this line until the distance
                # falls below the point spacing
                NPoints = int(d // pt_spc)
                new_points = np.zeros((NPoints, 2))
                for ii in range(0, NPoints):
                    new_points[ii,:] = np.array(lt.p1) + lt.v_hat * (ii + 1) * pt_spc
                cent_points = np.vstack((cent_points, new_points))
                # set the distance equal to the remaining distance
                d = d % pt_spc
        self.cent_lines = []
        for i,p1 in enumerate(cent_points):
            if i == (cent_points.shape[0] - 1):
                p2 = cent_points[0, :]
            else:
                p2 = cent_points[i + 1, :]
            self.cent_lines.append(Line(tuple(p1), tuple(p2)))

        # now set the starting position on the centre line
        self.startPos = np.array(self.cent_lines[0].p1)


    def pickle_track(self):
        """
            Save the track as a binary by pickling it - stops people from cheating!
        """
        import pickle
        pickle.dump(self, open(self.module_path + '/../data/track/' + self.track_name + '.track', 'wb'))


if __name__ == "__main__":
    from .track import Track
    t = Track()
    t.load_from_csv('barcelona')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for i in t.in_lines:
        ax.plot([i.x1, i.x2],[i.y1,i.y2],'r')
    for i in t.out_lines:
        ax.plot([i.x1, i.x2],[i.y1,i.y2],'b')
    for i in t.cent_lines:
        ax.plot(i.x1,i.y1,'g.')
    i = t.startLine
    ax.plot([i.x1, i.x2],[i.y1,i.y2],'g')
    ax.plot(t.startPos[0],t.startPos[1],'go')
    plt.show()
    t.pickle_track()
    print('Track Length: ',len(t.cent_lines), 'm')
