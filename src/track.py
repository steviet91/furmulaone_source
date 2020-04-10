import numpy as np
from .geom import Line
from .geom import rotate_point
from .geom import Circle
from .geom import calc_euclid_distance_2d
from .geom import check_for_intersection_lineseg_circle
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

        # initialise track handler variables
        self.bNewLap = False
        self.bCarNearStartLine = False
        self.tLap = []
        self.tLapStart = time.time()
        self.NLapsComplete = -1

    def start_new_lap(self):
        """
            New lap has started, log the lap times
        """
        if self.NLapsComplete == -1:
            # this is the first lap
            self.tLapStart = time.time()
            self.NLapsComplete += 1

        else:
            # standard new lap - set the time and roll over laps complete
            t = time.time()
            self.tLap.extend(t - self.tLapStart)
            self.NLapsComplete += 1

    def check_new_lap(self, xC: float, yC: float):
        """
            Check for a new lap by looking for a rising edge on the distance of the car to
            the start line vertices. This logic should also work for the first lap.
        """
        # calculate the distance between the car and each vertex
        d1 = np.sqrt((self.data.startLine.x1 - xC)**2 + (self.data.startLine.y1 - yC)**2)
        d2 = np.sqrt((self.data.startLine.x2 - xC)**2 + (self.data.startLine.y2 - yC)**2)

        # check the distance compared to the length of the start line + a tolerance
        if (d1 + d2) < (self.data.startLine.v_mag + self.data.startLine.v_mag * 0.001):
            # we're new the start line
            self.bCarNearStartLine = True

        else:

            if self.bCarNearStartLine:
                # this is a falling edge, and therefore a new lap has begun
                self.bNewLap = True
                self.bCarNearStartLine = False

            elif self.bNewLap:
                # rest the new lap trigger
                self.bNewLap = False

        # if new lap then roll update the lap information
        if self.bNewLap:
            self.start_new_lap()

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
        # check the distance of the points to the circle's centre
        if calc_euclid_distance_2d(l.p1, (c.x0, c.y0)) <= c.r:
            # p1 lies within the circumference of the circle
            return True
        elif calc_euclid_distance_2d(l.p2, (c.x0, c.y0)) <= c.r:
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
        aTrackRotation0 = np.arctan2(rotLine.v[1], rotLine.v[0])

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

        # now set the final start line and poisition
        self.startLine = Line(tuple(self.in_lines[0].p1), tuple(self.out_lines[0].p1))
        self.startPos = np.array([0, 0])

    def pickle_track(self):
        """
            Save the track as a binary by pickling it - stops people from cheating!
        """
        import pickle
        pickle.dump(self, open(self.module_path + '/../data/track/' + self.track_name + '.track', 'wb'))


if __name__ == "__main__":
    from .track import Track
    t = Track()
    t.load_from_csv('octo_track')
    t.pickle_track()
