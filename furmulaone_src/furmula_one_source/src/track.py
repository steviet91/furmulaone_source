import csv
import numpy as np
from .geom import Line
import os


class TrackHandler(object):
    def __init__(self, track_name: str):
        # get the module path
        self.module_path = os.path.dirname(os.path.abspath(__file__))

        import pickle
        # load the track
        self.data = pickle.load(open(self.module_path + '/../data/track/' + track_name + '.track', 'rb'))

    def 

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

    def load_from_csv(self, track_name: str):
        """
            Load raw points from the csv files and turn into two sets of line objects

            Assumes the csv file follows rows of points [X, Y] and the in/out datasets
            take a _IN / _OUT suffix
        """
        # save the track name
        self.track_name = track_name

        in_raw_points = np.genfromtxt(self.module_path + '/../data/track/' + track_name + '_IN.csv', delimiter = ',')
        out_raw_points = np.genfromtxt(self.module_path + '/../data/track/' + track_name + '_OUT.csv', delimiter = ',')

        # convert the in points to lines
        self.in_lines = []
        for i in range(0,in_raw_points.shape[0]):
            if i == in_raw_points.shape[0] - 1:
                # last point, join to the first
                p2 = tuple(in_raw_points[0, :])
            else:
                p2 = tuple(in_raw_points[i + 1, :])
            p1 = tuple(in_raw_points[i, :])
            self.in_lines.append(Line(p1, p2))

        # convert the out points to lines
        self.out_lines = []
        for i in range(0,out_raw_points.shape[0]):
            if i == out_raw_points.shape[0] - 1:
                # last point, join to the first
                p2 = tuple(out_raw_points[0, :])
            else:
                p2 = tuple(out_raw_points[i + 1, :])
            p1 = tuple(out_raw_points[i, :])
            self.out_lines.append(Line(p1, p2))

        # set the starting line
        self.startLine = Line(tuple(self.in_lines[0].p1), tuple(self.out_lines[0].p1))
        
        # set the starting position
        self.startPos = self.startLine.p1 + 0.5 * self.startLine.v

        # calculate the starting orientation to apply to the track
        # this is based on the vector of the first line in the inner set
        self.aTrackRotation0 = np.arctan2(self.in_lines[0].v[1],self.in_lines[0].v[0])

def pickle_track(track: Track):
    """
        Save the track as a binary by pickling it - stops people from cheating!
    """
    import pickle
    pickle.dump(track, open(track.module_path + '/../data/track/' + track.track_name + '.track', 'wb'))

if __name__ == "__main__":    
    t = Track()
    t.load_from_csv('octo_track')
    pickle_track(t)