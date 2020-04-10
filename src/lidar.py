import numpy as np
from .track import TrackHandler
from .geom import Line
from .geom import Circle
from .geom import get_intersection_point_lineseg_lineseg
from .geom import calc_euclid_distance_2d

class Lidar(object):
    """
        Object for the LIDAR containing the rays and intersect data
    """

    def __init__(self, track: TrackHandler, a0: float, x0: float, y0: float, aFov: float):
        """
            Initialise the LIDAR object
        """
        # save the arguments
        self.track = track
        self.a0 = a0  # the initial nominal angle of the lidar (the centre ray)
        self.x0 = x0  # the position of the lidar
        self.y0 = y0  # the y position of the lidar
        self.aFov = aFov  # the field of view of the lidar

        # set the properties of the lidar
        self.NRays = 20  # the number of rays in the lidar
        self.xLidarRange = 20  # range in meters

        # initialise the lidar rays
        self.initialise_rays()

        # initialise the lidar collision circle
        self.collisionCircle = Circle(self.x0, self.y0, self.xLidarRange)

        # intialise the collision array
        self.initialise_collision_array()


    def initialise_rays(self):
        """
            Set up the rays that represent the lidar
        """
        self.rays = []
        for i in range(0,self.NRays):
            a = self.a0 - self.aFov / 2 + i * self.aFov / self.NRays  # angle of this ray
            # going to work as a unit vector
            x = np.cos(a)
            y = np.sin(a)
            # instantiate the ray as a line
            self.rays.append(Line((self.x0, self.y0), (self.x0 + x, self.y0 + y)))

    def initialise_collision_array(self):
        """
            Collision array contains the distance of the collision for each ray.
            A negative number (-1) infers no collision found
        """
        self.collision_array = -1.0 * np.ones(self.NRays, dtype=np.float64)

    def rotate_lidar_by_delta(self, daRot: float, cX: float, cY: float):
        """
            Rotate the lidars about a pivot point
        """
        for r in self.rays:
            r.rotate_line_by_delta(daRot, cX, cY)

    def translate_lidars_by_delta(self, dX: float, dY: float):
        """
            Translate the lidars by the given deltas
        """
        for r in self.rays:
            r.translate_line_by_delta(dX, dY)
        self.collisionCircle.update_centre_by_delta(dX, dY)
        self.x0 += dX
        self.y0 += dY

    def reset_lidar(self):
        """
            Reset the lidars to their previous position/angles
        """
        for r in self.rays:
            r.reset_line()
        self.collisionCircle.update_centre_to_new_pos(self.x0, self.y0)
        self.initialise_collision_array()

    def fire_lidar(self):
        """
            Determine the distance to the nearest
        """
        # find the indexes of the track segments to check collision for
        in_idxs, out_idxs = self.track.get_line_idxs_for_collision(self.collisionCircle)

        # reset the collision array - i.e. set all values to -1
        self.initialise_collision_array()

        # check the inside track set fist
        if len(in_idxs) > 0:
            check_lines = [self.track.data.in_lines[i] for i in in_idxs]
            # cast each ray onto the track and return the minimum collision distance
            for i,r in enumerate(self.rays):
                # calc the distances to each segment for this ray
                ds = np.array([self.cast_ray(r, l) for l in check_lines])

                # check the returned values
                if np.max(ds) > 0:
                    # get the minimun distance
                    self.collision_array[i] = np.min(ds[np.where(ds > 0)[0]])

        # check the outside track
        if len(out_idxs) > 0:
            check_lines = [self.track.data.out_lines[i] for i in out_idxs]
            # cast each ray on the track and return the minimum collision distance
            for i,r in enumerate(self.rays):
                # calc the distance to each segment fro this ray
                ds = np.array([self.cast_ray(r, l) for l in check_lines])

                # check the returned values - this time we need to consider the result of
                # checking the in lines
                if np.max(ds) > 0:
                    # get the minimun distance and compare to the current value
                    min_d = np.min(ds[np.where(ds > 0)[0]])
                    if self.collision_array[i] < 0:
                        # the inner didn't score a collision, so the distance is this one
                        self.collision_array[i] = min_d
                    else:
                        # this inner scored a collision, so we take the min of both
                        self.collision_array[i] = min(min_d, self.collision_array[i])

    def cast_ray(self, r: Line, l: Line):
        """
            Cast the ray r and return the distance to the line l
        """
        pInt = get_intersection_point_lineseg_lineseg(l, r, l2_is_ray=True)

        if pInt is None:
            return float(-1)
        else:
            return float(calc_euclid_distance_2d(tuple(r.p1), tuple(pInt)))
