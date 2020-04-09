import numpy as np


class Circle(object):
    """
        Create a circle from a centre and radius
    """
    def __init__(self, x0: float, y0: float, r: float):
        """
            Initialise the object and create the circle
        """
        self.x0 = x0  # centre x
        self.y0 = y0  # centre y
        self.r = r  # radius

    def update_centre_by_delta(self, dX: float, dY: float):
        """
            Update the centre of the circle by a given amount
        """
        self.x0 += dX
        self.y0 += dY

    def update_centre_to_new_pos(self, x: float, y: float):
        """
            Update the centre to a new location
        """
        self.x0 = x
        self.y0 = y


class Arc(Circle):
    """
        Inherits circle but provides angular limits to create arc.
        0 rad is assumed to always be on the x-axis, therefore the arc
        will need to be rotated if it is on a moving body. Min is the limit in the anti-clockwise
        direction and max is the limit in the clockwise direction from the x-axis
    """
    def __init__(self, x0: float, y0: float, r: float, aMin: float, aMax: float, aInitRot: float):
        super().__init__(x0, y0, r)  # run the super class init
        self.aMin = aMin  # minimum angle
        self.aMax = aMax  # maximum angle
        self.aRot = aInitRot  # alignment of the x-axis
        self.aRotOrig = self.aRot  # original angle

    def update_orientation_by_detla(self, daRot: float):
        """
            Update the x-axis rotation by a given delta
        """
        self.aRot += daRot

    def reset_orientaion(self, aRotUpdate: float = None):
        """
            Reset the rotation of the x-axis. If None then back to the original.
            Else to the new angle provided
        """
        if aRotUpdate is None:
            self.aRot = self.aRotOrig
        else:
            self.aRot = aRotUpdate

class Line(object):
    """
        Create a line segment from two 2D points and provide some useful functions and properties
    """

    def __init__(self, p1: tuple, p2: tuple):
        """
            Initialise the object and create the line
        """
        self.x1 = p1[0]
        self.x2 = p2[0]
        self.y1 = p1[1]
        self.y2 = p2[1]
        self.p1 = np.array([self.x1, self.y1])
        self.p2 = np.array([self.x2, self.y2])
        self.p1orig = np.copy(self.p1)  # store the original points
        self.p2orig = np.copy(self.p2)  # store the original points
        self.dX = 0.0
        self.dY = 0.0
        self.set_vector()
        self.set_unit_vector()
        self.set_magnitude()

    def set_vector(self):
        """
            Sets the vector of the line
        """
        self.v = np.array([self.x2 - self.x1, self.y2 - self.y1])

    def set_magnitude(self):
        """
            Set the magnitude of the vector
        """
        self.v_mag = np.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)

    def set_unit_vector(self):
        """
            Calculate the unit vector from p1 to p2
        """
        self.v_hat = self.v / np.linalg.norm(self.v)

    def translate_line_by_delta(self, dX: float, dY: float):
        """
            Translate the line by a given amount
        """
        # translate the points
        self.p1 += np.array([dX, dY])
        self.p2 += np.array([dX, dY])

        # update the coordinates
        self.x1 = self.p1[0]
        self.y1 = self.p1[1]
        self.x2 = self.p2[0]
        self.y2 = self.p2[1]

    def rotate_line_by_delta(self, aRot: float, cX: float, cY: float):
        """
            Rotate a line by a given angle about point cX cY - note this is applied as a delta
        """
        # rotate the points
        self.p1 = rotate_point(cX, cY, aRot, self.p1)
        self.p2 = rotate_point(cX, cY, aRot, self.p2)

        # update the point coordinates
        self.x1 = self.p1[0]
        self.y1 = self.p1[1]
        self.x2 = self.p2[0]
        self.y2 = self.p2[1]

        # update the vectors
        self.set_unit_vector()
        self.set_vector()

    def rotate_line_to_new_angle(self, aRot: float, cX: float, cY: float):
        """
            Rotate the line to a new angle about point cX cY - note this is applied to the
            original coordinates, i.e. it assume p1orig and p2orig are at an angle of 0 rad,
            and then applies the total translatation
        """
        # rotate the original points to get the updates points
        self.p1 = rotate_point(cX, cY, aRot, np.copy(self.p1orig))
        self.p2 = rotate_point(cX, cY, aRot, np.copy(self.p2orig))

        # update the point coordinates
        self.x1 = self.p1[0]
        self.y1 = self.p1[1]
        self.x2 = self.p2[0]
        self.y2 = self.p2[1]

        # update the vectors
        self.set_unit_vector()
        self.set_vector()

    def reset_line(self):
        """
            Reset the line to it's original points
        """
        self.p1 = np.copy(self.p1orig)
        self.p2 = np.copy(self.p2orig)
        self.x1 = self.p1[0]
        self.y1 = self.p1[1]
        self.x2 = self.p2[0]
        self.x2 = self.p2[1]
        self.set_vector()
        self.set_unit_vector()
        self.set_magnitude()




# ### LINE SEG LINE SEG FUNCTION ####
def check_for_intersection_lineseg_lineseg(l1: Line, l2: Line, l2_is_ray: bool = False):
    """
        Returns true if the line provided intersects with provided second line. Logic for the case where l2 is simply a ray
    """
    # calculate the Bezier parameters
    t = calc_t_lineseg_lineseg(l1, l2)
    u = calc_u_lineseg_lineseg(l1, l2)
    if t is None or u is None:
        return False, None
    else:
        # return the check
        if l2_is_ray:
            if (0 <= t <= 1) and (u >= 0):
                return True, t
            else:
                return False, None
        else:
            if (0 <= t <= 1) and (0 <= u <= 1):
                return True, t
            else:
                return False, None


def get_intersection_point_lineseg_lineseg(l1: Line, l2: Line, l2_is_ray: bool = False):
    """
        Return the point of intersection between the two lines. If they do not intersect then None is returned
    """
    do_intersect, t = check_for_intersection_lineseg_lineseg(l1, l2, l2_is_ray=l2_is_ray)
    if do_intersect:
        return l1.p1 + t * l1.v
    else:
        return None

def calc_t_lineseg_lineseg(l1: Line, l2: Line):
    """
        Caclulate the Bezier parameter for line 1 (self)
    """
    n = (l1.x1 - l2.x1) * (l2.y1 - l2.y2) - (l1.y1 - l2.y1) * (l2.x1 - l2.x2)
    d = (l1.x1 - l1.x2) * (l2.y1 - l2.y2) - (l1. y1 - l1.y2) * (l2.x1 - l2.x2)
    if d == 0:
        return None
    else:
        return n / d


def calc_u_lineseg_lineseg(l1: Line, l2: Line):
    """
        Calculate the Bezier parameter for line 2
    """
    n = (l1.x1 - l1.x2) * (l1.y1 - l2.y1) - (l1.y1 - l1.y2) * (l1.x1 - l2.x1)
    d = (l1.x1 - l1.x2) * (l2.y1 - l2.y2) - (l1.y1 - l1.y2) * (l2.x1 - l2.x2)
    if d is None:
        return None
    else:
        return -1 * n / d


# ### LINE SEG ARC SEG FUNCTIONS ####
def get_intersection_point_lineseg_arcseg(l: Line, a: Arc):
    """
        Return the intersection between a line segment and an arc segment. If they do not intersect then None is returned
    """
    ps = get_intersection_point_lineseg_circle(l, a)

    if ps is None:
        return None
    else:
        for p in ps:
            theta = np.arcsin(p[0] / a.r) - a.aRot  # account for x-axis alignment
            if a.aMin <= theta <= a.aMax:
                # return the point, it lies on our arc. Note if it intersects more than once the nonly the first will be returned
                return p
        # No points have satified our arc
        return None


# ### LINE SEG CIRLE FUNCTIONS ####
def get_intersection_point_lineseg_circle(l: Line, c: Circle):
    """
        Return the intersection between a line segment and a circle. If they do not intersect then None is returned
    """
    do_intersect, ts = check_for_intersection_lineseg_circle(l, c)

    if do_intersect:
        ps = []
        for t in ts:
            ps.append(l.p1 + t * l.v)
        return ps
    else:
        return None


def check_for_intersection_lineseg_circle(l: Line, c: Circle):
    """
        Check for an intersection between a line segment and a circle.
        solves for t, where:
        Line Seg:
            x = x1 + t(x2 - x1)
            y = y1 + t(y2 - y1)
        Circle:
            (x - x0)**2 + (y - y0)**2 = r**2

        I've probably F****d up the substition and rearragnement to calc the coefficients
    """
    # need to solve a quadratic
    coeff = [0, 0, 0]
    # t**2
    coeff[0] = (l.x2**2 - 2 * l.x2 * l.x1 + l.x1**2 + l.y2**2 - 2 * l.y2 * l.y1 + l.y1**2)
    # t
    coeff[1] = (2 * l.x2 * l.x1 - 2 * l.x2 * c.x0 - 2 * l.x1**2 + 2 * l.x1 * c.x0 + 2 * l.y2 * l.y1 - 2 * l.y2 * c.y0 - 2 * l.y1**2 + 2 * l.y1 * c.y0)
    # const
    coeff[2] = (l.x1**2 - 2 * c.y0 * l.x1 + c.x0**2 + l.y1**2 - 2 * c.y0 * l.y1 + c.y0**2 - c.r**2)

    ts_temp = np.roots(coeff)
    print(ts_temp)
    if not np.isreal(ts_temp).all():
        # roots are complex
        return False, None
    else:
        ts = []
        do_intersect = False
        # check the roots lie within 0 to 1
        for t in ts_temp:
            if 0 <= t <= 1:
                ts.append(t)
                do_intersect = True
        if do_intersect:
            # roots are real and at least one lies on the line segment
            return do_intersect, ts
        else:
            # roots were real but they didn't lie on the line segment
            return do_intersect, None


def rotate_point(cx: float, cy: float, a: float, p):
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

def calc_euclid_distance_2d(p1: tuple, p2: tuple):
    """
        Returns the euclidian distance between p1 and p2
    """
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def calc_angle_between_unit_vectors(v1_hat, v2_hat):
    """
        Return the angle (rad) between two unit vectors
    """
    return np.arccos(np.dot(v1_hat,v2_hat))










if __name__ == "__main__":
    ls = [Line((-1.722077545, -4.1574579), (539.4740247, -545.35355804)),
        Line((539.4740247,  -545.35355804), (539.4740247,  -1310.72042277)),
        Line(( 539.4740247,  -1310.72042277), (-1.72207545e+00, -1.85191652e+03)),
        Line((-1.72207545e+00, -1.85191652e+03), ( -767.08894018, -1851.91652292)),
        Line(( -767.08894018, -1851.91652292), (-1308.28504032, -1310.72042277)),
        Line((-1308.28504032, -1310.72042277), (-1308.28504032,  -545.35355804)),
        Line((-1308.28504032,  -545.35355804), (-767.08894018,   -4.1574579)),
        Line((-767.08894018,   -4.1574579), (-1.72207545, -4.1574579))]
    #c = Circle(28.2472, -27.873434, 5)
    c = Circle(330, -336, 5)


    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    circle1 = plt.Circle((c.x0, c.y0), c.r, color='r', fill=True)
    ax.add_artist(circle1)
    for l in ls:
        ax.plot([l.x1, l.x2], [l.y1, l.y2])
        ps = get_intersection_point_lineseg_circle(l, c)
        if ps is not None:
            for p in ps:
                ax.plot(p[0], p[1], '*', markersize=10)
    plt.show()
