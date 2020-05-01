import numpy as np
from numba import jit
from numba import njit

class Circle(object):
    """
        Create a circle from a centre and radius
    """
    def __init__(self, x0: float, y0: float, r: float):
        """
            Initialise the object and create the circle
        """
        self.x0 = float(x0)  # centre x
        self.y0 = float(y0)  # centre y
        self.r = float(r)  # radius

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
        self.x1 = float(p1[0])
        self.y1 = float(p1[1])
        self.x2 = float(p2[0])
        self.y2 = float(p2[1])
        self.p1 = np.array([self.x1, self.y1], dtype=np.float64)
        self.p2 = np.array([self.x2, self.y2], dtype=np.float64)
        self.p1orig = np.copy(self.p1)  # store the original points
        self.p2orig = np.copy(self.p2)  # store the original points
        self.dX = float(0.0)
        self.dY = float(0.0)
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
        rotate_point(cX, cY, aRot, self.p1)
        rotate_point(cX, cY, aRot, self.p2)

        # update the point coordinates
        self.x1 = self.p1[0]
        self.y1 = self.p1[1]
        self.x2 = self.p2[0]
        self.y2 = self.p2[1]

        # update the vectors
        self.set_vector()
        self.set_unit_vector()

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
        self.set_vector()
        self.set_unit_vector()

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
    # t = calc_t_lineseg_lineseg(l1.x1, l1.y1, l1.x2, l1.y2, l2.x1, l2.y1, l2.x2, l2.y2)
    # u = calc_u_lineseg_lineseg(l1.x1, l1.y1, l1.x2, l1.y2, l2.x1, l2.y1, l2.x2, l2.y2)
    t, u = calc_t_u_lineseg_lineseg(l1.x1, l1.y1, l1.x2, l1.y2, l2.x1, l2.y1, l2.x2, l2.y2)
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

@njit
def calc_t_u_lineseg_lineseg(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float):
    """
        Caclulate the Bezier parameters for lines 1 (x/y 1,2) & 2 (x/y 3/4)
    """
    # Get deltas which will be reused in both calcs
    dx12 = x1 - x2
    dx34 = x3 - x4
    dy12 = y1 - y2
    dy34 = y3 - y4
    # Denominator - same for lines 1 & 2
    d = (dx12 * dy34) - (dy12 * dx34)
    if d == 0:
        # Fail fast, don't worry about numerators & return None for both t & u!
        return None, None
    # Numerators
    dx13 = x1 - x3
    dy13 = y1 - y3
    # Line 1
    n1 = (dx13 * dy34) - (dy13 * dx34)
    t = n1 / d
    # Line 2
    n2 = (dx12 * dy13) - (dy12 * dx13)
    u = -n2 / d
    return t, u

@njit
def calc_t_lineseg_lineseg(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float):
    """
        Caclulate the Bezier parameter for line 1 (x/y 1,2)

        Deprecated - replaced by calc_t_u_lineseg_lineseg above
    """
    n = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if d == 0:
        return None
    else:
        return n / d

@njit
def calc_u_lineseg_lineseg(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float):
    """
        Calculate the Bezier parameter for line 2 (x/y 3/4)

        Deprecated - replaced by calc_t_u_lineseg_lineseg above
    """
    n = (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if d == 0:
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
        Check for an intersection between a line segment (finite line) and a circle.

        Checks distance between infinite line and centre of the circle
        If the distance is greater than the radius, then there's no intersection
        If the distance is <= radius, checks the finite line segment touches the circle.
            1. Finds the point on the infinite line that is closest to the circle centre as a fraction of the length of the finite line, 't', as measured from P1->P2
            2. If radius == distance to line, then returns this point as the only intersection
            3. If radius > distance to line, then finds the two intersections, again as fractions of the length of the finite line
                - Find the length of half the chord of the circle cut by the line (simple trig, eqn below)
                - Divide it by the length of the finite line to get it as a t' value
                - Adds / subtracts this value to the 't' that measures the point closest to the circle centre
                - If either of these 't' values is 0 <= t <=1, adds it to the list of [ts] returned, and returns True
                - If neither are between 0-1, returns False

        :returns: Whether there's an intersection, list of 't' values, where t is the ratio along the line l of the intersection
        :rtype: bool, list
    """
    dy = l.y2 - l.y1
    dx = l.x2 - l.x1
    # Using formula from wikipedia (work with distance squared to save calculating roots as it's slow):
    d_numerator = ((dy * c.x0) - (dx * c.y0) + (l.x2 * l.y1) - (l.y2 * l.x1))
    d_denominator_sq = dy*dy + dx*dx
    d_sq = d_numerator * d_numerator / d_denominator_sq
    # Check whether the distance from the circle centre to the line is <= circle radius (again, work with squared values)
    r2 = c.r*c.r
    if d_sq > r2:
        # Circle centre further from infinite line than circle radius ==> no intersection
        return False, None

    # Vector from line p1 to circle centre
    dxc = c.x0 - l.x1
    dyc = c.y0 - l.y1
    # Line dotted with itself
    l2 = (dx * dx) + (dy * dy)
    # Ratio of line segment from P1 to clostest point to circle centre
    # (Dot product of line and vector from p1 to the circle centre) / (line dotted with itself)
    t_centre_normal = (dx * dxc + dy * dyc) / l2
    if d_sq == r2:
        # Then it's tangent
        # Intersection point = closest point ￼on the line to the circle centre
        # Ensure the intersection point is within the segment
        if t_centre_normal <= 1:
            # Then it's within the segment, so we have an intersection
            return True, [t_centre_normal]
        else:
            return False, None
    # If we get here then the infinite line passes through the circle ==> intersects twice
    # Length of chord of a circle, a = 2.sqrt(r2 - d_sq), so half length = sqrt(r2 - d_sq)
    half_chord_length_sq = r2 - d_sq
    t_half_chord_length = np.sqrt(half_chord_length_sq / l2)
    # 't' of the roots = t_centre_normal +/- half chord length
    t_smaller = t_centre_normal - t_half_chord_length
    t_larger = t_centre_normal + t_half_chord_length
    # Ensure they're within the line segment
    ts = []
    for t in [t_smaller, t_larger]:
        if t >= 0 and t <= 1:
            ts.append(t)
    if len(ts) == 0:
        return False, None
    # If we get here, we've got roots!
    return True, ts

def check_for_intersection_lineseg_circle_old(l: Line, c: Circle):
    """
        Check for an intersection between a line segment (finite line) and a circle.

        Uses the wolfram method for an infinite line and circle. If there are
        any intersects here then it will check if 0 <= t <= 1 to confirm the
        intersection lies on the line segment, where:
        Line Seg:
            x = x1 + t(x2 - x1)
            y = y1 + t(y2 - y1)

        The wolfram approach considers a circle who's centre lies at (0, 0),
        appropriate translations are made to account for this.

        An initial scaling is applied to ensure the values used in the equations
        remain small by checking the distance between the circles centre and the
        line vertices

    """

    # check the distance from the circle centre to the line points. We're trying
    # to reduce the magnitude of the number involved in the calculcation here.
    # given the return value is t the returns will remain valid

    large_threshold = float(100)
    d1 = calc_euclid_distance_2d(tuple(l.p1), (c.x0, c.y0))
    d2 = calc_euclid_distance_2d(tuple(l.p2), (c.x0, c.y0))
    if  d1 > large_threshold:
        scaling = d1 / large_threshold
        # scale the circle
        c = Circle(c.x0 / scaling, c.y0 / scaling, c.r / scaling)
        # scale the line
        l = Line(tuple(l.p1 / scaling), tuple(l.p2 / scaling))
    elif d2 > large_threshold:
        scaling = d2 / large_threshold
        # scale the circle
        c = Circle(c.x0 / scaling, c.y0 / scaling, c.r / scaling)
        # scale the line
        l = Line(tuple(l.p1 / scaling), tuple(l.p2 / scaling))

    # calculate the offset of the circle to (0, 0)
    c0 = np.array([c.x0, c.y0])

    # make a new temporary line with offset
    p1 = l.p1 - c0
    p2 = l.p2 - c0
    l0 = Line(tuple(p1), tuple(p2))

    # set the variables
    dx = l0.x2 - l0.x1
    dy = l0.y2 - l0.y1
    dr = np.sqrt(dx**2 + dy**2)
    D = l0.x1 * l0.y2 - l0.x2 * l0.y1

    # determine if the roots are complex
    w = (c.r**2 * dr**2 - D**2)
    if w < 0:
        # roots will be complex
        return False, None
    elif w == 0:
        # only one intersection, the line is tangent
        x_root = np.array([D * dy / dr**2])
        y_root = np.array([-1 * D * dx / dr**2])
    else:
        # intersection
        if dy < 0:
            # can't use np.sign as it return 0 in the case np.sign(0)
            sign_dy = -1.0
        else:
            sign_dy = 1.0
        ux = sign_dy * dx * np.sqrt(w)
        x_root = np.array([D * dy + ux, D * dy - ux]) / dr**2
        uy = np.abs(dy) * np.sqrt(w)
        y_root = np.array([-1 * D * dx + uy, -1 * D * dx - uy]) / dr**2

    # we now have the intersections of an infinite line with a cirlce at (0, 0)
    # translate the coordinates back to the circle origin and confirm
    # 0 <= t <= 1 is satisified, i.e. at least one of the intersections lies on
    # our finite line

    # use x to check {x = tx2 + (1 - t)x1} unless the demon is 0 (line is vertical)
    # in which case revert to y
    ts = []
    if (l0.x2 - l0.x1) != 0:
        for x in x_root:
            t = (x - l0.x1) / (l0.x2 - l0.x1)
            if 0 <= t <= 1:
                ts.append(t)
    else:
        for y in y_root:
            t = (y - l0.y1) / (l0.y2 - l0.y1)
            if 0 <= t <= 1:
                ts.append(t)

    if len(ts) == 0:
        # no intersections with our finite line
        return False, None
    else:
        # we have an intersection
        return True, ts

@njit
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

@jit
def calc_euclid_distance_2d(p1: tuple, p2: tuple):
    """
        Returns the euclidian distance between p1 and p2
    """
    return np.sqrt(calc_euclid_distance_2d_sq(p1, p2))

@jit
def calc_euclid_distance_2d_sq(p1: tuple, p2: tuple):
    """
        Returns the square of the euclidian distance between p1 and p2

        Useful as it's much cheaper than calculating the actual distance
        (as it save a call to sqrt())
        and if checking a < b, then a^2 < b^2 will also give the correct value
    """
    return (float(p2[0]) - float(p1[0]))**2 + (float(p2[1]) - float(p1[1]))**2

def calc_angle_between_unit_vectors(v1_hat, v2_hat):
    """
        Return the angle (rad) between two unit vectors
    """
    return np.arccos(np.dot(v1_hat,v2_hat))










if __name__ == "__main__":
    """
    ls = [Line((-1.722077545, -4.1574579), (539.4740247, -545.35355804)),
        Line((539.4740247,  -545.35355804), (539.4740247,  -1310.72042277)),
        Line(( 539.4740247,  -1310.72042277), (-1.72207545e+00, -1.85191652e+03)),
        Line((-1.72207545e+00, -1.85191652e+03), ( -767.08894018, -1851.91652292)),
        Line(( -767.08894018, -1851.91652292), (-1308.28504032, -1310.72042277)),
        Line((-1308.28504032, -1310.72042277), (-1308.28504032,  -545.35355804)),
        Line((-1308.28504032,  -545.35355804), (-767.08894018,   -4.1574579)),
        Line((-767.08894018,   -4.1574579), (-1.72207545, -4.1574579))]
    """

    ls = [Line((-2, -4), (534, -545)),
        Line((540,  -545.4), (540,  -1311)),
        Line(( 540,  -1311), (-2, -1852)),
        Line((-2, -1852), ( -767, -1852)),
        Line(( -767, -1852), (-1308, -1311)),
        Line((-1308, -1311), (-1308,  -545)),
        Line((-1308,  -545), (-767,   -4)),
        Line((-767,   -4), (-2, -4))]
    #c = Circle(28.2472, -27.873434, 5)
    #ls = [Line((-767,   -4), (-2, -4))]
    c = Circle(0, 0, 200)
    #ls = [Line((0.5,2),(2,8))]
    #c = Circle(1 , 4, 0.5)

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
