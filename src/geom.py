import numpy as np


class Circle(object):
    """
        Create a circle from a centre and radius
    """
    def __init__(self, x0: float, y0: float, r: float):
        """
            Initialise the object and create the circle
        """
        self.x0 = x0 # centre x
        self.y0 = y0 # centre y
        self.r = r # radiu

    def update_centre_by_delta(self, dX: float, dY: float):
        """
            Update the centre of the circle by a given amount
        """
        self.x0 += dX
        self.y0 += dY

class Arc(Circle):
    """
        Inherits circle but provides angular limits to create arc.
        0 rad is assumed to always be on the x-axis, therefore the arc
        will need to be rotated if it is on a moving body. Min is the limit in the anti-clockwise
        direction and max is the limit in the clockwise direction from the x-axis
    """
    def __init__(self, x0: float, y0: float, r: float, aMin: float, aMax: float, aInitRot: float):
        super().__init__(x0, y0, r) # initiale the 
        self.aMin = aMin # minimum angle
        self.aMax = aMax # maximum angle
        self.aRot = aInitRot # alignment of the x-axis
        self.aRotOrig = self.aRot # original angle

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
        self.v = np.array([self.x2 - self.x1, self.y2 - self.y1])
        self.v_hat = self.get_unit_vector(self.v)
        self.v_mag = np.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)

    def get_unit_vector(self, v):
        """
            Calculate the unit vector from p1 to p2
        """
        return v / np.linalg.norm(v)

    
#### LINE SEG LINE SEG FUNCTION ####
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
    do_intersect, t = check_for_intersection_lineseg_lineseg(l1, l2, l2_is_ray = l2_is_ray)
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
    print(n,d)
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
    print(n,d)
    if d == None:
        return None
    else:
        return -1 * n / d

#### LINE SEG ARC SEG FUNCTIONS ####
def get_intersection_point_lineseg_arcseg(l: Line, a: Arc):
    """
        Return the intersection between a line segment and an arc segment. If they do not intersect then None is returned
    """
    ps = get_intersection_point_lineseg_circle(l, a)

    if p is None:
        return None
    else:
        for p in ps:        
            theta = np.arcsin(p[0]/a.r) - a.aRot # account for x-axis alignment
            if a.aMin <= theta <= a.aMax:
                # return the point, it lies on our arc. Note if it intersects more than once the nonly the first will be returned
                return p
        # No points have satified our arc
        return None
            
#### LINE SEG CIRLE FUNCTIONS ####
def get_intersection_point_lineseg_circle(l: Line, c):
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

def check_for_intersection_lineseg_circle(l: Line, c):
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
    coeff = [0,0,0]
    # t**2
    coeff[0] = (l.x2**2 - 2 * l.x2 * l.x1 + l.x1**2 + l.y2**2 - 2 * l.y2 * l.y1 + l.y1**2)
    # t
    coeff[1] = (2 * l.x2 * l.x1 - 2 * l.x2 * c.x0 - 2 * l.x1**2 + 2 * l.x1 * c.x0 + 2 * l.y2 * l.y1 - 2 * l.y2 * c.y0 - 2 * l.y1**2 + 2 * l.y1 * c.y0)
    # const
    coeff[2] = (l.x1**2 - 2 * c.y0 * l.x1 + c.x0**2 + l.y1**2 - 2 * c.y0 * l.y1 + c.y0**2 - c.r**2)

    ts_temp =  np.roots(coeff)
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

if __name__ == "__main__":
    l1 = Line((-1,-1),(1,1))
    c = Circle(0, 0, 0.5)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    ax.plot([l1.x1, l1.x2],[l1.y1,l1.y2])
    circle1 = plt.Circle((c.x0,c.y0),c.r,color='r',fill=True)
    ax.add_artist(circle1)
    ps = get_intersection_point_lineseg_circle(l1,c)
    if ps is not None:
        for p in ps:
            ax.plot(p[0],p[1],'*',markersize=10)
    plt.show()
