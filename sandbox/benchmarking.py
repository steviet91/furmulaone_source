"""
Module to allow for easy benchmarking of functions, etc. to allow for A/B testing in efficiencies, etc.
"""
import timeit
import sys
import time
sys.path.append('../src')

# Number of times to loop through the function
loop_limit = 100000


def test_circle_line_intersection():
    # Import the function & required stuff
    from geom import Line, Circle, check_for_intersection_lineseg_circle_old, check_for_intersection_lineseg_circle
    # Create a line and circle
    line = Line((-200,-100), (-200,100))
    circle = Circle(35,35,380.5)
    # Benchmark it!
    start_time = time.time()
    for i in range(loop_limit):
        bIntersect, ts = check_for_intersection_lineseg_circle(line, circle)
        # bIntersect, ts = check_for_intersection_lineseg_circle_old(line, circle)
    end_time = time.time()
    run_time = end_time - start_time
    print("Benchmarking took {}s in total, so ~{}s per function call".format(run_time, run_time/loop_limit))
    # bIntersect, ts = check_for_intersection_lineseg_circle_alt(line, circle)
    print("Return values: {}, {}".format(bIntersect,  ts))
    # Performance improvement: ~100x faster if infinite line doesn't intersect, 50x faster if it does.

def test_t_u_lineseg_lineseg():
    # Import the function & required stuff
    from geom import Line, calc_t_lineseg_lineseg, calc_u_lineseg_lineseg, calc_t_u_lineseg_lineseg
    # Create a line and circle
    l1 = Line((-200,-100), (200,10))
    l2= Line((100,-100), (100,500))
    # Benchmark it!
    start_time = time.time()
    for i in range(loop_limit):
        t = calc_t_lineseg_lineseg(l1.x1, l1.y1, l1.x2, l1.y2, l2.x1, l2.y1, l2.x2, l2.y2)
        u = calc_u_lineseg_lineseg(l1.x1, l1.y1, l1.x2, l1.y2, l2.x1, l2.y1, l2.x2, l2.y2)
        # t, u = calc_t_u_lineseg_lineseg(l1.x1, l1.y1, l1.x2, l1.y2, l2.x1, l2.y1, l2.x2, l2.y2)
    end_time = time.time()
    run_time = end_time - start_time
    print("Benchmarking took {}s in total, so ~{}s per function call".format(run_time, run_time/loop_limit))
    # bIntersect, ts = check_for_intersection_lineseg_circle_alt(line, circle)
    print("Return values: {}, {}".format(t,  u))
    # Performance improvement: ~35% faster.





if __name__ == "__main__":
    # test_circle_line_intersection()
    test_t_u_lineseg_lineseg()