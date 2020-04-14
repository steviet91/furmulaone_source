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
    # for original check_for_intersection_lineseg_circle, Benchmarking took 0.675400972366333s in total, so ~6.75400972366333e-05s per function call

if __name__ == "__main__":
    test_circle_line_intersection()