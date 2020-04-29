from src.geom import Line
from src.geom import calc_angle_between_unit_vectors
from src.track import TrackStore
from copy import deepcopy
import numpy as np
import os

def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real

def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    """
        Create band limited noise
    """
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
    f[idx] = 1
    return fftnoise(f)

def create_base_signal(num_track_points: int, min_freq: int, max_freq: int):
    """
        Returns 1 second of band limited noise, normalised between 0 and 1 with corresponding x axis
    """
    sample_rate = int(num_track_points) # 1s worth of data
    y = band_limited_noise(min_freq, max_freq, num_track_points, sample_rate)
    x = np.linspace(0, 1 / sample_rate * num_track_points, num=num_track_points)
    y = (y - min(y)) / (max(y) - min(y))
    return x,y

def scale_track(x: np.ndarray, y: np.ndarray, x_track_max: float, track_length_max: float):
    # scale the x-axis, the ratio of x_track_max to track_length_max will determine
    # how 'straight' the track is
    x *= x_track_max

    # caclulate the current track length
    xdiffsq = np.diff(x)**2
    ydiff = np.diff(y)
    track_length = np.sum(np.sqrt(xdiffsq + (ydiff)**2))
    track_length_old = track_length
    # the track length is already longer than the max then just return as is
    if track_length < track_length_max:
        # take a first stab at the y_scale required to achieve the desired track length
        y_scale = track_length_max / track_length

        # increase the scaling until we obtain the desired track length
        y_scale_incr = 5
        while track_length < track_length_max:
            y_scale += y_scale_incr
            track_length_old = track_length
            track_length = np.sum(np.sqrt(xdiffsq + (ydiff*y_scale)**2))

        # interpolate between the final too values to get the correct scale
        y_scale -= (1-(track_length_max - track_length_old) / (track_length - track_length_old)) * y_scale_incr
        y *= y_scale

    return x, y

def curvature_too_low(x: np.ndarray, y: np.ndarray):
    # calcualte the instaneous radius of curvature
    dydx = np.gradient(y,x)
    dy2d2x = np.gradient(dydx,x)
    rad_curve = abs((1 + dydx**2)**(3 / 2) / dy2d2x)

    MAX_RADIUS_OF_CURVATURE = 6
    print(min(rad_curve))
    if min(rad_curve) < MAX_RADIUS_OF_CURVATURE:
        # min radius of curvature too low, throw this track away
        return True
    else:
        return False

def calc_track_extemities(x: np.ndarray, y: np.ndarray, in_w: np.ndarray, out_w: np.ndarray):
    n_point = len(x) - 2
    in_p = np.zeros((n_point,2))
    out_p = np.zeros((n_point,2))
    idx = 0
    for i in range(1, len(x)-1):
        # we take the angle between the current and preceeding vectors and bisect it
        l0 = Line((x[i - 1], y[i - 1]), (x[i], y[i]))
        l = Line((x[i], y[i]), (x[i + 1], y[i + 1]))
        a0 = calc_angle_between_unit_vectors(l0.v_hat, l.v_hat) / 2
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
        x1 = l.v_hat[0]  # give him some div by zero protection!
        if abs(x1) < 1e-15:
            if x1 >= 0:
                x1 = 1e-15
            else:
                x1 = -1e-15
        y1 = l.v_hat[1]
        yn = (np.cos(a0) - x0 * np.cos(a0) / x1) / (y0 - x0 * y1 / x1)
        xn = (np.cos(a0) - y1 * yn) / x1
        # generate line objects firing in opposite directions
        lt1 = Line(l.p1, (l.x1 + xn, l.y1 + yn))
        lt2 = Line(l.p1, (l.x1 - xn, l.y1 - yn))
        # determine which line is inner and which is outer. Given the original
        # noise time series cannot go back on itself, and is 2nd order smooth the
        # line with the higher y value @ point 2 will be the outer track
        if lt1.v_hat[1] > lt2.v_hat[1]:
            p_out = lt1.p1 + lt1.v_hat * out_w[i]
            p_in = lt2.p1 + lt2.v_hat * in_w[i]
        else:
            p_out = lt2.p1 + lt2.v_hat * out_w[i]
            p_in = lt1.p1 + lt1.v_hat * in_w[i]
        #p_in = np.copy(p_out)
        #p_in[1] -= out_w[i]+in_w[i]
        # set the points
        in_p[idx, :] = p_in
        out_p[idx, :] = p_out
        idx += 1

    return in_p, out_p

def add_track_to_store(ts: TrackStore, track_name: str, in_p: np.ndarray, out_p: np.ndarray):
    # save as csv files
    module_path = os.path.dirname(os.path.abspath(__file__))
    save_path = module_path + '/../data/track/'
    np.savetxt(save_path + track_name + '_IN.csv', in_p, delimiter=',')
    np.savetxt(save_path + track_name + '_OUT.csv', out_p, delimiter=',')

    # load from csv
    ts.load_from_csv(track_name, is_closed=False)

    # delete csv file
    os.remove(save_path + track_name + '_IN.csv')
    os.remove(save_path + track_name + '_OUT.csv')

NUM_TRACK_PER_COMPLEXITY = 5
MAX_COMPLEXITY = 5
NUM_TRACKS = NUM_TRACK_PER_COMPLEXITY * MAX_COMPLEXITY
MIN_HALF_TRACK_WIDTH = 3
MAX_HALF_TRACK_WIDTH = 7
MIN_FREQ = 1
NUM_SAMPLES = 400
PLOT_TRACKS = True

ts = TrackStore('rl_training_set')
ts.cat_length = NUM_TRACK_PER_COMPLEXITY
track_idx = 0
binned_tracks = 0

if PLOT_TRACKS:
    import matplotlib.pyplot as plt
    plt.ion()
    plt.show()

complexity = -1
track_binned = False
while track_idx < NUM_TRACKS:

    if track_idx % NUM_TRACK_PER_COMPLEXITY == 0 and not track_binned:
        complexity += 1

    if complexity == 0:
        # short track
        track_length_max = 100
        # straight track
        x_track_max = 99
        # low frequency content
        max_freq = 1
        # constant and symetric track width
        track_width_type = 0
    elif complexity == 1:
        # longer track length
        track_length_max = 100
        # increase curvature
        x_track_max = int(track_length_max * 0.8)
        # higher frequency content
        max_freq = np.random.randint(1, 3)
        # constant and symetric track width
        track_width_type = 0
    elif complexity == 2:
        # longer track length
        track_length_max = 400
        # maintain curvature
        x_track_max = int(track_length_max * 0.8)
        # same frequency content
        max_freq = np.random.randint(3, 5)
        # variable and symetric track width
        track_width_type = 0
    elif complexity == 3:
        # longer track length
        track_length_max = 600
        # increase curvature
        x_track_max = 400
        # higher frequency content
        max_freq = np.random.randint(4, 7)
        # variable and symetric track width
        track_width_type = 0
    elif complexity == 4:
        # longer track length
        track_length_max = 1000
        # increase curvature
        x_track_max = 700
        # higher frequency content
        max_freq = np.random.randint(7, 10)
        # variable and asymetric track width
        track_width_type = 0

    # create the scaled track using band limited noise
    x, y = create_base_signal(NUM_SAMPLES, MIN_FREQ, max_freq)
    x, y = scale_track(x, y, x_track_max, track_length_max)
    if curvature_too_low(x, y):
        track_binned = True
        binned_tracks += 1
        print(f'Thrown track of complexity {complexity} away...')
        continue
    else:
        track_binned = False

    print(f'Track {track_idx} passed, creating with complexity {complexity}')

    if track_width_type == 0:
        # constant and symetric track width about the centreline
        in_track_half_width = np.ones(NUM_SAMPLES) * (np.random.rand() * (MAX_HALF_TRACK_WIDTH - MIN_HALF_TRACK_WIDTH) + MIN_HALF_TRACK_WIDTH)
        out_track_half_width = np.copy(in_track_half_width)
    elif track_width_type == 1:
        # track width varies but is symetrical about the centreline
        nom_track_half_width = np.random.rand() * (MAX_HALF_TRACK_WIDTH - MIN_HALF_TRACK_WIDTH) + MIN_HALF_TRACK_WIDTH
        _, w_var = create_base_signal(NUM_SAMPLES, 1, 3)
        # scale based on limits and nominal
        w_var = w_var * (MAX_HALF_TRACK_WIDTH - MIN_HALF_TRACK_WIDTH) - (nom_track_half_width - MIN_HALF_TRACK_WIDTH)
        # set the in and out track half widths to be equal
        in_track_half_width = nom_track_half_width + w_var
        out_track_half_width = np.copy(in_track_half_width)
    else:
        for i in range(2):
            # track width varies assymetrically about the centreline
            nom_track_half_width = np.random.rand() * (MAX_HALF_TRACK_WIDTH - MIN_HALF_TRACK_WIDTH) + MIN_HALF_TRACK_WIDTH
            # calc the width variation
            _, w_var = create_base_signal(NUM_SAMPLES, 1, 3)
            # scale based on limits and nominal
            w_var = w_var * (MAX_HALF_TRACK_WIDTH - MIN_HALF_TRACK_WIDTH) - (nom_track_half_width - MIN_HALF_TRACK_WIDTH)
            if i == 0:
                in_track_half_width = nom_track_half_width + w_var
            else:
                out_track_half_width = nom_track_half_width + w_var

    in_points_final, out_points_final = calc_track_extemities(x, y, in_track_half_width, out_track_half_width)
    if PLOT_TRACKS:
        plt.clf()
        plt.plot(in_points_final[:,0], in_points_final[:,1])
        plt.plot(out_points_final[:,0], out_points_final[:,1])
        plt.title(f'Track Complexity {complexity}')
        max_x = max(in_points_final[:,0])+50
        plt.xlim(0, max_x)
        plt.ylim(-max_x/2, max_x)
        plt.draw()
        plt.pause(0.1)
    add_track_to_store(ts, ts.name+str(track_idx), in_points_final, out_points_final)
    track_idx += 1

ts.pickle_track()
print(f'Tracks Created with {binned_tracks} thrown away due to curvature')
