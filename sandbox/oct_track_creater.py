import numpy as np
import matplotlib.pyplot as plt
import csv
# small script to create an octogon shaped track

IN_RAD = 200 # m
TRACK_WIDTH = 9 # m
OUT_RAD = IN_RAD + TRACK_WIDTH

delta_angle = 2 * np.pi / 8
start_angle = np.pi


in_track_points = np.zeros((8,2)) # inner track
out_track_points = np.zeros((8,2)) # outer track

for i in range(0,8):
    s = np.sin(start_angle + delta_angle * i)
    c = np.cos(start_angle + delta_angle * i)
    in_track_points[i,0] = IN_RAD * s
    in_track_points[i,1] = IN_RAD * c
    out_track_points[i,0] = OUT_RAD * s
    out_track_points[i,1] = OUT_RAD * c

with open('../data/track/octo_track_IN.csv', 'w', newline = '') as f:
    writer = csv.writer(f, delimiter = ',')
    for i in range(0,8):
        writer.writerow([in_track_points[i,0], in_track_points[i,1]])

with open('../data/track/octo_track_OUT.csv', 'w', newline = '') as f:
    writer = csv.writer(f, delimiter = ',')
    for i in range(0,8):
        writer.writerow([out_track_points[i,0], out_track_points[i,1]])
