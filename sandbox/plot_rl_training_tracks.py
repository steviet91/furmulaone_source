from src.track import TrackStore
import numpy as np

ts = TrackStore.loader('rl_training_set')


import matplotlib.pyplot as plt
plt.ion()
plt.show()
fig, ax = plt.subplots(1,1)
fig.set_size_inches((10,10))
for i,t in enumerate(ts.store):
    plt.cla()
    plt.title(f'Track: {t.track_name}, Complexity {i//ts.cat_length}')
    xin = [l.x1 for l in t.in_lines] + [t.in_lines[-1].x2]
    yin = [l.y1 for l in t.in_lines] + [t.in_lines[-1].y2]
    ax.plot(xin, yin)
    xout = [l.x1 for l in t.out_lines] + [t.out_lines[-1].x2]
    yout = [l.y1 for l in t.out_lines] + [t.out_lines[-1].y2]
    ax.plot(xout, yout)
    xcent = [l.x1 for l in t.cent_lines]
    ycent = [l.y1 for l in t.cent_lines]
    ax.plot(xcent, ycent)
    ax.plot(t.startPos[0], t.startPos[1],'*')
    ax.plot([t.startLine.x1,t.startLine.x2],[t.startLine.y1,t.startLine.y2],'k--')
    ax.set_aspect('equal')
    plt.draw()
    plt.pause(1)
