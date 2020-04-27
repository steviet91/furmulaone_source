import numpy as np


data = np.genfromtxt('fit_data.csv', delimiter=',')
migr = np.genfromtxt('migr_data.csv', delimiter=',')

import matplotlib.pyplot as plt

fig,ax = plt.subplots()
f_idx = 272
x = np.cumsum(np.ones(272))

for i in range(data.shape[1]):
    ax.plot(x, data[0:f_idx, i], label='Island {}'.format(i))
ax.legend()
for i in range(migr.shape[0]):
    y1 = data[int(migr[i, 0]), int(migr[i,1])]
    x1 = migr[i,0]
    y2 = data[int(migr[i, 0]-1), int(migr[i,2])]
    x2 = migr[i,0]+1
    ax.plot([x1, x2], [y1, y2], 'k--', linewidth=0.5)
    ax.plot(x1, y1, 'r*')
    ax.plot(x2, y2, 'g*')

ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')

    #ax.text(migr[i,0],y-0.05, '{} sends {} to {}'.format(migr[i,1], migr[i,3], migr[i,2]))

plt.show()
