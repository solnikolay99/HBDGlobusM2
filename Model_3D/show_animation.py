# ******************************************************************************
# @author: L. I. Nurtdinova
# ******************************************************************************

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def name_reader(directory_path, pattern):
    return sorted(glob.glob(directory_path + '/' + pattern))


directory_path = os.getcwd() + '/data'

coorx = np.load(name_reader(directory_path, 'coorx*')[0])
coory = np.load(name_reader(directory_path, 'coory*')[0])
coorz = np.load(name_reader(directory_path, 'coorz*')[0])

r = 1
coorx, coory, coorz = coorx[::r, :], coory[::r, :], coorz[::r, :]
l = coorx.shape[0]
l = 300000
coorx, coory, coorz = coorx[:l, :], coory[:l, :], coorz[:l, :]

fig = plt.figure(figsize=(6, 6))

ax = fig.add_subplot(projection='3d')

l_color = coorx.shape[0]
initial_elev, initial_azim = 90, 0
colors = np.random.randint(1, 210, l_color)
x, y = 80, 50


def update(frame):
    print(frame)
    ax.clear()
    ax.view_init(elev=initial_elev, azim=initial_azim - frame)
    ax.view_init(elev=initial_elev, azim=initial_azim)
    ax.scatter(coorx[0:l, frame], coorz[0:l, frame], coory[0:l, frame], s=1, c=colors[::], alpha=1, cmap='plasma')
    ax.set_xlim3d(0, x)
    ax.set_ylim3d(0, y)
    ax.set_zlim3d(0, y)


if 1:
    ani = FuncAnimation(fig, update, frames=range(0, 500), interval=1)

plt.show()
