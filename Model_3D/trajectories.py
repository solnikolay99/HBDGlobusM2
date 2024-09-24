# ******************************************************************************
# @author: L. I. Nurtdinova
# ******************************************************************************

import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def name_reader(directory_path, pattern):
    return sorted(glob.glob(directory_path + '/' + pattern))


def replace_trailing_ones(matrix):
    for row in matrix:
        non_one_indices = np.where(row != 1)[0]
        if non_one_indices.size > 0:
            last_non_one_index = non_one_indices[-1]
            row[last_non_one_index + 1:] = row[last_non_one_index]
    return matrix


directory_path = os.getcwd() + '/data'

coorx = np.load(name_reader(directory_path, 'coorx*')[0])
coory = np.load(name_reader(directory_path, 'coory*')[0])
coorz = np.load(name_reader(directory_path, 'coorz*')[0])

ln = coorx.shape[0]
x, y = 150, 50
coorx, coory, coorz = replace_trailing_ones(coorx), replace_trailing_ones(coory), replace_trailing_ones(coorz)

plt.figure(figsize=(5, 5))

for i in range(0, ln, 10):
    plt.plot(coorx[i, :], coorz[i, :], linewidth=0.5)
    # plt.scatter(coorx[i, :], coorz[i, :], s=0.9)
plt.grid(color='black', linestyle='-', linewidth=0.2)
plt.show()
