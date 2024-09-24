# ******************************************************************************
# @author: L. I. Nurtdinova
# ******************************************************************************

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# from scipy.ndimage import gaussian_filter
import copy
# from scipy.ndimage import gaussian_filter
import copy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# import pandas as pd
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 14})


def characteristics(slice, lab):
    square = np.trapz(slice)
    arg_semi = np.argmin(np.abs(slice - np.max(slice) / 2))
    arg_full = np.argmin(np.abs(slice - np.max(slice)))
    semi_wid = np.abs(arg_full - arg_semi)
    print(semi_wid, square)
    new_row = [lab, semi_wid, square]


def load(filename, title, sr, sigma):
    den = np.round(np.load('data/' + filename))
    # den = den[:, :, 24:25]
    # den = den[:, 10:120, :]
    den = np.sum(den, axis=2)
    global x
    global y
    x = int(round(den.shape[1], - 2))
    y = int(round(den.shape[0], - 2))

    if show_profile:
        plt.figure(figsize=(8, 6))
        den_show = copy.deepcopy(den)

        plt.title(lab)
        plt.imshow(1 + den_show, norm=LogNorm(), cmap='plasma')
        plt.xlabel('x, мм')
        plt.ylabel('y, мм')
        plt.show()

    return den


show_profile = 1
sigma = 0
window_size = 15

name = '001'
lab = 'Отверстия в обеих диафрагмах'
slice = load(name + '.npy', lab, 120, sigma)[:, 120]
if not show_profile:
    kernel = np.ones(window_size) / window_size
    slice = np.convolve(slice, kernel, mode='valid')
    plt.plot(slice, label=lab)
    characteristics(slice, lab)

name = '002'
lab = 'Отверстия в первой диафрагме'
slice = load(name + '.npy', lab, 120, sigma)[:, 120]
if not show_profile:
    kernel = np.ones(window_size) / window_size
    slice = np.convolve(slice, kernel, mode='valid')
    plt.plot(slice, label=lab)
    characteristics(slice, lab)

name = '004'
lab = 'При наличии внешней стенки'
slice = load(name + '.npy', lab, 120, sigma)[:, 120]
if not show_profile:
    kernel = np.ones(window_size) / window_size
    slice = np.convolve(slice, kernel, mode='valid')
    plt.plot(slice, label=lab)
    characteristics(slice, lab)

name = '005'
lab = 'Без внешней стенки'
slice = load(name + '.npy', lab, 120, sigma)[:, 120]
if not show_profile:
    kernel = np.ones(window_size) / window_size
    slice = np.convolve(slice, kernel, mode='valid')
    plt.plot(slice, label=lab)
    characteristics(slice, lab)

plt.legend()
plt.xlabel('y, мм')
plt.ylabel('Плотность, относительные единицы')
plt.legend(loc='upper right')
plt.show()
# plt.show()
