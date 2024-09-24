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
plt.rcParams.update({'font.size': 12})


def load(filename, title, section, sigma):
    den = np.round(np.load('data/' + filename))
    den = den[42:den.shape[0] - 42, :den.shape[1] - 120]
    global x
    global y
    x = int(round(den.shape[1], - 2))
    y = int(round(den.shape[0], - 2))

    if show_profile:
        plt.figure(figsize=(10, 5))
        den_show = copy.deepcopy(den)
        den_show[:, section] = 42
        plt.title(title)
        plt.imshow(1 + den_show, norm=LogNorm(), cmap='plasma')
        numx, numy = 11, 5
        x_ticks = [f'{val:.0f}' if val.is_integer() else f'{val:.1f}'.rstrip('0').rstrip('.') for val in
                   np.linspace(0, x * 0.25, num=numx)]
        y_ticks = [f'{val:.0f}' if val.is_integer() else f'{val:.1f}'.rstrip('0').rstrip('.') for val in
                   np.linspace(0, y * 0.25, num=numy)]
        plt.title(title)
        if 1:
            plt.xticks(np.linspace(0, x, num=numx), x_ticks)
            plt.yticks(np.linspace(0, y, num=numy), y_ticks)
        plt.xlabel('x, мм')
        plt.ylabel('y, мм')
        plt.show()
    # den = gaussian_filter(den, sigma=sigma)
    return den


def characteristics(slice, lab):
    square = np.trapz(slice)
    arg_semi = np.argmin(np.abs(slice - np.max(slice) / 2))
    arg_full = np.argmin(np.abs(slice - np.max(slice)))
    semi_wid = np.abs(arg_full - arg_semi)
    print(semi_wid, square)
    new_row = [lab, semi_wid, square]
    # data.loc[len(data)] = new_row


# data = pd.read_excel("Results.xlsx", index_col=0)
show_profile = 1
sigma = 0
window_size = 25

# characteristics(slice, lab)


name = '008'
lab = '2D'
slice = load(name + '.npy', lab, 480, sigma)[:, 480]
kernel = np.ones(window_size) / window_size
slice = np.convolve(slice, kernel, mode='valid')
plt.plot(slice, label=lab, color='orange')
plt.legend()

# data.to_excel('Results.xlsx')
# Output in the required axes

if 0:
    numx = 6
    x_ticks = [f'{val:.0f}' if val.is_integer() else f'{val:.1f}'.rstrip('0').rstrip('.') for val in
               np.linspace(0, y * 0.25, num=numx)]
    plt.xticks(np.linspace(0, y, num=numx), x_ticks)
    plt.xlabel('y, мм')
    plt.ylabel('Плотность')
    plt.grid(color='black', linestyle='-', linewidth=0.2)
    plt.legend()

plt.show()
