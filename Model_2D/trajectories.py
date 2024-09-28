# ******************************************************************************
# @author: L. I. Nurtdinova
# ******************************************************************************

import glob
import os

import matplotlib.pyplot as plt
import numpy as np


# plt.rcParams.update({'font.size': 16})  # Установим глобальный размер шрифта
# Соритирка по имени названий файлов, хранящих координаты  частиц
def name_reader(dir_path, pattern):
    return sorted(glob.glob(dir_path + '/' + pattern))


if __name__ == '__main__':
    directory_path = os.getcwd() + '/data'
    # directory_path = os.getcwd() + '/1_01'
    pattern = 'mask*'
    matching_file = sorted(glob.glob(directory_path + '/' + pattern))
    mask = np.load(matching_file[0])
    coord_x = np.load(name_reader(directory_path, 'coord_x*')[0])
    coord_y = np.load(name_reader(directory_path, 'coord_y*')[0])
    ln = coord_x.shape[0]
    x, y = mask.shape[:2]
    m = 1

    plt.figure(figsize=(5, 5))
    #for i in range(2, ln, 10):
    for i in range(0, ln):
        print(i)
        plt.plot(m * coord_x[i, :], m * coord_y[i, :], linewidth=0.5)
        # plt.scatter(m*coord_x[i, :], m*coord_y[i, :], s=0.9)
    plt.imshow(mask)
    plt.grid(color='black', linestyle='-', linewidth=0.2)

    if True:
        x = int(round(mask.shape[1], -3))
        y = int(round(mask.shape[0], -3))
        numx = 11
        numy = 5
        x_ticks = [f'{val:.0f}' if val.is_integer() else f'{val:.1f}'.rstrip('0').rstrip('.') for val in
                   np.linspace(0, x * 0.25, num=numx)]
        y_ticks = [f'{val:.0f}' if val.is_integer() else f'{val:.1f}'.rstrip('0').rstrip('.') for val in
                   np.linspace(0, y * 0.25, num=numy)]
        plt.xticks(np.linspace(0, x, num=numx), x_ticks)
        plt.yticks(np.linspace(0, y, num=numy), y_ticks)
        plt.xlabel('x, мм')
        plt.ylabel('y, мм')

    plt.show()
