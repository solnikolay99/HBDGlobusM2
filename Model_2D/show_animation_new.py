# ******************************************************************************
# @author: L. I. Nurtdinova
# ******************************************************************************
import copy
import glob
import os
import pickle
import shutil

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from config import *

# plt.rcParams.update({'font.size': 5})
matplotlib.rcParams['animation.ffmpeg_path'] = "C:\\Program Files\\ffmpeg-2024-09-26-git-f43916e217-essentials_build\\bin\\ffmpeg.exe"


# Сортировка по имени названий файлов, хранящих координаты  частиц
def name_reader(dir_path, pattern):
    return sorted(glob.glob(dir_path + '/' + pattern))


def calculate_density() -> (list[float], list[list[float]]):
    with open(parts_dir_path + '/out_points.npy', 'rb') as fp:
        densities = pickle.load(fp)

    labels = [i for i in range(shape_y)]
    values = []
    out_density = [0 for _ in range(shape_y)]
    for density in densities:
        for y in density:
            out_density[round(y)] += 1
        values.append(copy.deepcopy(out_density))
    values = values[::pick_every_timeframe]
    return labels, values


def update(frame):
    coords_x = coords[frame, :, 0]
    coords_y = coords[frame, :, 1]

    values = density_values[frame]
    sum_count_points = sum(values)

    ax1.clear()
    ax1.set_title(f"{0.25 * frame * pick_every_timeframe: 6.2f} мкс")
    ax1.scatter(m * coords_x, m * coords_y, marker='.', s=0.5, color='#ff531f')

    ax2.clear()
    ax2.set_title(f"Плотность потока на выходе: {sum_count_points} частиц")
    ax2.barh(density_labels, values)

    plt.xlabel('x, ед.')
    plt.ylabel('y, мм')
    plt.xlim(0, 10)
    plt.ylim(bias - 1, shape_y - bias + 1)

    ax1.imshow(mask)
    #current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    #if saving_to_file:
    #    plt.savefig(str(main_dir_path) + '/profs/prof' + str(current_datetime) + '.png', dpi=200, bbox_inches='tight')
    #    print(f'{frame} frame was saved')
    print(f'{frame} frame')


if __name__ == '__main__':
    # Считывание массивов из файлов и их конкатенация
    main_dir_path = os.getcwd() + '/data/'
    parts_dir_path = main_dir_path + '/parts/'
    out_directory = main_dir_path + '/profs/'

    #shutil.rmtree(out_directory, ignore_errors=True)
    #os.makedirs(out_directory, exist_ok=True)

    pick_every_timeframe = 5
    pick_every_point = 1
    saving_to_file = 1
    m = 1

    mask = np.load(parts_dir_path + '/mask.npy')
    mask[mask != 255] = 0
    coords = None
    for file_name in name_reader(parts_dir_path, 'coords*'):
        temp_arr = np.load(file_name)
        temp_arr = temp_arr[::pick_every_timeframe, ::pick_every_point]
        if coords is None:
            coords = temp_arr
        else:
            coords = np.concatenate((coords, temp_arr), axis=0)

    density_labels, density_values = calculate_density()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), width_ratios=[2, 1])

    ani = FuncAnimation(fig, update, frames=range(len(coords)), interval=1)
    if saving_to_file:
        FFwriter = animation.FFMpegWriter(fps=10)
        #ani.save(main_dir_path + 'animation.mp4', writer=FFwriter)
        ani.save(main_dir_path + 'animation.gif')
    else:
        plt.show()

    shutil.rmtree(out_directory, ignore_errors=True)
