# ******************************************************************************
# @author: L. I. Nurtdinova
# ******************************************************************************

import datetime
import glob
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


# plt.rcParams.update({'font.size': 5})

# Соритирка по имени названий файлов, хранящих координаты  частиц
def name_reader(dir_path, pattern):
    return sorted(glob.glob(dir_path + '/' + pattern))


def update(frame):
    ax.clear()
    ax.scatter(m * coord_x[:, frame], m * coord_y[:, frame], marker='.', s=0.5, color='#ff531f')
    plt.title(str(0.25 * frame) + ' мкс')
    x, y = int(round(mask.shape[1], -3)), int(round(mask.shape[1], -3))
    numx, numy = 6, 11
    x_ticks = [f'{val:.0f}' if val.is_integer() else f'{val:.1f}'.rstrip('0').rstrip('.') for val in
               np.linspace(0, x * 0.25, num=numx)]
    y_ticks = [f'{val:.0f}' if val.is_integer() else f'{val:.1f}'.rstrip('0').rstrip('.') for val in
               np.linspace(0, y * 0.25, num=numy)]
    plt.xticks(np.linspace(0, x, num=numx), x_ticks)
    plt.yticks(np.linspace(0, y, num=numy), y_ticks)
    plt.xlabel('x, мм')
    plt.ylabel('y, мм')
    plt.xlim(0, mask.shape[1])
    plt.ylim(0, mask.shape[0])

    ax.imshow(mask)
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if saving:
        plt.savefig(str(directory_path) + '/profs/prof' + str(current_datetime) + '.png', dpi=600, bbox_inches='tight')
        #plt.savefig(str(directory_path) + '/profs/prof' + str(current_datetime) + '.png', dpi=100, bbox_inches='tight')
        print(f'{frame} frame was saved')


if __name__ == '__main__':
    # Считывание массивов из файлов и их конкатенация
    directory_path = os.getcwd() + '/data/'
    out_directory = os.getcwd() + '/data/profs/'

    shutil.rmtree(out_directory, ignore_errors=True)
    os.makedirs(out_directory, exist_ok=True)

    # directory_path = os.getcwd() + '/008'
    mask = np.load(sorted(glob.glob(directory_path + '/' + 'mask*'))[0])
    coord_x = np.load(name_reader(directory_path, 'coord_x*')[0])
    coord_y = np.load(name_reader(directory_path, 'coord_y*')[0])
    r = 1
    coord_x, coord_y = coord_x[::r, :], coord_y[::r, :]
    fig, ax = plt.subplots(figsize=(10, 6))
    m = 1

    saving = 1

    ani = FuncAnimation(fig, update, frames=range(0, len(coord_x[0]), 5), interval=1)
    if saving:
        #ani.save('animation.mp4', writer='ffmpeg')
        ani.save(directory_path + 'animation.gif')
    else:
        plt.show()

    shutil.rmtree(out_directory, ignore_errors=True)

    if not saving:
        x = int(round(mask.shape[1], -3))
        y = int(round(mask.shape[0], -3))
        numx = 11
        numy = 5
        x_ticks = [f'{val:.0f}' if val.is_integer() else f'{val:.1f}'.rstrip('0').rstrip('.') for val in
                   np.linspace(0, x * 0.025, num=numx)]
        y_ticks = [f'{val:.0f}' if val.is_integer() else f'{val:.1f}'.rstrip('0').rstrip('.') for val in
                   np.linspace(0, y * 0.025, num=numy)]
        plt.xticks(np.linspace(0, x, num=numx), x_ticks)
        plt.yticks(np.linspace(0, y, num=numy), y_ticks)
        # plt.title('Воспроизведенная геометрия')
        plt.xlabel('x, мм')
        plt.ylabel('y, мм')
        plt.xlim(0, mask.shape[1])
        plt.ylim(0, mask.shape[0])
        plt.grid(color='black', linestyle='-', linewidth=0.2)
        plt.show()
