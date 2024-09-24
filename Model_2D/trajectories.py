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


directory_path = os.getcwd() + '/data'
# directory_path = os.getcwd() + '/1_01'
pattern = 'mask*'
matching_file = sorted(glob.glob(directory_path + '/' + pattern))
mask = np.load(matching_file[0])
coorx = np.load(name_reader(directory_path, 'coorx*')[0])
coory = np.load(name_reader(directory_path, 'coory*')[0])
ln = coorx.shape[0]
x, y = mask.shape[:2]
m = 1

plt.figure(figsize=(5, 5))
for i in range(2, ln, 10):
    print(i)
    plt.plot(m * coorx[i, :], m * coory[i, :], linewidth=0.5)
    # plt.scatter(m*coorx[i, :], m*coory[i, :], s=0.9)
plt.imshow(mask)
plt.grid(color='black', linestyle='-', linewidth=0.2)

if 0:
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
