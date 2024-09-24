# ******************************************************************************
# @author: L. I. Nurtdinova
# ******************************************************************************

import datetime
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from config import *


def inverse_maxwell_distribution(x):
    m_hel = 6.6464764e-27
    k = 1.380649e-23
    T = 300
    return np.sqrt((-2 * k * T / m_hel) * np.log(1 - x)) * 1e4 / 2.5


def calc_particle_velocities(particles: list[list[float]]) -> np.array:
    part_len = len(particles)
    l2 = int((part_len - 1) * part_len // 2)
    pairs = np.zeros((l2, 9))
    n = 0
    for j in range(part_len):
        for i in range(j + 1, part_len):
            pairs[n][0] = particles[i][0]
            pairs[n][1] = particles[j][0]
            pairs[n][2] = np.sqrt((particles[i][1] - particles[j][1]) ** 2 + (particles[i][2] - particles[j][2]) ** 2)
            pairs[n][3] = particles[i][1] - particles[j][1]
            pairs[n][4] = particles[i][2] - particles[j][2]
            pairs[n][5] = particles[i][1]
            pairs[n][6] = particles[j][1]
            pairs[n][7] = particles[i][2]
            pairs[n][8] = particles[j][2]
            n += 1
    return pairs


def new_vel(n: np.array, vx: float, vy: float) -> (float, float):
    sc1 = n[0] * vx
    sc2 = n[1] * vy
    length = n[0] ** 2 + n[1] ** 2
    vx_out = vx - 2 * (sc1 * n[0] + sc2 * n[0]) / length
    vy_out = vy - 2 * (sc1 * n[1] + sc2 * n[1]) / length
    return 0.9 * vx_out, 0.9 * vy_out


def wall_mask(aperture_width: float, aperture_l2: float) -> np.array:
    x = np.arange(0, aperture_width)
    y = np.arange(0, aperture_width)
    X, Y = np.meshgrid(x, y)
    wall = np.zeros((aperture_l2 - aperture_width, aperture_width))
    # mask = + np.where(Y > l - 2 * X - 1, 4, 0) + np.where(Y > l - 0.2 * X - 1, 8, 0) + np.where(Y > 0.8 * X, 2, 0)
    # mask =+ np.where(Y > l - 1.5 * X - 1, 4, 0) + np.where(Y > l - X - 1, 8, 0) + np.where(Y > X - l, 2, 0)
    # mask = np.flip(mask, axis = 1)
    mask = + np.where(Y > aperture_width - X - 1, 4, 0) + np.where(Y > aperture_width - 1, 8, 0) + np.where(Y > X, 2, 0)

    mask[(mask == 2) | (mask == 4)] = 0
    mask[(mask == 10) | (mask == 14) | (mask == 12)] = 18
    mask[(mask == 20) | (mask == 18)] = 35
    wall = np.concatenate((wall, mask), axis=0)
    return wall


def mesh(shape_y: int, shape_x: int, height: float, m: int) -> np.array:
    """
    aperture_width - ширина диафрагмы
    aperture_l2 - расстояние между осью и нижней точкой диафрагмы
    x - координата по x левого края диафрагмы
    """
    shape_x *= m
    shape_y *= m
    height *= m
    mask = 35 * np.ones((shape_y, shape_x))
    '''
    aperture_width = 40 * m
    aperture_l2 = int(shape_y // 2 - 8 * height)
    x = 160 * m
    mask[:aperture_l2, x:x + aperture_width] = calc_wall(aperture_width, aperture_l2)
    mask[shape_y - aperture_l2:, x:x + aperture_width] = np.flipud(calc_wall(aperture_width, aperture_l2)) + 2

    aperture_width = 40 * m
    aperture_l2 = int(shape_y // 2 - 8 * height)
    x = 360 * m
    mask[:aperture_l2, x:x + aperture_width] = calc_wall(aperture_width, aperture_l2)
    mask[shape_y - aperture_l2:, x:x + aperture_width] = np.flipud(calc_wall(aperture_width, aperture_l2)) + 2
    '''
    aperture_width = 20 * m
    aperture_l2 = int(shape_y // 2 - 4 * height)
    x = 560 * m
    mask[:aperture_l2, x:x + aperture_width] = wall_mask(aperture_width, aperture_l2)
    mask[shape_y - aperture_l2:, x:x + aperture_width] = np.flipud(wall_mask(aperture_width, aperture_l2)) + 2

    mask[(mask == 10) | (mask == 12)] = 35
    return mask


def read_mask() -> np.array:
    pl = 35 * np.ones((shape_y, shape_x))
    return pl + cv2.imread('data/output_image.png')[:, :, 2]


# Считывание маски диафрагмы
# aperture_mask = read_mask()
aperture_mask = mesh(shape_y, shape_x, height, m)  # маска диафрагмы
# aperture_mask = mask2 + aperture_mask

# wall_tube
# size
# folder

if __name__ == '__main__':
    # Задание массивов координат и скоростей
    coord_y = np.ones((full_size, time_steps))
    coord_x = np.ones((full_size, time_steps))
    coord_x[:, 0] = np.random.uniform(5, thresh, full_size)
    coord_y[:, 0] = np.random.uniform(shape_y // 2 - height, shape_y // 2 + height, full_size)
    vy = np.zeros(full_size)
    vx = np.zeros(full_size)
    vx_f = np.zeros(full_size)
    velocities = []
    is_in = np.ones(full_size)
    particles = 0

    # Проход по временному циклу
    for i in range(1, time_steps):

        print(size, ' - len,', i, ' - time')
        grd = np.zeros((shape_y, shape_x, N, 3))
        grd_ch = np.zeros((shape_y, shape_x))

        # Проход по частицам
        for j in range(1, size):
            if is_in[j]:
                vel = np.sqrt(vx[j] ** 2 + vy[j] ** 2) * 1e-8
                if vel != 0 and vx_f[j] == 1:
                    velocities.append(vel)
                if coord_x[j, i - 1] > thresh and vx_f[j] == 0:
                    vx_f[j] = 1
                    v = inverse_maxwell_distribution(np.random.rand(1))
                    ang = 2 * np.pi * np.random.uniform(0, 1)
                    particles += 1
                    # print(particles)
                    vy[j] = v * np.sin(ang)
                    vx[j] = Vx + v * np.cos(ang)
                    if coord_y[j, i - 1] > shape_y // 2 + height - 0.5:
                        coord_y[j, i - 1] = shape_y // 2 + height - 0.5
                    if coord_y[j, i - 1] < shape_y // 2 - height - 0.5:
                        coord_y[j, i - 1] = shape_y // 2 - height - 0.5

                if vx_f[j] == 0:
                    vy[j] = 1
                    vx[j] = V_pot
                    if coord_y[j, i - 1] > shape_y // 2 + height - 0.5:
                        coord_y[j, i - 1] = shape_y // 2 + height - 0.5
                    if coord_y[j, i - 1] < shape_y // 2 - height - 0.5:
                        coord_y[j, i - 1] = shape_y // 2 - height - 0.5

                # Размножение частиц

                if size < full_size:
                    out = vx[j] > 0 and vx_f[j] == 1
                    if out and vx_f[j] == 1:
                        cond1 = coord_x[j, i - 1] < 159
                        cond2 = coord_x[j, i - 1] > shape_y - coord_y[j, i - 1] + 20
                        cond3 = coord_x[j, i - 1] > coord_y[j, i - 1] + 20
                        c_cond1 = cond1 and cond2 and cond3
                        cond1 = coord_x[j, i - 1] < 159 + 200
                        cond2 = coord_x[j, i - 1] > shape_y - coord_y[j, i - 1] + 20 + 200
                        cond3 = coord_x[j, i - 1] > coord_y[j, i - 1] + 20 + 200
                        c_cond2 = cond1 and cond2 and cond3
                        cond1 = coord_x[j, i - 1] < 159 + 400
                        cond2 = coord_x[j, i - 1] > shape_y - coord_y[j, i - 1] + 20 + 400
                        cond3 = coord_x[j, i - 1] > coord_y[j, i - 1] + 20 + 400
                        c_cond3 = cond1 and cond2 and cond3

                        di1_cond = di1 < (full_size - ln) * 0.2
                        di2_cond = di2 < (full_size - ln) * 0.3
                        di3_cond = di3 < (full_size - ln) * 0.5

                        if (c_cond1 and di1_cond) or (c_cond2 and di2_cond) or (c_cond3 and di3_cond):
                            if np.random.uniform(0, 1) < prob:
                                coord_x[size, :i - 2] = coord_x[j, i - 1]
                                coord_x[size, i - 2:] = coord_x[j, i - 1] + 0.5 * np.sign(
                                    np.random.uniform(-1, 1)) * np.random.uniform(0, 1)
                                coord_y[size, :i - 2] = coord_y[j, i - 1]
                                coord_y[size, i - 2:] = coord_y[j, i - 1] + 0.5 * np.sign(
                                    np.random.uniform(-1, 1)) * np.random.uniform(0, 1)
                                vx[size] = vx[j]
                                vy[size] = vy[j]
                                vx_f[size] = vx_f[j]
                                size += 1
                                if c_cond1 and di1_cond:
                                    di1 += 1
                                if c_cond2 and di2_cond:
                                    di2 += 1
                                if c_cond3 and di3_cond:
                                    di3 += 1

                calcDouble = vx[j] > 0 or j < ln
                # Добавление частиц в расчетную сетку для метода Монте-Карло
                if calcDouble:
                    a = grd[round(coord_y[j, i - 1]), round(coord_x[j, i - 1])]
                    index = int(grd_ch[round(coord_y[j, i - 1]), round(coord_x[j, i - 1])])
                    if index < N:
                        a[index][0] = j
                        a[index][1] = vx[j]
                        a[index][2] = vy[j]
                        grd_ch[round(coord_y[j, i - 1]), round(coord_x[j, i - 1])] += 1

        # Проход по сетке для расчета скоростей частиц по методу Монте-Карло
        for cell_i in range(thresh, shape_x):
            for cell_j in range(bias, shape_y - bias):
                num_part = int(grd_ch[cell_j, cell_i])
                if num_part > 0:

                    a = grd[cell_j, cell_i]
                    a = a[:num_part, :]

                    if num_part >= 2:
                        b = calc_particle_velocities(a)
                        max_element = np.max(b[:, 2])

                        for k in b:
                            if k[2] > max_element * np.random.uniform(0, 1):
                                q = k[0].astype(int)
                                w = k[1].astype(int)
                                zn = np.sign(np.random.uniform(-1, 1))
                                zn2 = np.sign(np.random.uniform(-1, 1))
                                ang1 = 2 * np.pi * np.random.uniform(0, 1)

                                vx[q] = (k[5] + k[6]) / 2 + zn * k[2] / 2 * np.cos(ang1)
                                vy[q] = (k[7] + k[8]) / 2 + zn2 * k[2] / 2 * np.sin(ang1)
                                vx[w] = (k[5] + k[6]) / 2 - zn * k[2] / 2 * np.cos(ang1)
                                vy[w] = (k[7] + k[8]) / 2 - zn2 * k[2] / 2 * np.sin(ang1)

        # Второй проход по частицам
        for j in range(1, size):
            if is_in[j]:
                a = aperture_mask[round(m * coord_y[j, i - 1]), round(m * coord_x[j, i - 1])]

                if a == 0 or a == 2:
                    vx[j] = -vx[j]
                    coord_x[j, i - 1] = coord_x[j, i - 2]
                    coord_y[j, i - 1] = coord_y[j, i - 2]

                if a == 6:
                    # nor = np.array([2, 10])
                    nor = np.array([0, 1])
                    vx[j], vy[j] = new_vel(nor, vx[j], vy[j])
                    coord_x[j, i - 1] = coord_x[j, i - 2]
                    coord_y[j, i - 1] = coord_y[j, i - 2]

                if a == 8:
                    # nor = np.array([2, -10])
                    nor = np.array([0, -1])
                    vx[j], vy[j] = new_vel(nor, vx[j], vy[j])
                    coord_x[j, i - 1] = coord_x[j, i - 2]
                    coord_y[j, i - 1] = coord_y[j, i - 2]

                if a == 98:
                    vy[j] = - vy[j]
                    coord_x[j, i - 1] = coord_x[j, i - 2]
                    coord_y[j, i - 1] = coord_y[j, i - 2]

                '''
                if a == 124 or a == 176:
                    nor = np.array([-1, 0])
                    vx[j], vy[j] = new_vel(nor, vx[j], vy[j])
                    coord_x[j, i - 1] = coord_x[j,i - 2]
                    coord_y[j, i - 1] = coord_y[j,i - 2]
    
    
                x = coord_x[j, i - 1]
                y = coord_y[j, i - 1]
                if a == 0:
    
                    nor = np.array([160 - x, 40 - y])
                    vx[j], vy[j] = new_vel(nor, vx[j], vy[j])
                    coord_x[j, i - 1] = coord_x[j,i - 2]
                    coord_y[j, i - 1] = coord_y[j,i - 2]
                    
                if a == 61:
                    nor = np.array([160 - x, 200 - y])
                    vx[j], vy[j] = new_vel(nor, vx[j], vy[j])
                    coord_x[j, i - 1] = coord_x[j,i - 2]
                    coord_y[j, i - 1] = coord_y[j,i - 2]'''

                # nor = np.array([0, -1])
                # nor = np.array([1, -1])

            vel_len = 1.5 * 1e7 / 2.5
            if vy[j] > vel_len:
                vy[j] = vel_len
            if vx[j] > vel_len:
                vx[j] = vel_len

            vel_len2 = -1.5 * 1e7 / 2.5
            if vy[j] < vel_len2:
                vy[j] = vel_len2
            if vx[j] < vel_len2:
                vx[j] = vel_len2

            coord_y[j, i] = coord_y[j, i - 1] + vy[j] * t_step
            coord_x[j, i] = coord_x[j, i - 1] + vx[j] * t_step

            if vx_f[j] == 1:

                cond1 = coord_y[j, i] > shape_y - bias
                cond2 = coord_y[j, i] < bias
                cond3 = coord_x[j, i] > shape_x - bias
                cond4 = coord_x[j, i] < x_min_lim
                cond5 = coord_x[j, i] < 159 + 400

                if wall_tube:
                    if (cond1 or cond2) and cond5:
                        vy[j] = -vy[j]
                        coord_y[j, i - 1] = coord_y[j, i - 2]

                    if (cond1 or cond2) and not cond5:
                        if cond1:
                            coord_y[j, i] = shape_y - bias
                        if cond2:
                            coord_y[j, i] = bias
                        if cond3:
                            coord_x[j, i] = shape_x - bias

                    if cond3:
                        is_in[j] = 0
                        coord_x[j, i:] = shape_x - bias
                    if cond4:
                        vx[j] = -vx[j]
                if not wall_tube:

                    if cond1:
                        coord_y[j, i] = shape_y - bias
                    if cond2:
                        coord_y[j, i] = bias
                    if cond3:
                        coord_x[j, i] = shape_x - bias
                    if cond4:
                        coord_x[j, i] = x_min_lim

    coord_x = coord_x[:size, :]
    coord_y = coord_y[:size, :]
    # Вывод траекторий, если требуется

    # Сохранение координат частиц в файл
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    params = {
        'sh': shape_x,
        'len': size,
        'time': time_steps,
        'height': height,
        'bias': bias,
        'thresh': thresh,
        'current_datetime': current_datetime
    }

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    filename_mask = 'data/mask.npy'
    filename_coord_x = 'data/coord_x.npy'
    filename_coord_y = 'data/coord_y.npy'
    filename_mask2 = out_folder + '/mask_sh={sh}_time={time}_len={len}_height={height}_bias={bias}_thresh={thresh}_{current_datetime}.npy'.format(
        **params)
    filename_coord_x2 = out_folder + '/coord_x_sh={sh}_time={time}_len={len}_height={height}_bias={bias}_thresh={thresh}_{current_datetime}.npy'.format(
        **params)
    filename_coord_y2 = out_folder + '/coord_y_sh={sh}_time={time}_len={len}_height={height}_bias={bias}_thresh={thresh}_{current_datetime}.npy'.format(
        **params)

    np.save(filename_coord_x, coord_x)
    np.save(filename_coord_y, coord_y)
    np.save(filename_mask, aperture_mask)
    np.save(filename_coord_x2, coord_x)
    np.save(filename_coord_y2, coord_y)
    np.save(filename_mask2, aperture_mask)

    if True:
        plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)
        for i in range(1, size):
            plt.plot(m * coord_x[i, :], m * coord_y[i, :], linewidth=0.5)
            # plt.scatter(coord_x[i,:], coord_y[i,:], s = 0.6)
        plt.imshow(aperture_mask)
        plt.show()

    # os.system('afplay/System/Library/Sounds/Glass.aiff')
