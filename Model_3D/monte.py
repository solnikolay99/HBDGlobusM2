# ******************************************************************************
# @author: L. I. Nurtdinova
# ******************************************************************************

import datetime
import os

import numpy as np


def inverse_maxwell_distribution(x):
    mHel = 6.6464764e-27
    k = 1.380649e-23
    T = 300
    return np.sqrt((-2 * k * T / mHel) * np.log(1 - x)) * 1e3


def prod(particles):
    l = len(particles)
    l2 = int((l - 1) * l // 2)
    pairs = np.zeros((l2, 12))
    n = 0
    for j in range(l):
        for i in range(j + 1, l):
            pairs[n][0] = particles[i][0]
            pairs[n][1] = particles[j][0]
            pairs[n][2] = np.sqrt(
                (particles[i][1] - particles[j][1]) ** 2 + (particles[i][2] - particles[j][2]) ** 2 + (
                            particles[i][3] - particles[j][3]) ** 2)
            pairs[n][3] = particles[i][1] - particles[j][1]
            pairs[n][4] = particles[i][2] - particles[j][2]
            pairs[n][5] = particles[i][3] - particles[j][3]
            pairs[n][6] = particles[i][1]
            pairs[n][7] = particles[j][1]
            pairs[n][8] = particles[i][2]
            pairs[n][9] = particles[j][2]
            pairs[n][10] = particles[i][3]
            pairs[n][11] = particles[j][3]
            n += 1
    return pairs


def new_vel(n, vx, vy, vz):
    sc1 = n[0] * vx
    sc2 = n[1] * vy
    sc3 = n[2] * vz
    len = n[0] ** 2 + n[1] ** 2 + n[2] ** 2
    vx_out = vx - 2 * (sc1 * n[0] + sc2 * n[0] + sc3 * n[0]) / len
    vy_out = vy - 2 * (sc1 * n[1] + sc2 * n[1] + sc3 * n[1]) / len
    vz_out = vz - 2 * (sc1 * n[2] + sc2 * n[2] + sc3 * n[2]) / len
    return vx_out, vy_out, vz_out


shape_x = 150
shape_y, shape_z = 50, 50
rad = shape_y // 2
dell = 1

time = int(375 * dell)
size = int(500e3)
fullsize = int(750e3)
folder = '001'

ver = 0.75
N = 15
height = 0.25
diadim = 1  # для расчетов
Vx = 1e6
Vy = height
thresh = 20
x_min = 10
m = 1
bias = 1
iter_number = 1
t_step = 1e-6 / dell
V_pot = 0.01 * 1e6
V_pot = 1e6
particles = 0
ln = size
di1, di2, di3 = 0, 0, 0

coory = np.ones((fullsize, time))
coorx = np.ones((fullsize, time))
coorz = np.ones((fullsize, time))
coorx[:, 0] = np.random.uniform(5, thresh, fullsize)
coory[:, 0] = np.random.uniform(shape_y // 2 - height, shape_y // 2 + height, fullsize)
coorz[:, 0] = np.random.uniform(shape_y // 2 - height, shape_y // 2 + height, fullsize)
vy = np.zeros(fullsize)
vx = np.zeros(fullsize)
vz = np.zeros(fullsize)
vx_f = np.zeros(fullsize)
is_in = np.ones(fullsize)
vel_save = []

for i in range(1, time):

    if not i % dell:
        print(size, ' - len,', i // dell, ' - time')

    grd = np.zeros((shape_y, shape_z, shape_x, N, 4))
    grd_ch = np.zeros((shape_y, shape_z, shape_x))

    for j in range(1, size):

        if is_in[j]:
            vel = np.sqrt(vx[j] ** 2 + vy[j] ** 2)
            if vel != 0 and vx_f[j] == 1: vel_save.append(vel)

            if coorx[j, i - 1] > thresh and vx_f[j] == 0:
                vx_f[j] = 1
                v = inverse_maxwell_distribution(np.random.rand(1))
                fi = 2 * np.pi * np.random.uniform(0, 1)
                theta = np.pi * np.random.uniform(0, 1)
                particles += 1
                # print(particles)
                vy[j] = v * np.sin(theta) * np.sin(fi)
                vz[j] = v * np.sin(theta) * np.cos(fi)
                vx[j] = Vx + v * np.cos(theta)

                ry = coory[j, i - 1] - rad
                rz = coorz[j, i - 1] - rad
                r = np.sqrt(ry ** 2 + rz ** 2)
                ang = 2 * np.pi * np.random.uniform(0, 1)
                if r > height:
                    coorz[j, i - 1] = height * np.cos(ang) + rad
                    coory[j, i - 1] = height * np.sin(ang) + rad

            if vx_f[j] == 0:
                vy[j] = 1
                vz[j] = 1
                vx[j] = V_pot
                ry = coory[j, i - 1] - rad
                rz = coorz[j, i - 1] - rad
                r = np.sqrt(ry ** 2 + rz ** 2)
                ang = 2 * np.pi * np.random.uniform(0, 1)
                if r > height:
                    coorz[j, i - 1] = height * np.sin(ang) + rad
                    coory[j, i - 1] = height * np.cos(ang) + rad

            if not i % dell:
                if size < fullsize:

                    out = vx[j] > 0 and vx_f[j] == 1
                    if out and vx_f[j] == 1:
                        cond1 = coorx[j, i - 1] < 39
                        cond2 = coorx[j, i - 1] > shape_y - coory[j, i - 1] + 12
                        cond3 = coorx[j, i - 1] > coory[j, i - 1] + 12
                        cond4 = coorx[j, i - 1] > shape_z - coorz[j, i - 1] + 12
                        cond5 = coorx[j, i - 1] > coorz[j, i - 1] + 12
                        cond_all_1 = cond1 and cond2 and cond3 and cond4 and cond5

                        cond1 = coorx[j, i - 1] < 0
                        cond2 = coorx[j, i - 1] > shape_y - coory[j, i - 1] + 40
                        cond3 = coorx[j, i - 1] > coory[j, i - 1] + 40
                        cond4 = coorx[j, i - 1] > shape_z - coorz[j, i - 1] + 40
                        cond5 = coorx[j, i - 1] > coorz[j, i - 1] + 40
                        cond_all_2 = cond1 and cond2 and cond3 and cond4 and cond5
                        di1_cond = di1 < (fullsize - ln) * 0.3
                        di2_cond = di2 < (fullsize - ln) * 0.7

                        if (cond_all_1 and di1_cond) or (cond_all_2 and di2_cond):
                            if np.random.uniform(0, 1) < ver:
                                coorx[size, :i - 2] = coorx[j, i - 1]
                                coorx[size, i - 2:] = coorx[j, i - 1] + 0.5 * np.sign(
                                    np.random.uniform(-1, 1)) * np.random.uniform(0, 1)
                                coory[size, :i - 2] = coory[j, i - 1]
                                coory[size, i - 2:] = coory[j, i - 1] + 0.5 * np.sign(
                                    np.random.uniform(-1, 1)) * np.random.uniform(0, 1)
                                coorz[size, :i - 2] = coorz[j, i - 1]
                                coorz[size, i - 2:] = coorz[j, i - 1] + 0.5 * np.sign(
                                    np.random.uniform(-1, 1)) * np.random.uniform(0, 1)
                                vx[size] = vx[j]
                                vy[size] = vy[j]
                                vz[size] = vz[j]
                                vx_f[size] = vx_f[j]
                                is_in[size] = is_in[j]
                                size += 1
                                if (cond_all_1 and di1_cond): di1 += 1
                                if (cond_all_2 and di2_cond): di2 += 1

            if not i % dell:
                calcDouble = vx[j] > 0 or j < ln
                if calcDouble:
                    a = grd[round(coory[j, i - 1]), round(coorz[j, i - 1]), round(coorx[j, i - 1])]
                    index = int(grd_ch[round(coory[j, i - 1]), round(coorz[j, i - 1]), round(coorx[j, i - 1])])
                    if index < N:
                        a[index][0] = j
                        a[index][1] = vx[j]
                        a[index][2] = vy[j]
                        a[index][3] = vz[j]
                        grd_ch[round(coory[j, i - 1]), round(coorz[j, i - 1]), round(coorx[j, i - 1])] += 1

    edge = 50
    if not i % dell:
        for cell_i in range(thresh, shape_x):
            for cell_j in range(bias, shape_y - bias):
                for cell_k in range(bias, shape_y - bias):
                    num_part = int(grd_ch[cell_j, cell_j, cell_i])
                    if num_part > 0:
                        a = grd[cell_j, cell_j, cell_i]
                        a = a[:num_part, :]
                        if num_part >= 2:
                            b = prod(a)
                            max_element = np.max(b[:, 2])
                            for k in b:
                                if k[2] > max_element * np.random.uniform(0, 1):
                                    q = k[0].astype(int)
                                    w = k[1].astype(int)
                                    zn = np.sign(np.random.uniform(-1, 1))
                                    zn2 = np.sign(np.random.uniform(-1, 1))
                                    zn3 = np.sign(np.random.uniform(-1, 1))
                                    fi = 2 * np.pi * np.random.uniform(0, 1)
                                    teta = np.pi * np.random.uniform(0, 1)

                                    vx[q] = (k[6] + k[7]) / 2 + zn * k[2] / 2 * np.cos(teta)
                                    vy[q] = (k[8] + k[9]) / 2 + zn2 * k[2] / 2 * np.sin(teta) * np.sin(fi)
                                    vz[q] = (k[10] + k[11]) / 2 + zn3 * k[2] / 2 * np.sin(teta) * np.cos(fi)
                                    vx[w] = (k[6] + k[7]) / 2 - zn * k[2] / 2 * np.cos(teta)
                                    vy[w] = (k[8] + k[9]) / 2 - zn2 * k[2] / 2 * np.sin(teta) * np.sin(fi)
                                    vz[w] = (k[10] + k[11]) / 2 - zn3 * k[2] / 2 * np.sin(teta) * np.cos(fi)

    for j in range(1, size):
        if is_in[j]:

            if vx_f[j] == 1:
                ry = coory[j, i - 1] - rad
                rz = coorz[j, i - 1] - rad
                radtube = 15
                dia_1_cond = 40 < coorx[j, i - 1] < 45
                dia_2_cond = 70 < coorx[j, i - 1] < 75
                # dia_2_cond = coorx[j, i-1] < 0
                if dia_1_cond: di_cor = 40
                if dia_2_cond: di_cor = 75

                firstWall_cond = coorx[j, i - 1] < 5
                r = np.sqrt(ry ** 2 + rz ** 2)
                big_tube_cond = r > 23 and coorx[j, i - 1] < 80
                tube_cond = r < radtube

                if (dia_1_cond or dia_2_cond) and tube_cond:
                    if r > diadim:
                        if r - diadim < 0.5 * (coorx[j, i - 1] - di_cor) and r - diadim < 2.5 - 0.5 * (
                                coorx[j, i - 1] - di_cor):
                            nor = np.array([0, ry, rz])
                            vx[j], vy[j], vz[j] = new_vel(nor, vx[j], vy[j], vz[j])
                        else:
                            vx[j] = -vx[j]
                        coorz[j, i - 1] = coorz[j, i - 2]
                        coory[j, i - 1] = coory[j, i - 2]
                        coorx[j, i - 1] = coorx[j, i - 2]

                if firstWall_cond and tube_cond:
                    vx[j] = -vx[j]
                    coorz[j, i - 1] = coorz[j, i - 2]
                    coory[j, i - 1] = coory[j, i - 2]
                    coorx[j, i - 1] = coorx[j, i - 2]

                if big_tube_cond:
                    nor = np.array([0, ry, rz])
                    vx[j], vy[j], vz[j] = new_vel(nor, vx[j], vy[j], vz[j])
                    coorz[j, i - 1] = coorz[j, i - 2]
                    coory[j, i - 1] = coory[j, i - 2]
                    coorx[j, i - 1] = coorx[j, i - 2]

                if x_min < coorx[j, i - 1] < 85:
                    holcr_1 = 30
                    holcr_2 = 60
                    ring = 2
                    hole_rad = 5
                    rx = coorx[j, i - 1] - holcr_1
                    rx2 = coorx[j, i - 1] - holcr_2
                    y_condition = radtube + ring > r > radtube

                    if 1:

                        cir1 = np.sqrt(ry ** 2 + rx ** 2 + (rz + radtube) ** 2)
                        cir2 = np.sqrt((ry + radtube) ** 2 + rx ** 2 + rz ** 2)
                        cir3 = np.sqrt(ry ** 2 + rx ** 2 + (rz - radtube) ** 2)
                        cir4 = np.sqrt((ry - radtube) ** 2 + rx ** 2 + rz ** 2)
                        cir5 = np.sqrt(ry ** 2 + rx2 ** 2 + (rz + radtube) ** 2)
                        cir6 = np.sqrt((ry + radtube) ** 2 + rx2 ** 2 + rz ** 2)
                        cir7 = np.sqrt(ry ** 2 + rx2 ** 2 + (rz - radtube) ** 2)
                        cir8 = np.sqrt((ry - radtube) ** 2 + rx2 ** 2 + rz ** 2)

                        cir1_cond1 = cir1 < hole_rad + ring
                        cir2_cond1 = cir2 < hole_rad + ring
                        cir3_cond1 = cir3 < hole_rad + ring
                        cir4_cond1 = cir4 < hole_rad + ring
                        cir5_cond1 = cir5 < hole_rad + ring
                        cir6_cond1 = cir6 < hole_rad + ring
                        cir7_cond1 = cir7 < hole_rad + ring
                        cir8_cond1 = cir8 < hole_rad + ring

                        cir1_cond2 = hole_rad < cir1 < hole_rad + ring
                        cir2_cond2 = hole_rad < cir2 < hole_rad + ring
                        cir3_cond2 = hole_rad < cir3 < hole_rad + ring
                        cir4_cond2 = hole_rad < cir4 < hole_rad + ring
                        cir5_cond2 = hole_rad < cir5 < hole_rad + ring
                        cir6_cond2 = hole_rad < cir6 < hole_rad + ring
                        cir7_cond2 = hole_rad < cir7 < hole_rad + ring
                        cir8_cond2 = hole_rad < cir8 < hole_rad + ring

                        if 1:
                            cir_cond1 = (
                                        cir1_cond1 or cir2_cond1 or cir3_cond1 or cir4_cond1 or cir5_cond1 or cir6_cond1 or cir7_cond1 or cir8_cond1)
                            cir_cond2 = (
                                        cir1_cond2 or cir2_cond2 or cir3_cond2 or cir4_cond2 or cir5_cond2 or cir6_cond2 or cir7_cond2 or cir8_cond2)

                        if 0:
                            cir_cond1 = (cir1_cond1 or cir2_cond1 or cir3_cond1 or cir4_cond1)
                            cir_cond2 = (cir1_cond2 or cir2_cond2 or cir3_cond2 or cir4_cond2)

                        if 0:
                            cir_cond1 = (cir5_cond1 or cir6_cond1 or cir7_cond1 or cir8_cond1)
                            cir_cond2 = (cir5_cond2 or cir6_cond2 or cir7_cond2 or cir8_cond2)

                        if y_condition and not cir_cond1:
                            nor = np.array([0, ry, rz])
                            vx[j], vy[j], vz[j] = new_vel(nor, vx[j], vy[j], vz[j])
                            coorz[j, i - 1] = coorz[j, i - 2]
                            coory[j, i - 1] = coory[j, i - 2]
                            coorx[j, i - 1] = coorx[j, i - 2]

                        if y_condition and cir_cond2:

                            if 1:
                                if (cir1_cond2 or cir3_cond2 or cir5_cond2 or cir7_cond2):
                                    nor = np.array([coorx[j, i - 1] - hole_rad, coory[j, i - 1] - hole_rad, 0])
                                if (cir2_cond2 or cir4_cond2 or cir6_cond2 or cir8_cond2):
                                    nor = np.array([coorx[j, i - 1] - hole_rad, 0, coorz[j, i - 1] - hole_rad])

                            if 0:
                                if (cir1_cond2 or cir3_cond2):
                                    nor = np.array([coorx[j, i - 1] - hole_rad, coory[j, i - 1] - hole_rad, 0])
                                if (cir2_cond2 or cir4_cond2):
                                    nor = np.array([coorx[j, i - 1] - hole_rad, 0, coorz[j, i - 1] - hole_rad])

                            if 0:
                                if (cir5_cond2 or cir7_cond2):
                                    nor = np.array([coorx[j, i - 1] - hole_rad, coory[j, i - 1] - hole_rad, 0])
                                if (cir6_cond2 or cir8_cond2):
                                    nor = np.array([coorx[j, i - 1] - hole_rad, 0, coorz[j, i - 1] - hole_rad])

                            vx[j], vy[j], vz[j] = new_vel(nor, vx[j], vy[j], vz[j])
                            coorz[j, i - 1] = coorz[j, i - 2]
                            coory[j, i - 1] = coory[j, i - 2]
                            coorx[j, i - 1] = coorx[j, i - 2]

            vel_len = 1.5e6
            if vy[j] > vel_len: vy[j] = vel_len
            if vx[j] > vel_len: vx[j] = vel_len
            if vz[j] > vel_len: vz[j] = vel_len

            vel_len2 = -1.5e6
            if vy[j] < vel_len2: vy[j] = vel_len2
            if vx[j] < vel_len2: vx[j] = vel_len2
            if vz[j] < vel_len2: vz[j] = vel_len2

            coory[j, i] = coory[j, i - 1] + vy[j] * t_step
            coorx[j, i] = coorx[j, i - 1] + vx[j] * t_step
            coorz[j, i] = coorz[j, i - 1] + vz[j] * t_step

            if vx_f[j] == 1:
                cond1 = coory[j, i] > shape_y - bias or coory[j, i] < bias
                cond2 = coorz[j, i] > shape_z - bias or coorz[j, i] < bias
                cond3 = coorx[j, i] > shape_x - bias or coorx[j, i] < x_min
                if cond1 or cond2 or cond3:
                    is_in[j] = 0
                    if coorx[j, i] > shape_x - bias: coorx[j, i:] = shape_x - bias
                    if coorx[j, i] < x_min: coorx[j, i:] = x_min
                    if coorz[j, i] > shape_z - bias: coorz[j, i:] = shape_z - bias
                    if coorz[j, i] < bias: coorz[j, i:] = bias
                    if coory[j, i] > shape_y - bias: coory[j, i:] = shape_y - bias
                    if coory[j, i] < bias: coory[j, i:] = bias

os.system('afplay /System/Library/Sounds/Glass.aiff')

coorx, coory, coorz = coorx[:, ::dell], coory[:, ::dell], coorz[:, ::dell]
coorx, coory, coorz = coorx.astype(np.float16), coory.astype(np.float16), coorz.astype(np.float16)

cur_dt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
params = {'len': size, 'time': time, 'cur_dt': cur_dt}

if not os.path.exists(folder): os.makedirs(folder)

filename_mask = 'data/mask.npy'
filename_coorx = 'data/coorx.npy'
filename_coory = 'data/coory.npy'
filename_coorz = 'data/coorz.npy'
filename_mask2 = folder + '/mask_time={time}_len={len}_{cur_dt}.npy'.format(**params)
filename_coorx2 = folder + '/coorx_time={time}_len={len}_{cur_dt}.npy'.format(**params)
filename_coory2 = folder + '/coory_time={time}_len={len}_{cur_dt}.npy'.format(**params)
filename_coorz2 = folder + '/coorz_time={time}_len={len}_{cur_dt}.npy'.format(**params)

np.save('data/vel.npy', vel_save)
np.save(filename_coorx, coorx)
np.save(filename_coory, coory)
np.save(filename_coorz, coorz)
np.save(filename_coorx2, coorx)
np.save(filename_coory2, coory)
np.save(filename_coorz2, coorz)

if 0:
    plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)
    for i in range(1, size):
        plt.plot(m * coorx[i, :], m * coory[i, :], linewidth=0.5)
        # plt.scatter(coorx[i,:], coory[i,:], s = 0.6)
    plt.show()
