# ******************************************************************************
# @author: L. I. Nurtdinova
# ******************************************************************************

import copy
import datetime
import os
import time
import pickle
import shutil

import cv2
import matplotlib.pyplot as plt

from config import *
from monteclass import *

# pre-defined randoms
randoms_1 = Randoms()  # for randoms from [0, 1)
randoms_2pi = Randoms()  # for randoms from [0, 2*pi)
randoms_mp1 = Randoms()  # for randoms from [-1, 1)

timers = {}


def start_timer(timer_name: str):
    timers[timer_name] = time.time()
    pass


def release_timer(timer_name: str):
    print(f"Execution time for '{timer_name}' is {time.time() - timers[timer_name]:.4f}")
    pass


def initiate_randoms():
    global randoms_1, randoms_2pi, randoms_mp1
    randoms_1 = Randoms(0, 1, full_size * 10)  # for randoms from [0, 1)
    randoms_2pi = Randoms(0, 2 * np.pi, full_size * 10)  # for randoms from [0, 2*pi)
    randoms_mp1 = Randoms(-1, 1, full_size * 10)  # for randoms from [-1, 1)
    pass


def inverse_maxwell_distribution(x):
    return np.sqrt(k_t_m_hel * np.log(1 - x)) * denominator


def calc_point_velocities(points: list[Point]) -> list[list]:
    part_len = len(points)
    pairs = []
    for j in range(part_len):
        for i in range(j + 1, part_len):
            pairs.append([np.sqrt((points[i].v_x - points[j].v_x) * (points[i].v_x - points[j].v_x)
                                  + (points[i].v_y - points[j].v_y) * (
                                          points[i].v_y - points[j].v_y)),
                          points[i],
                          points[j]])
    return pairs


def new_velocity(coeff_x: int, coeff_y: int, vx: float, vy: float) -> (float, float):
    sc1 = coeff_x * vx
    sc2 = coeff_y * vy
    length_inv = 1 / (coeff_x * coeff_x + coeff_y * coeff_y)
    vx_out = vx - 2 * (sc1 * coeff_x + sc2 * coeff_x) * length_inv
    vy_out = vy - 2 * (sc1 * coeff_y + sc2 * coeff_y) * length_inv
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


def mesh(max_y: int, max_x: int, height: float, m: int) -> np.array:
    """
    aperture_width - ширина диафрагмы
    aperture_l2 - расстояние между осью и нижней точкой диафрагмы
    x - координата по x левого края диафрагмы
    """
    max_x *= m
    max_y *= m
    height *= m
    mask = 35 * np.ones((max_y, max_x))

    aperture_width = 20 * m
    aperture_l2 = int(max_y // 2 - 8 * height)
    x = 160 * m
    mask[:aperture_l2, x:x + aperture_width] = wall_mask(aperture_width, aperture_l2)
    mask[max_y - aperture_l2:, x:x + aperture_width] = np.flipud(wall_mask(aperture_width, aperture_l2)) + 2

    aperture_width = 20 * m
    aperture_l2 = int(max_y // 2 - 8 * height)
    x = 360 * m
    mask[:aperture_l2, x:x + aperture_width] = wall_mask(aperture_width, aperture_l2)
    mask[max_y - aperture_l2:, x:x + aperture_width] = np.flipud(wall_mask(aperture_width, aperture_l2)) + 2

    aperture_width = 20 * m
    aperture_l2 = int(max_y // 2 - 4 * height)
    x = 560 * m
    mask[:aperture_l2, x:x + aperture_width] = wall_mask(aperture_width, aperture_l2)
    mask[max_y - aperture_l2:, x:x + aperture_width] = np.flipud(wall_mask(aperture_width, aperture_l2)) + 2

    mask[(mask == 10) | (mask == 12)] = 35
    return mask


def read_mask() -> np.array:
    pl = 35 * np.ones((shape_y, shape_x))
    return pl + cv2.imread('data/output_image.png')[:, :, 2]


def initiate_points(count_points: int, max_length: int, y_size: int, half_height: float) -> list[Point]:
    coords_x = np.random.uniform(5, max_length, count_points)
    coords_y = np.random.uniform(y_size // 2 - half_height, y_size // 2 + half_height, count_points)
    return [Point(x=coords_x[i],
                  y=coords_y[i],
                  # z=0,
                  v_x=0,
                  v_y=0,
                  v_z=0,
                  is_in=True) for i in range(count_points)]


def check_capillary(point: Point) -> Point:
    if point.y > shape_y_top:
        point.y = shape_y_top
    elif point.y < shape_y_bottom:
        point.y = shape_y_bottom
    return point


def split_to_cells(points: list[Point]) -> dict[str, list[Point]]:
    grid_dict = {}
    for j in range(len(points)):
        if points[j].is_in:
            if points[j].x > capillary_length and vx_f[j] == 0:
                vx_f[j] = 1
                v = inverse_maxwell_distribution(randoms_1.get_next())
                angel = randoms_2pi.get_next()
                points[j].v_x = Vx + v * np.cos(angel)
                points[j].v_y = v * np.sin(angel)
                points[j] = check_capillary(points[j])
            elif vx_f[j] == 0:
                points[j].v_x = V_pot
                points[j].v_y = 1
                points[j] = check_capillary(points[j])

            # Размножение частиц
            '''
            if len(points) < full_size:
                if points[j].x > 0 and vx_f[j] == 1:
                    cond1 = points[j].x < 159
                    cond2 = points[j].x > shape_y - points[j].y + 20
                    cond3 = points[j].x > points[j].y + 20
                    c_cond1 = cond1 and cond2 and cond3
                    cond1 = points[j].x < 159 + 200
                    cond2 = points[j].x > shape_y - points[j].y + 20 + 200
                    cond3 = points[j].x > points[j].y + 20 + 200
                    c_cond2 = cond1 and cond2 and cond3
                    cond1 = points[j].x < 159 + 400
                    cond2 = points[j].x > shape_y - points[j].y + 20 + 400
                    cond3 = points[j].x > points[j].y + 20 + 400
                    c_cond3 = cond1 and cond2 and cond3

                    di1_cond = di[0] < (full_size - ln) * 0.2
                    di2_cond = di[1] < (full_size - ln) * 0.3
                    di3_cond = di[2] < (full_size - ln) * 0.5

                    if (c_cond1 and di1_cond) or (c_cond2 and di2_cond) or (c_cond3 and di3_cond):
                        if randoms_1.get_next() < prob:
                            coord_x[size, :i - 2] = points[j].x
                            coord_x[size, i - 2:] = points[j].x + 0.5 * np.sign(
                                randoms_mp1.get_next()) * randoms_1.get_next()
                            coord_y[size, :i - 2] = points[j].y
                            coord_y[size, i - 2:] = points[j].y + 0.5 * np.sign(
                                randoms_mp1.get_next()) * randoms_1.get_next()
                            vx[size] = vx[j]
                            vy[size] = vy[j]
                            vx_f[size] = vx_f[j]
                            size += 1
                            if c_cond1 and di1_cond:
                                di[0] += 1
                            if c_cond2 and di2_cond:
                                di[1] += 1
                            if c_cond3 and di3_cond:
                                di[2] += 1
            '''

            # Добавление частиц в расчетную сетку для метода Монте-Карло
            if points[j].x > 0 or j < ln:
                key = f"{round(points[j].y)}_{round(points[j].x)}"
                if key not in grid_dict:
                    grid_dict[key] = [points[j]]
                else:
                    if len(grid_dict[key]) < max_points_per_cell:
                        grid_dict[key].append(points[j])
    return grid_dict


def iterate_over_grid(grid: dict[str, list[Point]]):
    for cell_points in grid.values():
        if len(cell_points) >= 2:
            point_pairs = calc_point_velocities(cell_points)
            velocity_squares = [point_pair[0] for point_pair in point_pairs]
            max_element = np.max(velocity_squares)

            for point_pair in point_pairs:
                if point_pair[0] > max_element * randoms_1.get_next():
                    zn1 = np.sign(randoms_mp1.get_next())
                    zn2 = np.sign(randoms_mp1.get_next())
                    angle = 2 * np.pi * randoms_1.get_next()

                    point_pair[1].v_x = \
                        (point_pair[1].v_x + point_pair[2].v_x) * 0.5 + zn1 * point_pair[0] * 0.5 * np.cos(angle)
                    point_pair[1].v_y = \
                        (point_pair[1].v_y + point_pair[2].v_y) * 0.5 + zn2 * point_pair[0] * 0.5 * np.sin(angle)
                    point_pair[2].v_x = \
                        (point_pair[1].v_x + point_pair[2].v_x) * 0.5 - zn1 * point_pair[0] * 0.5 * np.cos(angle)
                    point_pair[2].v_y = \
                        (point_pair[1].v_y + point_pair[2].v_y) * 0.5 - zn2 * point_pair[0] * 0.5 * np.sin(angle)
    pass


def checking_boundaries(points: list[Point]):
    for j in range(1, size):
        if points[j].is_in:
            a = aperture_mask[round(m * points[j].y), round(m * points[j].x)]

            if a == 0 or a == 2:
                points[j].v_x = -points[j].v_x
                points[j].x = time_frames[t - 1][j].x
                points[j].y = time_frames[t - 1][j].y

            elif a == 6:
                points[j].v_x, points[j].v_y = new_velocity(0, 1, points[j].v_x, points[j].v_y)
                points[j].x = time_frames[t - 1][j].x
                points[j].y = time_frames[t - 1][j].y

            elif a == 8:
                points[j].v_x, points[j].v_y = new_velocity(0, -1, points[j].v_x, points[j].v_y)
                points[j].x = time_frames[t - 1][j].x
                points[j].y = time_frames[t - 1][j].y

            elif a == 98:
                points[j].v_y = -points[j].v_y
                points[j].x = time_frames[t - 1][j].x
                points[j].y = time_frames[t - 1][j].y

            '''
            elif a == 124 or a == 176:
                points[j].v_x, points[j].v_y = new_velocity(-1, 0, points[j].v_x, points[j].v_y)
                points[j].x = time_frames[t - 1][j].x
                points[j].y = time_frames[t - 1][j].y

            if a == 0:
                points[j].v_x, points[j].v_y = new_velocity(160 - round(points[j].x), 40 - round(points[j].y),
                                                            points[j].v_x, points[j].v_y)
                points[j].x = time_frames[t - 1][j].x
                points[j].y = time_frames[t - 1][j].y
            elif a == 61:
                points[j].v_x, points[j].v_y = new_velocity(160 - round(points[j].x), 200 - round(points[j].y),
                                                            points[j].v_x, points[j].v_y)
                points[j].x = time_frames[t - 1][j].x
                points[j].y = time_frames[t - 1][j].y
            '''

        if points[j].v_x > max_velocity:
            points[j].v_x = max_velocity
        if points[j].v_y > max_velocity:
            points[j].v_y = max_velocity

        if points[j].v_x < min_velocity:
            points[j].v_x = min_velocity
        if points[j].v_y < min_velocity:
            points[j].v_y = min_velocity

        points[j].x += points[j].v_x * t_step
        points[j].y += points[j].v_y * t_step

        if vx_f[j] == 1:

            cond1 = points[j].y > shape_y - bias
            cond2 = points[j].y < bias
            cond3 = points[j].x > shape_x - bias
            cond4 = points[j].x < x_min_lim
            cond5 = points[j].x < 159 + 400

            if wall_tube:
                if (cond1 or cond2) and cond5:
                    points[j].v_y = -points[j].v_y
                    points[j].y += points[j].v_y * t_step

                if (cond1 or cond2) and not cond5:
                    if cond1:
                        points[j].y = shape_y - bias
                    if cond2:
                        points[j].y = bias
                    if cond3:
                        points[j].x = shape_x - bias

                if cond3:
                    points[j].is_in = False
                    points[j].x = shape_x - bias
                if cond4:
                    points[j].v_x = -points[j].v_x
            else:
                if cond1:
                    points[j].y = shape_y - bias
                if cond2:
                    points[j].y = bias
                if cond3:
                    points[j].x = shape_x - bias
                if cond4:
                    points[j].x = x_min_lim
    pass


def dump_to_file(timestamp: datetime, frames: list[list[Point]]):
    cur_date_str = timestamp.strftime('%Y-%m-%d_%H-%M-%S')

    os.makedirs(out_folder, exist_ok=True)

    filename_mask1 = 'data/mask.npy'
    filename_coord_x1 = 'data/coord_x.npy'
    filename_coord_y1 = 'data/coord_y.npy'
    filename_time_frames1 = 'data/timeframes.pkl'
    params_tmpl = f"{shape_x}_time={time_steps}_len={size}_height={height}_bias={bias}_thresh={capillary_length}_{cur_date_str}"
    filename_mask2 = f"{out_folder}/mask_sh={params_tmpl}.npy"
    filename_coord_x2 = f"{out_folder}/coord_x_sh={params_tmpl}.npy"
    filename_coord_y2 = f"{out_folder}/coord_y_sh={params_tmpl}.npy"
    filename_time_frames2 = f"{out_folder}/timeframes={params_tmpl}.pkl"

    start_timer('Convert to old arrays')
    coord_x = np.zeros((full_size, time_steps))
    coord_y = np.zeros((full_size, time_steps))
    for i in range(len(frames)):
        for j in range(len(frames[i])):
            coord_x[j][i] = frames[i][j].x
            coord_y[j][i] = frames[i][j].y
    release_timer('Convert to old arrays')

    start_timer('Dump old arrays')
    np.save(filename_coord_x1, coord_x)
    np.save(filename_coord_y1, coord_y)
    shutil.copyfile(filename_coord_x1, filename_coord_x2)
    shutil.copyfile(filename_coord_y1, filename_coord_y2)
    release_timer('Dump old arrays')

    np.save(filename_mask1, aperture_mask)
    shutil.copyfile(filename_mask1, filename_mask2)

    start_timer('Dump new array')
    with open(filename_time_frames1, 'wb') as out_file:
        pickle.dump(frames, out_file)
    shutil.copyfile(filename_time_frames1, filename_time_frames2)
    release_timer('Dump new array')

    pass


def plot_trajectories():
    plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)
    for n in range(size):
        coords_x = [points[n].x for points in time_frames]
        coords_y = [points[n].y for points in time_frames]
        plt.plot(m * coords_x, m * coords_y, linewidth=0.5)
        # plt.scatter(coord_x[i,:], coord_y[i,:], s = 0.6)
    plt.imshow(aperture_mask)
    plt.show()
    pass


# Считывание маски диафрагмы
# aperture_mask = read_mask()
aperture_mask = mesh(shape_y, shape_x, height, m)  # маска диафрагмы
# aperture_mask = mask2 + aperture_mask

# wall_tube
# size
# folder

if __name__ == '__main__':
    # Задание массивов координат и скоростей
    start_timer('Initiate points')
    list_points = initiate_points(full_size, capillary_length, shape_y, height)
    release_timer('Initiate points')

    start_timer('First deep copy')
    time_frames: list[list[Point]] = [copy.deepcopy(list_points)]
    release_timer('First deep copy')

    start_timer('Initiate randoms')
    initiate_randoms()
    release_timer('Initiate randoms')

    vx_f = np.zeros(full_size)
    # is_in = np.ones(full_size)

    # Проход по временному циклу
    for t in range(1, time_steps):
        print(size, ' - len,', t, ' - time step')

        start_timer('Main cycle')

        # Проход по частицам
        start_timer('Split to cells')
        grid_map = split_to_cells(list_points)
        release_timer('Split to cells')

        # Проход по сетке для расчета скоростей частиц по методу Монте-Карло
        # print(f"Count cells for recalculation is {len(grid_map.keys())}")
        start_timer('Iteration over cells')
        iterate_over_grid(grid_map)
        release_timer('Iteration over cells')

        # Второй проход по частицам
        start_timer('Checking boundaries')
        checking_boundaries(list_points)
        release_timer('Checking boundaries')

        start_timer('Deep copy into timeframes')
        time_frames.append(copy.deepcopy(list_points))
        release_timer('Deep copy into timeframes')

        release_timer('Main cycle')

    # Сохранение координат частиц в файл
    dump_to_file(datetime.datetime.now(), time_frames)

    # Вывод траекторий, если требуется
    if True:
        plot_trajectories()
