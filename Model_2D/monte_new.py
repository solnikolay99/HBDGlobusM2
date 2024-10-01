# ******************************************************************************
# @author: L. I. Nurtdinova
# ******************************************************************************

import copy
import datetime
import math
import os
import shutil
import time
from threading import Thread

import cv2
import matplotlib.pyplot as plt

from config import *
from monteclass import *

# pre-defined randoms
randoms_1 = Randoms()  # for randoms from [0, 1)
randoms_2pi = Randoms()  # for randoms from [0, 2*pi)
randoms_mp1 = Randoms()  # for randoms from [-1, 1)

timers = {}  # times for profiling app


def start_timer(timer_name: str):
    if debug:
        timers[timer_name] = time.time()
    pass


def release_timer(timer_name: str):
    if debug:
        print(f"Execution time for '{timer_name}' is {time.time() - timers[timer_name]:.4f}")
    pass


def initiate_randoms():
    global randoms_1, randoms_2pi, randoms_mp1
    randoms_1 = Randoms(0, 1, full_size * 10)  # for randoms from [0, 1)
    randoms_2pi = Randoms(0, 2 * np.pi, full_size * 10)  # for randoms from [0, 2*pi)
    randoms_mp1 = Randoms(-1, 1, full_size * 10)  # for randoms from [-1, 1)
    pass


def inverse_maxwell_distribution(x: float) -> float:
    return np.sqrt(k_t_m_hel * np.log(1 - x)) * denominator


def calc_point_velocities(points: list[Point]) -> (list[list], float):
    part_len = len(points)
    pairs = []
    velocity_max = 0
    for j in range(part_len):
        for i in range(j + 1, part_len):
            vel_square = np.sqrt((points[i].v_x - points[j].v_x) * (points[i].v_x - points[j].v_x) +
                                 (points[i].v_y - points[j].v_y) * (points[i].v_y - points[j].v_y))
            if vel_square > velocity_max:
                velocity_max = vel_square
            pairs.append([vel_square,
                          points[i],
                          points[j]])
    return pairs, velocity_max


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


def mesh(max_y: int, max_x: int, height: float, multiply: int) -> np.array:
    """
    aperture_width - ширина диафрагмы
    aperture_l2 - расстояние между осью и нижней точкой диафрагмы
    x - координата по x левого края диафрагмы
    """
    max_x *= multiply
    max_y *= multiply
    height *= multiply
    mask = 35 * np.ones((max_y, max_x))

    '''
    aperture_width = 20 * multiply
    aperture_l2 = int(max_y // 2 - 4 * height)
    x = 160 * multiply
    mask[:aperture_l2, x:x + aperture_width] = wall_mask(aperture_width, aperture_l2)
    mask[max_y - aperture_l2:, x:x + aperture_width] = np.flipud(wall_mask(aperture_width, aperture_l2)) + 2
    '''

    aperture_width = 20 * multiply
    aperture_l2 = int(max_y // 2 - 4 * height)
    x = 360 * multiply
    mask[:aperture_l2, x:x + aperture_width] = wall_mask(aperture_width, aperture_l2)
    mask[max_y - aperture_l2:, x:x + aperture_width] = np.flipud(wall_mask(aperture_width, aperture_l2)) + 2

    aperture_width = 20 * multiply
    aperture_l2 = int(max_y // 2 - 12 * height)
    x = 560 * multiply
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
                  # v_z=0,
                  is_in=True,
                  in_capillary=True) for i in range(count_points)]


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
            if points[j].in_capillary:
                if points[j].x > capillary_length:
                    points[j].in_capillary = False
                    v = inverse_maxwell_distribution(randoms_1.get_next())
                    angel = randoms_2pi.get_next()
                    points[j].v_x = Vx + v * np.cos(angel)
                    points[j].v_y = v * np.sin(angel)
                else:
                    points[j].v_x = V_pot
                    points[j].v_y = 1
                points[j] = check_capillary(points[j])

            # Размножение частиц
            if len(points) < full_size:
                if points[j].x > 0 and not points[j].in_capillary:
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

                    di1_cond = di[0] < (full_size - size) * 0.2
                    di2_cond = di[1] < (full_size - size) * 0.3
                    di3_cond = di[2] < (full_size - size) * 0.5

                    if (c_cond1 and di1_cond) or (c_cond2 and di2_cond) or (c_cond3 and di3_cond):
                        if randoms_1.get_next() < prob:
                            new_point = Point(
                                x=points[j].x + 0.5 * np.sign(randoms_mp1.get_next()) * randoms_1.get_next(),
                                y=points[j].y + 0.5 * np.sign(randoms_mp1.get_next()) * randoms_1.get_next(),
                                v_x=points[j].v_x,
                                v_y=points[j].v_y,
                                is_in=points[j].is_in,
                                in_capillary=points[j].in_capillary)
                            points.append(new_point)
                            if c_cond1 and di1_cond:
                                di[0] += 1
                            if c_cond2 and di2_cond:
                                di[1] += 1
                            if c_cond3 and di3_cond:
                                di[2] += 1

            # Добавление частиц в расчетную сетку для метода Монте-Карло
            if points[j].v_x > 0 or j < full_size:
                key = f"{round(points[j].y)}_{round(points[j].x)}"
                if key not in grid_dict:
                    grid_dict[key] = [points[j]]
                else:
                    if len(grid_dict[key]) < max_points_per_cell:
                        grid_dict[key].append(points[j])
    return grid_dict


def _iterate_over_grid(grid: dict[str, list[Point]]):
    for cell_points in grid.values():
        if len(cell_points) > 1:
            point_pairs, vel_square_max = calc_point_velocities(cell_points)

            for point_pair in point_pairs:
                if point_pair[0] > vel_square_max * randoms_1.get_next():
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


def iterate_over_grid(grid: dict[str, list[Point]]):
    if use_multithread:
        threads = []
        keys = list(grid.keys())
        range_per_thread = math.ceil(len(keys) / thread_count)
        for i in range(thread_count):
            part_grid = {}
            for key in keys[i * range_per_thread:(i + 1) * range_per_thread]:
                part_grid[key] = grid.get(key)
            thread = Thread(target=_iterate_over_grid, args=(part_grid,))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _iterate_over_grid(grid)
    pass


def _checking_boundaries(points: list[Point]):
    last_frame = len(time_frames) - 1
    for j in range(len(points) - 1, -1, -1):
        if points[j].is_in:
            a = aperture_mask[round(m * points[j].y), round(m * points[j].x)]

            if a == 0 or a == 2:
                points[j].v_x = -points[j].v_x
                points[j].x = time_frames[last_frame - 1][j].x
                points[j].y = time_frames[last_frame - 1][j].y

            elif a == 6:
                points[j].v_x, points[j].v_y = new_velocity(0, 1, points[j].v_x, points[j].v_y)
                points[j].x = time_frames[last_frame - 1][j].x
                points[j].y = time_frames[last_frame - 1][j].y

            elif a == 8:
                points[j].v_x, points[j].v_y = new_velocity(0, -1, points[j].v_x, points[j].v_y)
                points[j].x = time_frames[last_frame - 1][j].x
                points[j].y = time_frames[last_frame - 1][j].y

            elif a == 98:
                points[j].v_y = -points[j].v_y
                points[j].x = time_frames[last_frame - 1][j].x
                points[j].y = time_frames[last_frame - 1][j].y

            '''
            elif a == 124 or a == 176:
                points[j].v_x, points[j].v_y = new_velocity(-1, 0, points[j].v_x, points[j].v_y)
                points[j].x = time_frames[last_frame - 1][j].x
                points[j].y = time_frames[last_frame - 1][j].y

            if a == 0:
                points[j].v_x, points[j].v_y = new_velocity(160 - round(points[j].x), 40 - round(points[j].y),
                                                            points[j].v_x, points[j].v_y)
                points[j].x = time_frames[last_frame - 1][j].x
                points[j].y = time_frames[last_frame - 1][j].y
            elif a == 61:
                points[j].v_x, points[j].v_y = new_velocity(160 - round(points[j].x), 200 - round(points[j].y),
                                                            points[j].v_x, points[j].v_y)
                points[j].x = time_frames[last_frame - 1][j].x
                points[j].y = time_frames[last_frame - 1][j].y
            '''

            if points[j].v_x > max_velocity:
                points[j].v_x = max_velocity
            elif points[j].v_x < min_velocity:
                points[j].v_x = min_velocity

            if points[j].v_y > max_velocity:
                points[j].v_y = max_velocity
            elif points[j].v_y < min_velocity:
                points[j].v_y = min_velocity

            points[j].x += points[j].v_x * t_step
            points[j].y += points[j].v_y * t_step

            if not points[j].in_capillary:

                cond1 = points[j].y > shape_y - bias
                cond2 = points[j].y < bias
                cond3 = points[j].x > shape_x - bias
                cond4 = points[j].x < x_min_lim
                cond5 = points[j].x < 159 + 400

                if wall_tube:
                    if (cond1 or cond2) and cond5:
                        points[j].v_y = -points[j].v_y
                        points[j].y += points[j].v_y * t_step
                    elif (cond1 or cond2) and not cond5:
                        if cond1:
                            points[j].y = shape_y - bias
                        elif cond2:
                            points[j].y = bias
                        elif cond3:
                            points[j].x = shape_x - bias

                    if cond3:
                        points[j].is_in = False
                        points[j].x = shape_x - bias
                    elif cond4:
                        points[j].v_x = -points[j].v_x
                else:
                    if cond1 and cond5:
                        points[j].y = shape_y - bias
                        points[j].is_in = False
                    elif cond2 and cond5:
                        points[j].y = bias
                        points[j].is_in = False
                    if cond3:
                        points[j].x = shape_x - bias
                        points[j].is_in = False
                    elif cond4 and cond5:
                        points[j].x = x_min_lim
                        points[j].v_x = -points[j].v_x
    pass


def checking_boundaries(points: list[Point]):
    if use_multithread:
        threads = []
        range_per_thread = math.ceil(len(points) / thread_count)
        for i in range(0, len(points), range_per_thread):
            part_points = points[i:i + range_per_thread]
            thread = Thread(target=_checking_boundaries, args=(part_points,))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _checking_boundaries(points)
    pass


def dump_part(frames: list[list[Point]]) -> list[list[Point]]:
    if cur_time_frame[0] == 0:
        shutil.rmtree(data_folder, ignore_errors=True)
        os.makedirs(data_folder, exist_ok=True)

    # Сохранение параметров маски в файл
    if not os.path.exists(data_folder + '/mask.npy'):
        np.save(data_folder + '/mask.npy', aperture_mask)

    cur_time_frame[0] += 1

    filename_coord_x = f"{data_folder}/coord_x_{cur_time_frame[0]:06n}.npy"
    filename_coord_y = f"{data_folder}/coord_y_{cur_time_frame[0]:06n}.npy"

    count_frames = dump_every if len(frames) > dump_every else len(frames)
    coord_x = np.zeros((count_frames, full_size))
    coord_y = np.zeros((count_frames, full_size))

    for i in range(count_frames):
        for j in range(len(frames[i])):
            coord_x[i][j] = frames[i][j].x
            coord_y[i][j] = frames[i][j].y

    np.save(filename_coord_x, coord_x)
    np.save(filename_coord_y, coord_y)

    return frames[count_frames:]


def duplicate_dumped_data(cur_date: datetime):
    cur_date_str = cur_date.strftime('%Y-%m-%d_%H-%M-%S')

    os.makedirs(out_folder, exist_ok=True)

    duplicate_folder_path = f"{out_folder}/{cur_date_str}_mask_sh={shape_x}_time={time_steps}" \
                            f"_len={size}_height={height}_bias={bias}_thresh={capillary_length}"

    shutil.copytree(data_folder, duplicate_folder_path)

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
    list_points = initiate_points(size, capillary_length, shape_y, height)
    release_timer('Initiate points')

    start_timer('First deep copy')
    time_frames: list[list[Point]] = [copy.deepcopy(list_points)]
    release_timer('First deep copy')

    start_timer('Initiate randoms')
    initiate_randoms()
    release_timer('Initiate randoms')

    # Проход по временному циклу
    for t in range(1, time_steps):
        print(datetime.datetime.now().strftime('%H:%M:%S.%f'), '   ', size, ' - len,', t, ' - time step')

        start_timer('Main cycle')

        # Проход по частицам
        start_timer('Split to cells')
        grid_map = split_to_cells(list_points)
        release_timer('Split to cells')

        # Проход по сетке для расчета скоростей частиц по методу Монте-Карло
        start_timer('Iteration over cells')
        iterate_over_grid(grid_map)
        release_timer('Iteration over cells')

        # Второй проход по частицам
        start_timer('Checking boundaries')
        checking_boundaries(list_points)
        release_timer('Checking boundaries')

        # print('>>> Len points is', len(list_points))

        start_timer('Deep copy into timeframes')
        time_frames.append(copy.deepcopy(list_points))
        release_timer('Deep copy into timeframes')

        if len(time_frames) > dump_every + 1:
            start_timer('Partially dump data')
            time_frames = dump_part(time_frames)
            release_timer('Partially dump data')

        release_timer('Main cycle')

    if len(time_frames) < dump_every + 1:
        start_timer('Partially dump data')
        time_frames = dump_part(time_frames)
        release_timer('Partially dump data')

    # Сохранение координат частиц в файл
    duplicate_dumped_data(datetime.datetime.now())

    # Вывод траекторий, если требуется
    # plot_trajectories()
