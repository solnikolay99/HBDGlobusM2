# ******************************************************************************
# @author: L. I. Nurtdinova
# ******************************************************************************

import copy
import datetime
import math
import os
import pickle
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


def add_aperture_vertical(mask: np.array,
                          max_y: int,
                          multiply: int,
                          app_height: float,
                          app_width: int,
                          app_r: int,
                          x_position: int) -> np.array:
    aperture_width = app_width * multiply
    aperture_l2 = int(max_y * 0.5 - app_r * app_height)
    x = x_position * multiply
    mask[:aperture_l2, x:x + aperture_width] = wall_mask(aperture_width, aperture_l2)
    mask[max_y - aperture_l2:, x:x + aperture_width] = np.flipud(wall_mask(aperture_width, aperture_l2)) + 2

    return mask


def mesh(max_y: int, max_x: int, ap_height: float, multiply: int) -> np.array:
    """
    aperture_width - ширина диафрагмы
    aperture_l2 - расстояние между осью и нижней точкой диафрагмы
    x - координата по x левого края диафрагмы
    """
    max_x *= multiply
    max_y *= multiply
    ap_height *= multiply
    mask = 35 * np.ones((max_y, max_x))

    # mask = add_aperture_vertical(mask, max_y, multiply, ap_height, app_width=20, app_r=12, x_position=160)
    mask = add_aperture_vertical(mask, max_y, multiply, ap_height, app_width=20, app_r=8, x_position=360)
    mask = add_aperture_vertical(mask, max_y, multiply, ap_height, app_width=20, app_r=4, x_position=560)

    mask[(mask == 10) | (mask == 12)] = 35
    return mask


def read_mask() -> np.array:
    # pl = 35 * np.ones((shape_y, shape_x))
    pl = np.zeros((shape_y, shape_x))
    return pl + cv2.imread('data/output_image.bmp')[:, :, 2]


def initiate_points(count_points: int, max_length: int, y_size: int, half_height: float) -> list[Point]:
    coords_x = np.random.uniform(0, max_length, count_points)
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


def get_new_coords(x: float, y: float, v_x: float, v_y: float) -> (float, float, int, int):
    new_x = x + v_x * t_step
    new_y = y + v_y * t_step

    mask_x = round(m * new_x)
    mask_y = round(m * new_y)

    if mask_x < 0:
        mask_x = 0
        new_x = 0
    elif mask_x >= shape_x:
        mask_x = shape_x - 1
        new_x = shape_x - 1
    if mask_y < 0:
        mask_y = 0
        new_y = 0
    elif mask_y >= shape_y:
        mask_y = shape_y - 1
        new_y = shape_y - 1
    return new_x, new_y, mask_x, mask_y


def find_point_of_intersection(point: Point, new_x: float, new_y: float, mask: np.array) -> (float, float, float):
    x_length = abs(point.x - new_x)
    y_length = abs(point.y - new_y)
    if x_length > y_length:
        count_steps = int(round(x_length))
    else:
        count_steps = int(round(y_length))
    if count_steps == 0:
        return point.x, point.y, 1.0
    x_step = x_length / count_steps
    y_step = y_length / count_steps
    for i in range(count_steps - 1, 0, -1):
        test_x = round(m * (new_x - x_step * i))
        test_y = round(m * (new_y - y_step * i))
        mask_point = mask[test_y, test_x]
        if mask_point == 255:
            return test_x, test_y, 1 - i / count_steps
    return point.x, point.y, 1.0


def _checking_boundaries(points: list[Point], out_points: list):
    out_points.append([])
    for j in range(len(points) - 1, -1, -1):
        if points[j].is_in:
            if points[j].v_x > max_velocity:
                points[j].v_x = max_velocity
            elif points[j].v_x < min_velocity:
                points[j].v_x = min_velocity

            if points[j].v_y > max_velocity:
                points[j].v_y = max_velocity
            elif points[j].v_y < min_velocity:
                points[j].v_y = min_velocity

            new_x, new_y, mask_x, mask_y = get_new_coords(points[j].x, points[j].y, points[j].v_x, points[j].v_y)
            mask_point = aperture_mask[mask_y, mask_x]

            '''
            Соответствие цветов (в Paint) в числами в маске
            0 - black. Любая поверхность, от которой точка должна отражаться строго назад по x.
            34 - green. Наклонённая на 20 градусов поверхность.
            127 - gray. Любая поверхность, от которой точка должна отражаться строго назад по y.
            255 - white. Действия отсутствуют.
            '''
            if mask_point == 255 or points[j].in_capillary:
                points[j].x = new_x
                points[j].y = new_y
            else:
                points[j].x, points[j].y, coeff = find_point_of_intersection(points[j], new_x, new_y, aperture_mask)

                if mask_point == 0:
                    points[j].v_x = -points[j].v_x

                elif mask_point == 34:
                    points[j].v_x = np.cos(np.pi * 0.5 + np.radians(20)) * points[j].v_x
                    points[j].v_y = np.sin(np.pi * 0.5 + np.radians(20)) * points[j].v_y

                elif mask_point == 127:
                    points[j].v_y = -points[j].v_y

                new_x, new_y, mask_x, mask_y =\
                    get_new_coords(points[j].x, points[j].y, coeff * points[j].v_x, coeff * points[j].v_y)
                mask_point = aperture_mask[mask_y, mask_x]
                if mask_point == 255:
                    points[j].x = new_x
                    points[j].y = new_y

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
                    else:
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
                    '''
                    if cond1:
                        points[j].y = shape_y - bias
                        if cond5:
                            points[j].is_in = False
                        else:
                            points[j].v_y = -points[j].v_y
                    elif cond2:
                        points[j].y = bias
                        if cond5:
                            points[j].is_in = False
                        else:
                            points[j].v_y = -points[j].v_y
                    if cond3:
                        points[j].x = shape_x - bias
                        points[j].is_in = False
                    elif cond4:
                        points[j].x = x_min_lim
                        points[j].v_x = -points[j].v_x
                    '''
                    if cond1:
                        points[j].y = shape_y - bias
                        points[j].is_in = False
                    elif cond2:
                        points[j].y = bias
                        points[j].is_in = False
                    if cond3:
                        points[j].x = shape_x - bias
                        points[j].is_in = False
                        out_points[len(out_points) - 1].append(points[j].y)
                    elif cond4:
                        points[j].x = x_min_lim
                        points[j].v_x = -points[j].v_x
    pass


def checking_boundaries(points: list[Point], out_points: list):
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
        _checking_boundaries(points, out_points)
    pass


def dump_part(frames: list[list[Point]]) -> list[list[Point]]:
    if cur_time_frame[0] == 0:
        shutil.rmtree(data_folder, ignore_errors=True)
        os.makedirs(data_folder, exist_ok=True)

    # Сохранение параметров маски в файл
    if not os.path.exists(data_folder + '/mask.npy'):
        np.save(data_folder + '/mask.npy', aperture_mask)

    cur_time_frame[0] += 1

    filename_coords = f"{data_folder}/coords_{cur_time_frame[0]:06n}.npy"

    count_frames = dump_every if len(frames) > dump_every else len(frames)
    coords = np.zeros((count_frames, full_size, 2))

    for i in range(count_frames):
        for j in range(len(frames[i])):
            coords[i][j] = [frames[i][j].x, frames[i][j].y]

    np.save(filename_coords, coords)

    return frames[count_frames:]


def dump_statistics(out_points: list[list[list[float]]]):
    with open(data_folder + '/out_points.npy', 'wb') as f:
        pickle.dump(out_points, f)
    pass


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
aperture_mask_tmpl = read_mask()
# aperture_mask = mesh(shape_y, shape_x, height, m)  # маска диафрагмы
aperture_mask = aperture_mask_tmpl  # + aperture_mask

# aperture_mask[aperture_mask == 220] = 290
uniq_colors = np.unique(aperture_mask)

# wall_tube
# size
# folder

if __name__ == '__main__':
    start_timer('Full time execution')

    # Задание массивов координат и скоростей
    start_timer('Initiate points')
    list_points = initiate_points(size, capillary_length, shape_y, height)
    list_out_points = [[]]
    release_timer('Initiate points')

    start_timer('First deep copy')
    time_frames: list[list[Point]] = [copy.deepcopy(list_points)]
    release_timer('First deep copy')

    start_timer('Initiate randoms')
    initiate_randoms()
    release_timer('Initiate randoms')

    # Проход по временному циклу
    for t in range(1, time_steps):
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
        checking_boundaries(list_points, list_out_points)
        release_timer('Checking boundaries')

        # print('>>> Len points is', len(list_points))

        start_timer('Deep copy into timeframes')
        time_frames.append(copy.deepcopy(list_points))
        release_timer('Deep copy into timeframes')

        if len(time_frames) > dump_every:
            start_timer('Partially dump data')
            time_frames = dump_part(time_frames)
            release_timer('Partially dump data')

        release_timer('Main cycle')

        print(f"Time step: {t: 5n}; count points: {len(list_points): 6n};"
              f" iteration exec time: {time.time() - timers['Main cycle']:.4f}")

    if len(time_frames) < dump_every + 1:
        start_timer('Partially dump data')
        time_frames = dump_part(time_frames)
        release_timer('Partially dump data')

    dump_statistics(list_out_points)

    # Сохранение координат частиц в файл
    # duplicate_dumped_data(datetime.datetime.now())

    # Вывод траекторий, если требуется
    # plot_trajectories()

    print(f"Full exec time: {time.time() - timers['Full time execution']:.1f} seconds")
