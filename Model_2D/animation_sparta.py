# ******************************************************************************
# @author: L. I. Nurtdinova
# ******************************************************************************
import copy
import glob
import os
import time
import uuid
import math

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from matplotlib.animation import FuncAnimation
import re

# plt.rcParams.update({'font.size': 5})
matplotlib.rcParams[
    'animation.ffmpeg_path'] = "C:\\Program Files\\ffmpeg-2024-09-26-git-f43916e217-essentials_build\\bin\\ffmpeg.exe"
multiplayer = 100


def start_timer():
    return time.time()


def release_timer(timer_name: str, time_data: time):
    #print(f"Execution time for '{timer_name}' is {time.time() - time_data:.4f}")
    pass


# Сортировка по имени названий файлов, хранящих координаты  частиц
def name_reader(dir_path, pattern):
    return sorted(glob.glob(dir_path + '\\' + pattern))


def pars_in_file(f_name: str) -> dict[str, any]:
    out_params = dict()
    with open(f_name) as file:
        lines = [line.rstrip() for line in file]
    for line in lines:
        line = line.strip()
        line = re.sub('[\t]', '', line)
        line = re.sub(' +', ' ', line)
        params = line.strip().split(' ')
        if len(params) <= 0:
            continue
        if params[0].startswith('#'):
            continue
        elif params[0].strip() == 'global':
            for i in range(1, len(params), 2):
                out_params[params[i].strip()] = params[i + 1].strip()
        elif params[0].strip() == 'timestep':
            out_params['timestep'] = params[1].strip()
    return out_params


def pars_data(f_name: str) -> (list[list[float]], int, int):
    with open(f_name) as file:
        lines = [line.rstrip() for line in file]
    mask_width = int(float(lines[5].split(' ')[1]) * multiplayer)
    mask_height = int(float(lines[6].split(' ')[1]) * multiplayer)
    header = lines[8]
    lines = lines[9:]

    headers = header.replace('ITEM: ATOMS ', '').strip().split(' ')
    points = []
    for line in lines:
        params = line.strip().split(' ')
        points.append([float(params[headers.index('x')]) * multiplayer,
                       float(params[headers.index('y')]) * multiplayer,
                       int(params[headers.index('id')])])

    return points, mask_width, mask_height


def pars_surf_file(f_name: str) -> (dict[int, dict[str, int]], dict[int, dict[str, int]]):
    points = dict()
    lines = dict()

    with open(f_name) as file:
        f_lines = [line.rstrip() for line in file]

    count_points = int(f_lines[2].strip().split(' ')[0])
    count_lines = int(f_lines[3].strip().split(' ')[0])

    for i in range(7, 7 + count_points):
        point = f_lines[i].strip().split(' ')
        points[int(point[0])] = {'x': int(float(point[1]) * multiplayer), 'y': int(float(point[2]) * multiplayer)}
    for i in range(10 + count_points, 10 + count_points + count_lines):
        line = f_lines[i].strip().split(' ')
        lines[int(line[0])] = {'from': int(line[1]), 'to': int(line[2])}

    return points, lines


def create_mask_template(data_file: str) -> Image:
    points, lines = pars_surf_file(data_file)

    max_x = 0
    max_y = 0
    for key in points.keys():
        if points[key]['x'] > max_x:
            max_x = points[key]['x']
        if points[key]['y'] > max_y:
            max_y = points[key]['y']

    image = Image.new('RGBA', (max_x + 1, max_y + 1), (0, 100, 0, 255))
    draw = ImageDraw.Draw(image)

    for key in lines.keys():
        point_from = points[lines[key]['from']]
        point_to = points[lines[key]['to']]
        draw.line((point_from['x'], point_from['y'], point_to['x'], point_to['y']), fill=255)

    return image


def create_mask(template: Image, new_width: int, new_height: int) -> Image:
    background = Image.new('RGBA', (new_width, new_height), (0, 100, 0, 255))
    background.paste(template)
    return background


'''
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
'''


def calculate_density(width: int,
                      height: int,
                      timeframe: list[list[float]],
                      labels: list[float],
                      densities: list[int],
                      point_ids: dict[int, dict[str, int]]) -> (list[float], list[int], dict[int, dict[str, int]]):
    calc_density = start_timer()
    if len(labels) < 1:
        labels = [i for i in range(height)]
    if len(densities) < 1:
        densities = [0 for _ in range(height)]

    put_data_to_point_ids = start_timer()
    seed = uuid.uuid4()
    for point in timeframe:
        # if width - 1 >= point[0] >= width - 2:
        #    densities[int(point[1])] += 1
        if point[0] > width - 10:
            point_ids[int(point[2])] = {'y': int(point[1]), 'seed': seed}
    release_timer('Put data to point_ids', put_data_to_point_ids)

    store_data_density = start_timer()
    keys = list(point_ids.keys())
    for key in keys:
        if point_ids[key]['seed'] != seed:
            densities[point_ids[key]['y']] += 1
            del point_ids[key]
    release_timer('Store data to densities', store_data_density)

    release_timer('Calculation density', calc_density)
    return labels, densities, point_ids


def calculate_density_diameter(densities: list[int]) -> (float, float, float):
    min_y = len(densities)
    max_y = 0
    for i in range(len(densities)):
        if densities[i] > 0:
            if i < min_y:
                min_y = i
            if i > max_y:
                max_y = i
    if min_y < max_y:
        return max_y - min_y, min_y, max_y
    else:
        return 0.0, 0.0, 0.0


def calculate_angel(length: float, width: float, min_y: float, max_y: float) -> float:
    if min_y == 0.0 and max_y == 0.0:
        return 0.0
    max_r = max(abs(width * 0.5 - min_y), abs(width * 0.5 - max_y))
    return math.degrees(math.atan(max_r / length))


def update(frame):
    global width, height, mask, density_labels, uniq_points

    try:
        '''
        timeframe, width, height = pars_data(data_files[frame])

        if mask is None:
            mask = create_mask(mask_template, width, height)
        '''

        timeframe = timeframes[frame]

        for point in timeframe:
            uniq_points.add(int(point[2]))

        coords_x = np.array(timeframe)[:, 0]
        coords_y = np.array(timeframe)[:, 1]

        '''
        density_labels, density_values, density_point_ids =\
            calculate_density(width, height, timeframe, density_labels, density_values, density_point_ids)
        '''

        density_values = densities[frame]

        sum_count_points = sum(density_values)
        density_diameter, min_y, max_y = calculate_density_diameter(density_values)
        angel = calculate_angel(width, height, min_y, max_y)

        timestep = float(global_params.get('timestep'))

        ax1.clear()
        ax1.set_title(f"{(timestep * 1e6 * frame * pick_every_timeframe): 6.2f} мкс")
        ax1.scatter(coords_x, coords_y, marker='.', s=0.5, color='#ff531f')

        fnum = float(global_params.get('fnum'))

        ax2.clear()
        ax2.set_title(f"Общее число частиц: {((len(uniq_points) * fnum) ** (3 / 2)) * 1e3:.2e} в сек\n"
                      f"Плотность потока на выходе: {((sum_count_points * fnum) ** (3 / 2)) * 1e3:.2e} частиц в сек\n"
                      f"Диаметр потока: {density_diameter * 0.1:.1f} / {angel:.2f} градусов")
        ax2.plot(density_values, density_labels)

        plt.xlabel('x, ед.')
        plt.ylabel('y, 100 мкм')
        #plt.xlim(0, 10)
        plt.ylim(0, height)
    except:
        pass

    ax1.imshow(mask)
    # current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # if saving_to_file:
    #    plt.savefig(str(main_dir_path) + '/profs/prof' + str(current_datetime) + '.png', dpi=200, bbox_inches='tight')
    #    print(f'{frame} frame was saved')
    print(f'{frame} frame')


def save_mask():
    mask_tmpl = create_mask_template(main_dir_path + '\\data.step')
    mask_tmpl = create_mask(mask_tmpl, 960, 240)
    mask_tmpl.save(os.getcwd() + '/data/mask_tmpl_nozzle.png')


if __name__ == '__main__':
    # Считывание массивов из файлов и их конкатенация
    main_dir_path = '\\\\wsl.localhost\\Ubuntu\\home\\c\\sparta_build\\textor'
    parts_dir_path = main_dir_path + '\\dumps\\'

    pick_every_timeframe = 100
    saving_to_file = 1

    global_params = pars_in_file(main_dir_path + '\\in.step')

    data_files = name_reader(parts_dir_path, 'dump.*.txt')
    timeframes, densities = [], []
    density_labels, density_values, density_point_ids = [], [], dict()
    uniq_points = set()

    width, height = 0, 0
    for i in range(len(data_files)):
        if i % 100 == 0:
            print(f"Store data from file '{data_files[i]}'")
        timeframe, width, height = pars_data(data_files[i])
        if i % pick_every_timeframe == 0:
            timeframes.append(timeframe)
        density_labels, density_values, density_point_ids = \
            calculate_density(width, height, timeframe, density_labels, density_values, density_point_ids)
        if i % pick_every_timeframe == 0:
            densities.append(copy.deepcopy(density_values))

    mask_template = create_mask_template(main_dir_path + '\\data.step')
    mask = create_mask(mask_template, width, height)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), width_ratios=[2, 1])

    ani = FuncAnimation(fig, update, frames=range(len(timeframes)), interval=1)
    if saving_to_file:
        FFwriter = animation.FFMpegWriter(fps=10)
        ani.save(os.getcwd() + '/data/animation.mp4', writer=FFwriter)
        # ani.save(os.getcwd() + '/data/animation.gif')
    else:
        plt.show()
