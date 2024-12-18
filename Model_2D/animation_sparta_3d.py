# ******************************************************************************
# @author: L. I. Nurtdinova
# ******************************************************************************
import copy
import glob
import math
import os
import re
import time
import traceback
import uuid

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from matplotlib.animation import FuncAnimation

matplotlib.rcParams[
    'animation.ffmpeg_path'] = "C:\\Program Files\\ffmpeg-2024-09-26-git-f43916e217-essentials_build\\bin\\ffmpeg.exe"
matplotlib.rcParams['legend.markerscale'] = 10
multiplayer = 100


def start_timer():
    return time.time()


def release_timer(timer_name: str, time_data: time):
    # print(f"Execution time for '{timer_name}' is {time.time() - time_data:.4f}")
    pass


def name_reader(dir_path, pattern):
    return sorted(glob.glob(os.path.join(dir_path, pattern)))


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
        if params[0].strip() == 'global':
            for i in range(1, len(params), 2):
                out_params[params[i].strip()] = params[i + 1].strip()
        elif params[0].strip() == 'timestep':
            out_params['timestep'] = params[1].strip()
        elif params[0].strip() == 'create_box':
            out_params['width'] = int(float(params[2].strip()) - float(params[1].strip())) * multiplayer
            out_params['height'] = int(float(params[4].strip()) - float(params[3].strip())) * multiplayer
            out_params['x_dim_lo'] = int(float(params[1].strip()) * multiplayer)
            out_params['x_dim_hi'] = int(float(params[2].strip()) * multiplayer)
            out_params['y_dim_lo'] = int(float(params[3].strip()) * multiplayer)
            out_params['y_dim_hi'] = int(float(params[4].strip()) * multiplayer)
            out_params['z_dim_lo'] = int(float(params[5].strip()) * multiplayer)
            out_params['z_dim_hi'] = int(float(params[6].strip()) * multiplayer)
        elif params[0].strip() == 'read_surf':
            if 'surfs' not in out_params:
                out_params['surfs'] = []
            out_params['surfs'].append(params[1].strip())
    return out_params


def pars_data(f_name: str) -> list[list[float]]:
    with open(f_name) as file:
        lines = [line.rstrip() for line in file]
    header = lines[8]
    lines = lines[9:]

    headers = header.replace('ITEM: ATOMS ', '').strip().split(' ')
    points = []
    for line in lines:
        x, y, z = 0, 0, 0
        params = line.strip().split(' ')
        if 'x' in headers:
            x = float(params[headers.index('x')]) * multiplayer
        if 'y' in headers:
            y = float(params[headers.index('y')]) * multiplayer
        if 'z' in headers:
            z = float(params[headers.index('z')]) * multiplayer
        points.append([x, y, z, int(params[headers.index('id')])])

    return points


def pars_surf_file(f_name: str) -> (dict[int, dict[str, int]], dict[int, dict[str, int]]):
    points = dict()
    lines = dict()

    with open(f_name) as file:
        f_lines = [line.rstrip() for line in file]

    count_points = int(f_lines[2].strip().split(' ')[0])
    count_lines = int(f_lines[3].strip().split(' ')[0])

    polygons = []
    add_new_polygon = True

    for i in range(7, 7 + count_points):
        point = f_lines[i].strip().split(' ')
        points[int(point[0])] = {'x': int(float(point[1]) * multiplayer), 'y': int(float(point[2]) * multiplayer)}
    for i in range(10 + count_points, 10 + count_points + count_lines):
        line = f_lines[i].strip().split(' ')
        lines[int(line[0])] = {'from': int(line[1]), 'to': int(line[2])}
        if add_new_polygon:
            polygons.append([(points[int(line[1])]['x'], points[int(line[1])]['y'])])
        else:
            polygons[len(polygons) - 1].append((points[int(line[1])]['x'], points[int(line[1])]['y']))
        if int(line[0]) > int(line[2]):
            add_new_polygon = True
        else:
            add_new_polygon = False

    return points, lines, polygons


def create_mask_template(path_main_dir: str) -> Image:
    global global_params

    if 'surfs' not in global_params:
        return Image.new('RGBA', (1, 1), (0, 100, 0, 255))

    surfs = []
    max_x = 0
    max_y = 0

    for surf_file in global_params['surfs']:
        points, lines, polygons = pars_surf_file(os.path.join(path_main_dir, surf_file))
        surfs.append({
            'points': copy.deepcopy(points),
            'lines': copy.deepcopy(lines),
            'polygons': copy.deepcopy(polygons),
        })
        for key in points.keys():
            if points[key]['x'] > max_x:
                max_x = points[key]['x']
            if points[key]['y'] > max_y:
                max_y = points[key]['y']

    image = Image.new('RGBA', (max_x + 1, max_y + 1), (0, 100, 0, 255))
    draw = ImageDraw.Draw(image)

    '''
    for key in lines.keys():
        point_from = points[lines[key]['from']]
        point_to = points[lines[key]['to']]
        draw.line((point_from['x'], point_from['y'], point_to['x'], point_to['y']), fill=(0, 0, 0, 255))
    '''

    for surf in surfs:
        for polygon in surf['polygons']:
            draw.polygon(polygon, fill=(127, 127, 127, 255))

    return image


def create_mask(template: Image, new_width: int, new_height: int) -> Image:
    background = Image.new('RGBA', (new_width, new_height), (0, 100, 0, 255))
    background.paste(template)
    return background


def calculate_density(width: int,
                      height: int,
                      timeframe: list[list[float]],
                      labels: list[float],
                      densities: list[int],
                      point_ids: dict[int, dict[str, int]],
                      smoothing: int) -> (list[float], list[int], dict[int, dict[str, int]]):
    if len(labels) < 1:
        labels = [i for i in range(int(height / smoothing) + 1)]
    if len(densities) < 1:
        densities = [0 for _ in range(int(height / smoothing) + 1)]

    seed = uuid.uuid4()
    for point in timeframe:
        # if width - 1 >= point[0] >= width - 2:
        #    densities[int(point[1])] += 1
        if point[0] > width - 10:
            point_ids[int(point[2])] = {'y': int(point[1]), 'seed': seed}

    keys = list(point_ids.keys())
    for key in keys:
        if point_ids[key]['seed'] != seed:
            densities[int(point_ids[key]['y'] / smoothing)] += 1
            del point_ids[key]

    return labels, densities, point_ids


def calculate_density_diameter(densities: list[int], percentile: float, reducer: int) -> (float, float, float):
    min_y = len(densities)
    max_y = 0

    if len(densities) == 0:
        return 0.0, 0.0, 0.0

    min_value = max(densities) * (1 - percentile)
    for i in range(len(densities)):
        if densities[i] > min_value:
            if i < min_y:
                min_y = i
            if i > max_y:
                max_y = i
    if min_y < max_y:
        return (max_y - min_y) * reducer, min_y * reducer, max_y * reducer
    else:
        return 0.0, 0.0, 0.0


def calculate_angel(length: float, width: float, min_y: float, max_y: float) -> float:
    if min_y == 0.0 and max_y == 0.0:
        return 0.0
    max_r = max(abs(width * 0.5 - min_y), abs(width * 0.5 - max_y))
    return math.degrees(math.atan(max_r / length))


def get_frame_data(frame: int, count_frames: int):
    global density_labels, density_values, density_point_ids, uniq_points, last_uniq_points

    if frame == 0:
        return [], [], []

    i = 0
    timeframe = []
    for i in range(frame - count_frames, frame):
        timeframe = pars_data(data_files[i])
        density_labels, density_values, density_point_ids = \
            calculate_density(width, height, timeframe, density_labels, density_values, density_point_ids,
                              density_smoothing)

        last_uniq_points = set()
        for point in timeframe:
            if int(point[2]) not in uniq_points:
                last_uniq_points.add(int(point[2]))
            uniq_points.add(int(point[2]))

    print(f"Stored data from file '{data_files[i]}'")

    return timeframe, density_labels, density_values


def show_textor_graph(data: list[list[float]]):
    values, labels = [], []
    for point in data:
        values.append(point[0])
        labels.append(point[1])
    ax2.clear()
    ax2.plot(values, labels, color='green', label='Textor')


def get_only_nonzero_labels(in_values: list[int], in_labels: list[float]) -> (list[int], list[float]):
    out_values, out_labels = [], []
    for i in range(len(in_values)):
        if in_values[i] > 0:
            out_values.append(in_values[i])
            out_labels.append(in_labels[i])
    return out_values, out_labels


def shift_labels_by_smoothing(in_values: list[int], in_labels: list[float], smoothing: int) -> (list[int], list[float]):
    out_values, out_labels = [], []
    for i in range(len(in_labels)):
        out_values.append(in_values[i])
        out_labels.append(in_labels[i] * smoothing)
    return out_values, out_labels


def calculate_kn(cell_points: list[list[float]]) -> float:
    fnum = float(global_params.get('fnum'))
    sigma = 2.33e-8
    L = 5e-3
    n = (len(cell_points) * fnum) / (L * L * 1)
    Kn = 1 / (math.sqrt(2) * sigma * sigma * n) / L
    return Kn


def separate_points_by_kn(timeframe: list[list[float]]) -> list[dict[str, list[list[float]]]]:
    out_graph = [
        {
            'color': '#0000ff',
            'legend': 'Kn < 0.001',
            'points': [[], []]
        },
        {
            'color': '#ff531f',
            'legend': '0.001 < Kn < 1',
            'points': [[], []]
        },
        {
            'color': '#ffffff',
            'legend': '1 <= Kn',
            'points': [[], []]
        },
    ]

    grid = dict()
    for point in timeframe:
        key = f'{int(point[0] / 50)}_{int(point[1] / 50)}'
        if key not in grid:
            grid[key] = [[], []]
        grid[key][0].append(point[0])
        grid[key][1].append(point[1])

    for key in grid.keys():
        kn = calculate_kn(grid[key])
        if kn < 1e-3:
            for point in grid[key][0]:
                out_graph[0]['points'][0].append(point)
            for point in grid[key][1]:
                out_graph[0]['points'][1].append(point)
        elif kn >= 1:
            for point in grid[key][0]:
                out_graph[2]['points'][0].append(point)
            for point in grid[key][1]:
                out_graph[2]['points'][1].append(point)
        else:
            for point in grid[key][0]:
                out_graph[1]['points'][0].append(point)
            for point in grid[key][1]:
                out_graph[1]['points'][1].append(point)

    return out_graph


def separate_points_by_density(timeframe: list[list[float]]) -> list[dict[str, list[list[float]]]]:
    fnum = float(global_params.get('fnum'))
    # fnum = 1e0

    out_graph = [
        {
            'color': '#ff531f',
            'legend': f'N <  {2 * fnum:.0e}',
            'points': [[], [], []]
        },
        {
            'color': '#00A2E8',
            'legend': f'{2 * fnum:.0e} ≤ N < {4 * fnum:.0e}',
            'points': [[], [], []]
        },
        {
            'color': '#0000ff',
            'legend': f'{4 * fnum:.0e} ≤ N < {6 * fnum:.0e}',
            'points': [[], [], []]
        },
        {
            'color': '#000000',
            'legend': f'{6 * fnum:.0e} ≤ N < {8 * fnum:.0e}',
            'points': [[], [], []]
        },
        {
            'color': '#ffffff',
            'legend': f'{8 * fnum:.0e} ≤ N < {10 * fnum:.0e}',
            'points': [[], [], []]
        },
        {
            'color': '#00ff00',
            'legend': f'N ≥ {10 * fnum:.0e}',
            'points': [[], [], []]
        },
    ]

    grid = dict()
    for point in timeframe:
        key = f'{int(point[0] / (0.05 * multiplayer))}_{int(point[1] / (0.05 * multiplayer))}_{int(point[2] / (0.05 * multiplayer))}'
        if key not in grid:
            grid[key] = [[], [], []]
        grid[key][0].append(point[0])
        grid[key][1].append(point[1])
        grid[key][2].append(point[2])

    for key in grid.keys():
        po = len(grid[key][0])

        if po < 2:
            for point in grid[key][0]:
                out_graph[0]['points'][0].append(point)
            for point in grid[key][1]:
                out_graph[0]['points'][1].append(point)
            for point in grid[key][2]:
                out_graph[0]['points'][2].append(point)
        elif 2 <= po < 4:
            for point in grid[key][0]:
                out_graph[1]['points'][0].append(point)
            for point in grid[key][1]:
                out_graph[1]['points'][1].append(point)
            for point in grid[key][2]:
                out_graph[1]['points'][2].append(point)
        elif 4 <= po < 6:
            for point in grid[key][0]:
                out_graph[2]['points'][0].append(point)
            for point in grid[key][1]:
                out_graph[2]['points'][1].append(point)
            for point in grid[key][2]:
                out_graph[2]['points'][2].append(point)
        elif 6 <= po < 8:
            for point in grid[key][0]:
                out_graph[3]['points'][0].append(point)
            for point in grid[key][1]:
                out_graph[3]['points'][1].append(point)
            for point in grid[key][2]:
                out_graph[3]['points'][2].append(point)
        elif 8 <= po < 10:
            for point in grid[key][0]:
                out_graph[4]['points'][0].append(point)
            for point in grid[key][1]:
                out_graph[4]['points'][1].append(point)
            for point in grid[key][2]:
                out_graph[4]['points'][2].append(point)
        else:
            for point in grid[key][0]:
                out_graph[5]['points'][0].append(point)
            for point in grid[key][1]:
                out_graph[5]['points'][1].append(point)
            for point in grid[key][2]:
                out_graph[5]['points'][2].append(point)

    return out_graph


def update(frame):
    global mask, density_labels, density_values, density_point_ids, uniq_points, last_uniq_points, data_fig9a

    try:
        timeframe, density_labels, density_values = get_frame_data(frame, pick_every_timeframe)

        sum_count_points = sum(density_values)
        density_diameter, min_y, max_y = calculate_density_diameter(density_values, 1.0, density_smoothing)
        angel = calculate_angel(width, height, min_y, max_y)
        density_diameter_90, min_y_90, max_y_90 = calculate_density_diameter(density_values, 0.9, density_smoothing)
        angel_percentile = calculate_angel(width, height, min_y_90, max_y_90)

        timestep = float(global_params.get('timestep'))

        ax1.clear()
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("z")
        ax1.set_xlim(global_params['x_dim_lo'], global_params['x_dim_hi'])
        ax1.set_ylim(global_params['y_dim_lo'], global_params['y_dim_hi'])
        ax1.set_zlim(global_params['z_dim_lo'], global_params['z_dim_hi'])
        ax1.set_title(f"{(timestep * 1e6 * frame): 6.2f} мкс")
        #ax1.view_init(180, 0)
        # graphs = separate_points_by_kn(timeframe)
        graphs = separate_points_by_density(timeframe)

        for graph in graphs:
            if len(graph['points']) > 0:
                ax1.scatter(graph['points'][0], graph['points'][1],
                            marker='.', s=0.5, color=graph['color'], label=graph['legend'])

        ax1.legend(bbox_to_anchor=(0, -0.8, 1, -0.2), loc="lower left",
                   mode="expand", borderaxespad=0, ncol=3, facecolor="darkgreen")

        fnum = float(global_params.get('fnum'))

        ax2.clear()

        show_textor_graph(data_fig9a)

        total_points = (len(uniq_points) * fnum)
        out_points = (sum_count_points * fnum)
        if len(uniq_points) == 0:
            out_percent = 0
        else:
            out_percent = (out_points / total_points) * 100

        values, labels = shift_labels_by_smoothing(density_values, density_labels, density_smoothing)

        ax2.set_title(f"Общее число частиц: {total_points:.2e} ({len(uniq_points)} модельных)\n"
                      f"Плотность на выходе: {out_points:.2e} ({sum_count_points} модельных) {out_percent:.2f}%\n"
                      f"Диаметр потока: {density_diameter * 0.1:.1f} ({density_diameter_90 * 0.1:.1f}) мм / {angel:.3f} ({angel_percentile:.3f}) градусов")
        # ax2.plot(density_values, density_labels)
        ax2.plot(labels, values, label='Modeling')

        ax2.legend()

        # plt.xlabel(f"y, {100 * density_smoothing} мкм")
        plt.xlabel(f"y, 100 мкм")
        plt.ylabel('x, ед.')
        # plt.xlim(0, int(height / density_smoothing))
        plt.xlim(0, height)
    except Exception:
        print(traceback.format_exc())

    #ax1.imshow(mask, extent=[0, mask.width, 0, mask.height])

    if frame == 200:
        plt.savefig(os.path.join(os.getcwd(), 'data', 'last_frame.png'))

    print(f'{frame} frame')


if __name__ == '__main__':
    main_dir_path = '\\\\wsl.localhost\\Ubuntu\\home\\c\\temp_sparta\\textor'
    #parts_dir_path = '\\\\wsl.localhost\\Ubuntu\\home\\c\\cache\\'
    parts_dir_path = 'D:\\temp_dumps\\'

    pick_every_timeframe = 100
    saving_to_file = 1
    density_smoothing = 8

    global_params = pars_in_file(os.path.join(main_dir_path, 'in.step'))
    width, height = global_params['width'], global_params['height']
    print(f'width = {width}, height = {height}')

    mask_template = create_mask_template(main_dir_path)
    mask = create_mask(mask_template, width, height)

    data_fig9a = pars_data(os.path.join(main_dir_path, 'dump.fig_9a.txt'))

    data_files = name_reader(parts_dir_path, 'dump.*.txt')
    density_labels, density_values, density_point_ids = [], [], dict()
    uniq_points, last_uniq_points = set(), set()

    data_files = data_files[:201]

    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), width_ratios=[3, 1], subplot_kw={'projection': '3d'})
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(1, 3, (1, 1), projection='3d')
    ax1.set_box_aspect((global_params['width'], global_params['height'], abs(global_params['z_dim_hi'] - global_params['z_dim_hi'])))
    ax2 = fig.add_subplot(1, 3, 3)

    ani = FuncAnimation(fig, update, frames=range(0, len(data_files), pick_every_timeframe), interval=1)
    FFwriter = animation.FFMpegWriter(fps=10)
    ani.save(os.path.join(os.getcwd(), 'data', 'animation.mp4'), writer=FFwriter)
