# ******************************************************************************
# @author: L. I. Nurtdinova
# ******************************************************************************

import glob

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

multiplayer = 100


def name_reader(dir_path, pattern):
    return sorted(glob.glob(dir_path + '\\' + pattern))


def pars_data(f_name: str, points: dict[int, list[list[float]]]) -> (dict[int, list[list[float]]], int, int):
    with open(f_name) as file:
        lines = [line.rstrip() for line in file]
    mask_width = int(float(lines[5].split(' ')[1]) * multiplayer)
    mask_height = int(float(lines[6].split(' ')[1]) * multiplayer)
    header = lines[8]
    lines = lines[9:]

    headers = header.replace('ITEM: ATOMS ', '').strip().split(' ')
    for line in lines:
        params = line.strip().split(' ')
        if int(params[headers.index('id')]) in points:
            points[int(params[headers.index('id')])] \
                .append([float(params[headers.index('x')]) * multiplayer,
                         float(params[headers.index('y')]) * multiplayer])
        else:
            points[int(params[headers.index('id')])] = [[float(params[headers.index('x')]) * multiplayer,
                                                         float(params[headers.index('y')]) * multiplayer]]

    return points, mask_width, mask_height


def pars_surf_file(f_name: str) -> (dict[int, dict[str, int]], dict[int, dict[str, int]]):
    surf_points = dict()
    surf_lines = dict()

    with open(f_name) as file:
        f_lines = [line.rstrip() for line in file]

    count_points = int(f_lines[2].strip().split(' ')[0])
    count_lines = int(f_lines[3].strip().split(' ')[0])

    for i in range(7, 7 + count_points):
        point = f_lines[i].strip().split(' ')
        surf_points[int(point[0])] = {'x': int(float(point[1]) * multiplayer), 'y': int(float(point[2]) * multiplayer)}
    for i in range(10 + count_points, 10 + count_points + count_lines):
        line = f_lines[i].strip().split(' ')
        surf_lines[int(line[0])] = {'from': int(line[1]), 'to': int(line[2])}

    return surf_points, surf_lines


def create_mask_template(data_file: str) -> Image:
    mask_points, mask_lines = pars_surf_file(data_file)

    max_x = 0
    max_y = 0
    for key in mask_points.keys():
        if mask_points[key]['x'] > max_x:
            max_x = mask_points[key]['x']
        if mask_points[key]['y'] > max_y:
            max_y = mask_points[key]['y']

    image = Image.new('RGBA', (max_x + 1, max_y + 1), (0, 100, 0, 255))
    draw = ImageDraw.Draw(image)

    for key in mask_lines.keys():
        point_from = mask_points[mask_lines[key]['from']]
        point_to = mask_points[mask_lines[key]['to']]
        draw.line((point_from['x'], point_from['y'], point_to['x'], point_to['y']), fill=255)

    return image


def create_mask(template: Image, new_width: int, new_height: int) -> Image:
    background = Image.new('RGBA', (new_width, new_height), (0, 100, 0, 255))
    background.paste(template)
    return background


def remove_elements(points: dict[int, list[list[float]]], width: int) -> dict[int, list[list[float]]]:
    keys = list(points.keys())
    for key in keys:
        list_points = points[key]
        flg_remove = True
        end_index = len(list_points) - 10
        if end_index < 0:
            end_index = 0
        for i in range(len(list_points) - 1, end_index, -1):
            if list_points[i][0] > width - 10:
                flg_remove = False
                break
        if flg_remove:
            del points[key]
    return points


def reduce_elements(points: dict[int, list[list[float]]], count_elements: int) -> dict[int, list[list[float]]]:
    keys = list(points.keys())
    keys = keys[count_elements:]
    for key in keys:
        del points[key]
    return points


if __name__ == '__main__':
    main_dir_path = '\\\\wsl.localhost\\Ubuntu\\home\\c\\sparta_build\\textor'
    parts_dir_path = main_dir_path + '\\dumps\\'

    data_files = name_reader(parts_dir_path, 'dump.*.txt')

    points = dict()

    width, height = 0, 0
    for i in range(0, len(data_files), 1):
        print(f"Store data from file '{data_files[i]}'")
        points, width, height = pars_data(data_files[i], points)

    #points = remove_elements(points, width)
    points = reduce_elements(points, 1000)

    mask_template = create_mask_template(main_dir_path + '\\data.step')
    mask = create_mask(mask_template, width, height)

    plt.figure(figsize=(5, 5))
    for key in points.keys():
        coord_x = np.array(points[key])[:, 0]
        coord_y = np.array(points[key])[:, 1]
        plt.plot(coord_x, coord_y, linewidth=0.5)
    plt.imshow(mask)
    plt.grid(color='black', linestyle='-', linewidth=0.2)

    if False:
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
