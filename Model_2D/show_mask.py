# ******************************************************************************
# @author: L. I. Nurtdinova
# ******************************************************************************
import copy
import os
import re
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

multiplayer = 1000


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
        elif params[0].strip() == 'region':
            if params[2].strip() == 'cylinder':
                y = float(params[4].strip())
                r = float(params[6].strip())
                out_params['region_y_lo'] = f"{y - r}"
                out_params['region_y_hi'] = f"{y + r}"
                out_params['region_x_lo'] = params[7].strip()
                out_params['region_x_hi'] = params[8].strip()
            elif params[2].strip() == 'block':
                out_params['region_x_lo'] = params[3].strip()
                out_params['region_x_hi'] = params[4].strip()
                out_params['region_y_lo'] = params[5].strip()
                out_params['region_y_hi'] = params[6].strip()
        elif params[0].strip() == 'read_surf':
            if 'surfs' not in out_params:
                out_params['surfs'] = []
            out_params['surfs'].append(params[1].strip())
    return out_params


def pars_surf_file(f_name: str) -> (dict[int, dict[str, int]], dict[int, dict[str, int]], list[dict[int]]):
    points = dict()
    lines = dict()

    with open(main_dir_path + '\\' + f_name) as file:
        f_lines = [line.rstrip() for line in file]

    count_points = int(f_lines[2].strip().split(' ')[0])
    count_lines = int(f_lines[3].strip().split(' ')[0])
    polygons = []
    add_new_polygon = True

    for i in range(7, 7 + count_points):
        point = f_lines[i].strip().split(' ')
        points[int(point[0])] = {'x': round(float(point[1]) * multiplayer), 'y': round(float(point[2]) * multiplayer)}
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


def create_mask_template(in_file: str) -> (Image, int):
    surfs = []
    max_x = 0
    max_y = 0

    global_params = pars_in_file(in_file)
    for surf_file in global_params['surfs']:
        points, lines, polygons = pars_surf_file(surf_file)
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

    image = Image.new('RGBA', (max_x + 1, max_y + 1), (255, 255, 255, 255))
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

    if 'region_y_lo' in global_params:
        x_lo = float(global_params['region_x_lo']) * multiplayer
        x_hi = float(global_params['region_x_hi']) * multiplayer
        y_lo = float(global_params['region_y_lo']) * multiplayer
        y_hi = float(global_params['region_y_hi']) * multiplayer
        draw.line((x_lo, y_lo, x_hi, y_lo), fill=(255, 0, 0, 255))
        draw.line((x_lo, y_hi, x_hi, y_hi), fill=(255, 0, 0, 255))
        draw.line((x_lo, y_lo, x_lo, y_hi), fill=(255, 0, 0, 255))
        draw.line((x_hi, y_lo, x_hi, y_hi), fill=(255, 0, 0, 255))

    for point in real_points:
        x = int(point[0] * multiplayer)
        y = int(point[1] * multiplayer)
        draw.point((x, y), fill=(0, 255, 0, 255))

    return image, max_x


def create_mask(template: Image, new_width: int, new_height: int, max_x: int) -> Image:
    background = Image.new('RGBA', (max(new_width, max_x + 50), new_height), (255, 255, 255, 255))
    background.paste(template)
    back_crop = background.crop((0, round(new_height / 2) - 60, max_x, round(new_height / 2) + 60))
    #back_crop = background.crop((0, 0, max_x, 60))
    return back_crop


def save_mask(file_name: str):
    mask_tmpl, max_x = create_mask_template(main_dir_path + '\\' + file_name)
    max_x += 50
    mask_tmpl = create_mask(mask_tmpl, 3 * multiplayer, 4 * multiplayer, 120)
    fig, (ax1) = plt.subplots(1, 1, figsize=(15, 6))

    plt.xlabel('мм')
    plt.ylabel('мм')
    x_ticks = np.arange(0, mask_tmpl.width + 1, 500)
    x_labels = [f'{int(x / 100)}' for x in x_ticks]
    ax1.set_xticks(x_ticks, labels=x_labels)
    y_ticks = np.arange(0, mask_tmpl.height + 1, 500)
    y_labels = [f'{int(y / 100)}' for y in y_ticks]
    ax1.set_yticks(y_ticks, labels=y_labels)
    ax1.imshow(mask_tmpl, extent=[0, mask_tmpl.width, 0, mask_tmpl.height])

    #mask_tmpl.save(os.getcwd() + '/data/mask_tmpl.png')
    plt.savefig(os.getcwd() + '/data/mask_tmpl.png')


if __name__ == '__main__':
    main_dir_path = '\\\\wsl.localhost\\Ubuntu\\home\\c\\sparta_git\\textor'

    real_points = [
        #[0.004177, 1.997785, 0.000000],
        #[0.009604, 1.996509, 0.000000],
        #[0.012529, 1.995110, 0.000000],
        #[0.004183, 2.001787, 0.000000],
        #[0.007708, 2.000062, 0.000000],
    ]

    save_mask('in.step')
    #save_mask('in.step.temp')
