# ******************************************************************************
# @author: L. I. Nurtdinova
# ******************************************************************************
import os

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import re

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
    return out_params


def pars_surf_file(f_name: str) -> (dict[int, dict[str, int]], dict[int, dict[str, int]], list[dict[int]]):
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


def create_mask_template(in_file: str, data_file: str) -> (Image, int):
    global_params = pars_in_file(in_file)
    points, lines, polygons = pars_surf_file(data_file)

    max_x = 0
    max_y = 0
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

    for polygon in polygons:
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

    return image, max_x


def create_mask(template: Image, new_width: int, new_height: int, max_x: int) -> Image:
    background = Image.new('RGBA', (max(new_width, max_x + 50), new_height), (255, 255, 255, 255))
    background.paste(template)
    back_crop = background.crop((0, round(new_height / 2) - 60, max_x, round(new_height / 2) + 60))
    return back_crop


def save_mask():
    mask_tmpl, max_x = create_mask_template(main_dir_path + '\\in.step', main_dir_path + '\\data.step')
    max_x += 50
    mask_tmpl = create_mask(mask_tmpl, 2 * multiplayer, 4 * multiplayer, 120)
    fig, (ax1) = plt.subplots(1, 1, figsize=(15, 6))

    plt.xlabel('мкм')
    plt.ylabel('мкм')
    # plt.xlim(0, 10)
    # plt.ylim(0, height)
    ax1.imshow(mask_tmpl)

    #mask_tmpl.save(os.getcwd() + '/data/mask_tmpl.png')
    plt.savefig(os.getcwd() + '/data/mask_tmpl.png')


if __name__ == '__main__':
    main_dir_path = '\\\\wsl.localhost\\Ubuntu\\home\\c\\sparta_build\\textor'
    save_mask()
