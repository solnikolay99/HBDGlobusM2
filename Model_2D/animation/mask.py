import copy
import os

from PIL import Image, ImageDraw

from animation.dump.data import *


def create_mask_template(path_main_dir: str, multiply: int = 1, set_max_surf: bool = True) -> Image:
    if 'surfs' not in gp.global_params:
        return Image.new('RGBA', (1, 1), (193, 219, 225, 255))

    surfs = []
    max_x = 0
    max_y = 0

    for surf_file in gp.global_params['surfs']:
        points, lines, polygons = pars_surf_file(os.path.join(path_main_dir, surf_file), multiply)
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

    if set_max_surf:
        gp.last_surf_x = max_x
    image = Image.new('RGBA', (max_x + 1, max_y + 1), (193, 219, 225, 255))
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


def create_mask(new_width: int, new_height: int) -> Image:
    template = create_mask_template(gp.main_dir_path)
    background = Image.new('RGBA', (new_width, new_height), (193, 219, 225, 255))
    background.paste(template)
    return background


def create_small_mask(new_width: int, new_height: int) -> Image:
    multiply = 20
    template = create_mask_template(gp.main_dir_path, multiply, False)
    background = Image.new('RGBA', (new_width * multiply, new_height * multiply), (193, 219, 225, 255))
    background.paste(template)
    back_crop = background.crop((0, round(new_height * multiply / 2) - 300, 3000, round(new_height * multiply / 2) + 300))
    #back_crop = background.crop((0, new_height * multiply - 300, 3000, new_height * multiply))
    return back_crop
