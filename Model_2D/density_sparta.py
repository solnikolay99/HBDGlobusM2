# ******************************************************************************
# @author: L. I. Nurtdinova
# ******************************************************************************

import glob
import os
import re

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.animation import FuncAnimation

matplotlib.rcParams[
    'animation.ffmpeg_path'] = "C:\\Program Files\\ffmpeg-2024-09-26-git-f43916e217-essentials_build\\bin\\ffmpeg.exe"
multiplayer = 100


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
        points.append([round(float(params[headers.index('x')]) * multiplayer),
                       round(float(params[headers.index('y')]) * multiplayer)])

    return points, mask_width, mask_height


def draw_density(new_width: int, new_height: int, timeframe: list[list[int]]) -> Image:
    image = Image.new('RGBA', (new_width, new_height + 1), (0, 0, 0, 255))
    if len(timeframe) > 0:
        for coord in timeframe:
            pixel_coord = (coord[0], coord[1])
            pixel = list(image.getpixel(pixel_coord))
            if pixel[0] < 250:
                pixel[0] += 30
                pixel[1] += 30
                pixel[2] += 30
                image.putpixel((coord[0], coord[1]), tuple(pixel))
    image_crop = image.crop((0, round(new_height/2) - 30, 60, round(new_height/2) + 30))
    return image_crop


def update(frame):
    global width, height

    try:
        timeframe = timeframes[frame]

        timestep = float(global_params.get('timestep'))

        ax1.clear()
        ax1.set_title(f"{(timestep * 1e6 * frame): 6.2f} мкс")

        density_image = draw_density(width, height, timeframe)
        ax1.imshow(density_image)
    except Exception as exc:
        print(exc)
        pass

    print(f'{frame} frame')


if __name__ == '__main__':
    main_dir_path = '\\\\wsl.localhost\\Ubuntu\\home\\c\\sparta_build\\textor'
    parts_dir_path = main_dir_path + '\\dumps\\'

    global_params = pars_in_file(main_dir_path + '\\in.step')

    data_files = name_reader(parts_dir_path, 'dump.*.txt')

    data_files = data_files[:500]

    timeframes = []

    width, height = 0, 0
    for i in range(len(data_files)):
        if i > 0 and i % 100 == 0:
            print(f"Store data from file '{data_files[i]}'")
        timeframe, width, height = pars_data(data_files[i])
        timeframes.append(timeframe)

    fig, (ax1) = plt.subplots(1, 1, figsize=(15, 6))

    ani = FuncAnimation(fig, update, frames=range(len(timeframes)), interval=1)
    FFwriter = animation.FFMpegWriter(fps=10)
    ani.save(os.getcwd() + '/data/density.mp4', writer=FFwriter)
    # ani.save(os.getcwd() + '/data/animation.gif')
