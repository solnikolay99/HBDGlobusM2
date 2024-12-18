import re
import os

import matplotlib.pyplot as plt
from PIL import Image

multiplayer = 100
multi_x = 650/400
multi_y = 300/100
multi_y = 550/164
#multi_x = 1
#multi_y = 1


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
    return out_params


def pars_data(f_name: str) -> list[list[float]]:
    with open(f_name) as file:
        lines = [line.rstrip() for line in file]
    header = lines[8]
    lines = lines[9:]

    headers = header.replace('ITEM: ATOMS ', '').strip().split(' ')
    points = []
    for line in lines:
        params = line.strip().split(' ')
        if len(params) < headers.index('y') + 1:
            continue
        points.append([float(params[headers.index('x')]) * multiplayer * multi_x,
                       float(params[headers.index('y')]) * multiplayer * multi_y,
                       int(params[headers.index('id')])])

    return points


def show_textor_graph(data: list[list[float]]):
    back_image = Image.open('\\\\wsl.localhost\\Ubuntu\\home\\c\\sparta_git\\textor\\Untitled.png')

    values, labels = [], []
    for point in data:
        values.append(point[0])
        labels.append(point[1])
        print(f"{point[2]}: x = {point[0]}; y = {point[1]}")

    ax2.clear()
    ax2.set_title(f"Fig 9a")
    ax2.plot(values, labels, color='green')

    ax2.imshow(back_image, extent=[0, back_image.width, 0, back_image.height])

    plt.xlabel(f"y, {100 * 8} мкм")
    plt.ylabel('x, ед.')
    #plt.xlim(0, 50)


if __name__ == '__main__':
    main_dir_path = '\\\\wsl.localhost\\Ubuntu\\home\\c\\sparta_git\\textor'
    parts_dir_path = main_dir_path + '\\dumps\\'

    global_params = pars_in_file(main_dir_path + '\\in.step')

    data_fig9a = pars_data(main_dir_path + '\\dump.fig_9a.txt')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), width_ratios=[1, 2])

    show_textor_graph(data_fig9a)

    plt.savefig(os.getcwd() + '/data/fig9a.png')
