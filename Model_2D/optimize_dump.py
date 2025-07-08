import glob
import os
import sys

import numpy as np

import animation.globparams as gp
from animation.dump.data import pars_data, pars_in_file
from animation.graphdata.calculate_params import calculate_density


def name_reader(dir_path: str, pattern):
    return sorted(glob.glob(os.path.join(dir_path, pattern)))


def get_dir_size(dir_path: str, folder: str):
    return sum(d.stat().st_size for d in os.scandir(os.path.join(dir_path, folder)) if d.is_file())


def optimize_data(f_name: str):
    lines = []
    with open(f_name) as file:
        for line in file:
            if not line.endswith(f'0 0 \n'):
                lines.append(line)

    with open(f_name, 'w') as file:
        for line in lines:
            file.write(line)


def optimize_grid_params():
    files = name_reader(gp.parts_dir_path, 'grid.*.txt')

    for i in range(len(files)):
        print(f"Pars grid file {files[i]}")
        optimize_data(files[i])

    print(f'Optimized {len(files)} files')


def pars_influx_data(f_name: str, velocities: list, uniq_points: set):
    with open(f_name) as file:
        lines = [line for line in file]

    header = lines[8]
    lines = lines[9:]

    headers = header.replace('ITEM: ATOMS ', '').strip().split(' ')
    header_id_index = headers.index('id')
    #header_vx_index = headers.index('vx')

    for line in lines:
        params = line.split(' ')
        if int(params[header_id_index]) not in uniq_points:
            uniq_points.add(int(params[header_id_index]))
            #velocities.append([float(params[header_vx_index])])


def store_influx_data(f_name: str, point_velocities: list):
    np_point_velocities = np.array(point_velocities)
    with open(f_name, 'w') as file:
        file.write(f'Min velocity: {int(np.min(np_point_velocities))}\n')
        file.write(f'Max velocity: {int(np.max(np_point_velocities))}\n')
        file.write(f'Mean velocity: {int(np.mean(np_point_velocities))}\n')


def store_influx_ids_data(f_name: str, count_points: int):
    with open(f_name, 'w') as file:
        file.write(f'Count points: {count_points}\n')


def optimize_influx():
    files = name_reader(gp.parts_dir_path, 'influx.*.txt')

    point_velocities = []
    uniq_point_ids = set()
    for i in range(len(files)):
        pars_influx_data(files[i], point_velocities, uniq_point_ids)
        if i % 100 == 0 and i > 0:
            print(f"Pars influx file {files[i]}")
            #store_influx_data(files[i].replace('influx.', 'influx_sum.'), point_velocities)
            store_influx_ids_data(files[i].replace('influx.', 'influx_sum.'), len(uniq_point_ids))
        os.remove(files[i])


def store_target_data(f_name: str, density_labels: list, density_values: list):
    with open(f_name, 'w') as file:
        for i in range(len(density_labels)):
            file.write(f'{density_labels[i]} {density_values[i]}\n')


def optimize_target():
    files = name_reader(gp.parts_dir_path, 'target.*.txt')

    for i in range(len(files)):
        target_points = pars_data(files[i])
        gp.density_labels, gp.density_values, gp.density_point_ids = calculate_density(gp.global_params['width'],
                                                                                       gp.global_params['height'],
                                                                                       target_points,
                                                                                       gp.density_labels,
                                                                                       gp.density_values,
                                                                                       gp.density_point_ids,
                                                                                       gp.density_smoothing)

        if i % 100 == 0 and i > 0:
            print(f"Pars target file {files[i]}")
            store_target_data(files[i].replace('target.', 'target_sum.'), gp.density_labels, gp.density_values)
        #os.remove(files[i])


if __name__ == '__main__':
    gp.global_params = pars_in_file(os.path.join(gp.main_dir_path, 'in.step' if len(sys.argv) < 2 else sys.argv[1]))
    #gp.density_smoothing = 8 if len(sys.argv) < 3 else float(sys.argv[2])

    print(f'Folder size before optimization: {(get_dir_size(gp.parts_dir_path, "/") / 1024 ** 3):.2f} GB')

    #optimize_grid_params()
    #optimize_influx()
    optimize_target()

    print(f'Folder size after optimization: {(get_dir_size(gp.parts_dir_path, "/") / 1024 ** 3):.2f} GB')
