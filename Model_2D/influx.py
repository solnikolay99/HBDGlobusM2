import glob
import os
import numpy as np


def name_reader(dir_path, pattern):
    return sorted(glob.glob(os.path.join(dir_path, pattern)))


def pars_data(f_name: str, velocities: list, uniq_points: set):
    with open(f_name) as file:
        lines = [line for line in file]

    header = lines[8]
    lines = lines[9:]

    headers = header.replace('ITEM: ATOMS ', '').strip().split(' ')
    header_id_index = headers.index('id')
    header_vx_index = headers.index('vx')
    #header_vy_index = headers.index('vy')

    for line in lines:
        params = line.split(' ')
        if int(params[header_id_index]) not in uniq_points:
            uniq_points.add(int(params[header_id_index]))
            velocities.append([abs(float(params[header_vx_index]))])


if __name__ == '__main__':
    data_files = name_reader('D:\\dumps\\', 'influx.*.txt')

    #data_files = data_files[:1001]

    point_velocities = []
    uniq_point_ids = set()
    for i in range(len(data_files)):
        if i % 100 == 0 and i > 0:
            print(f"Pars influx file {data_files[i]}")
        pars_data(data_files[i], point_velocities, uniq_point_ids)

    print(len(point_velocities))

    np_point_velocities = np.array(point_velocities)
    print(f'Min velocity {int(np.min(np_point_velocities))}')
    print(f'Max velocity {int(np.max(np_point_velocities))}')
    print(f'Mean velocity {int(np.mean(np_point_velocities))}')
