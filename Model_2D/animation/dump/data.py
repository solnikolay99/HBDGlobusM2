import re

import animation.debug.logger as adl
import animation.globparams as gp


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
        elif params[0].strip() == 'units':
            gp.unit_system_CGS = (params[1].strip() == 'cgs')
            if not gp.unit_system_CGS:
                gp.multiplayer *= 100
                gp.density_smoothing *= 10
        elif params[0].strip() == 'timestep':
            out_params['timestep'] = params[1].strip()
        elif params[0].strip() == 'create_box':
            out_params['width'] = int((float(params[2].strip()) - float(params[1].strip())) * gp.multiplayer)
            out_params['height'] = int((float(params[4].strip()) - float(params[3].strip())) * gp.multiplayer)
        elif params[0].strip() == 'create_grid':
            out_params['width_cells'] = int(params[1].strip())
            out_params['height_cells'] = int(params[2].strip())
        elif params[0].strip() == 'read_surf':
            if 'surfs' not in out_params:
                out_params['surfs'] = []
            out_params['surfs'].append(params[1].strip())
    return out_params


def pars_data(f_name: str) -> list[list[float]]:
    start_read_file = adl.start_timer()
    with open(f_name) as file:
        lines = [line for line in file]
    # adl.release_timer("Read data from file", start_read_file)

    header = lines[8]
    lines = lines[9:]

    headers = header.replace('ITEM: ATOMS ', '').strip().split(' ')
    header_x_index = headers.index('x')
    header_y_index = headers.index('y')
    header_id_index = headers.index('id')
    if 'c_pTemp[*]' in headers:
        header_temp_index = headers.index('c_pTemp[*]')
    else:
        header_temp_index = 0

    start_pars_data = adl.start_timer()
    points = []
    for line in lines:
        params = line.split(' ')
        points.append([float(params[header_x_index]) * gp.multiplayer,
                       float(params[header_y_index]) * gp.multiplayer,
                       int(params[header_id_index]),
                       float(params[header_temp_index]) * gp.kB1])
    # adl.release_timer("Pars file data", start_pars_data)

    return points


def pars_data2(f_name: str) -> list[list[float]]:
    start_read_file = adl.start_timer()
    with open(f_name) as file:
        lines = [line for line in file]
    adl.release_timer("Read data from file", start_read_file)

    header = lines[8]
    lines = lines[9:]

    headers = header.replace('ITEM: ATOMS ', '').strip().split(' ')
    header_x_index = headers.index('x')
    header_y_index = headers.index('y')
    header_id_index = headers.index('id')
    header_cell_id_index = headers.index('cellID')

    start_pars_data = adl.start_timer()
    points = []
    for line in lines:
        params = line.split(' ')
        points.append([float(params[header_x_index]) * gp.multiplayer,
                       float(params[header_y_index]) * gp.multiplayer,
                       int(params[header_id_index]),
                       int(params[header_cell_id_index])])
    adl.release_timer("Pars file data", start_pars_data)

    return points


def pars_surf_file(f_name: str, multiply: int = 1) -> (dict[int, dict[str, int]], dict[int, dict[str, int]]):
    print(f"Pars surf file {f_name}")

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
        points[int(point[0])] = {'x': int(float(point[1]) * gp.multiplayer * multiply),
                                 'y': int(float(point[2]) * gp.multiplayer * multiply)}
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


def pars_density(f_name: str, densities: list[int]) -> list[int]:
    print(f"Pars density file {f_name}")

    with open(f_name) as file:
        lines = [line for line in file]

    for line in lines:
        params = line.split(' ')
        densities[int(params[0])] = int(params[1])

    return densities


def pars_grid(f_name: str) -> list[list[float]]:
    print(f"Pars grid cells file {f_name}")

    with open(f_name) as file:
        lines = [line for line in file]

    grid: list[list[float]] = [[] for _ in range(len(lines) - 9 + 1)]

    header = lines[8]
    lines = lines[9:]

    headers = header.replace('ITEM: CELLS ', '').strip().split(' ')
    header_id_index = headers.index('id')
    header_xlo_index = headers.index('xlo')
    header_ylo_index = headers.index('ylo')
    header_xhi_index = headers.index('xhi')
    header_yhi_index = headers.index('yhi')

    for line in lines:
        params = line.split(' ')
        grid[int(params[header_id_index])] = [
            float(params[header_xlo_index]) * gp.multiplayer,
            float(params[header_ylo_index]) * gp.multiplayer,
            float(params[header_xhi_index]) * gp.multiplayer,
            float(params[header_yhi_index]) * gp.multiplayer,
        ]

    return grid


def pars_grid_params(f_name: str) -> dict[int, list[float]]:
    print(f"Pars grid params file {f_name}")

    with open(f_name) as file:
        lines = [line for line in file]

    grid_params: dict[int, list[float]] = dict()

    header = lines[8]
    lines = lines[9:]

    headers = header.replace('ITEM: CELLS ', '').strip().split(' ')
    header_id_index = headers.index('id')
    #header_p_index = headers.index('c_gTemp[1]')
    #header_t_index = headers.index('c_gTemp[2]')
    header_p_index = headers.index('f_aveGridTemp[1]')
    header_t_index = headers.index('f_aveGridTemp[2]')

    for line in lines:
        if line.endswith(f'0 0 \n'):
            continue
        params = line.split(' ')
        p = float(params[header_p_index])
        temp = float(params[header_t_index])
        if p > 0 and temp > 0:
            cell_id = int(params[header_id_index])
            grid_params[cell_id] = [p, temp]

    return grid_params


def pars_target(f_name: str) -> (list[int], list[int]):
    with open(f_name) as file:
        lines = [line for line in file]

    density_labels, density_values = [], []
    for line in lines:
        params = line.split(' ')
        density_labels.append(int(params[0]) * gp.density_smoothing)
        density_values.append(int(params[1]))

    return density_labels, density_values
