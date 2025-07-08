from animation.graphdata.calculate_params import *


def separate_points_by_kn(timeframe: list[list[float]]) -> list[dict[str, list[list[float]]]]:
    cmap = {
        '#0000ff': 'Kn < 0.001',
        '#ff531f': '0.001 < Kn < 1',
        '#ffffff': '1 <= Kn',
    }

    out_graph = [{
        'color': key,
        'legend': cmap[key],
        'points': [[], []]
    } for key in cmap]

    grid = dict()
    for point in timeframe:
        key = f'{int(point[0] / 50)}_{int(point[1] / 50)}'
        if key not in grid:
            grid[key] = [[], []]
        grid[key][0].append(point[0])
        grid[key][1].append(point[1])

    separators = [0, 0.001, 1, 1e6]
    for key in grid.keys():
        kn = calculate_kn(grid[key])
        for i in range(len(separators) - 1):
            if separators[i] <= kn < separators[i + 1]:
                for point in grid[key][0]:
                    out_graph[0]['points'][0].append(point)
                for point in grid[key][1]:
                    out_graph[0]['points'][1].append(point)
                break

    return out_graph


def separate_points_by_density(timeframe: list[list[float]]) -> list[dict[str, list[list[float]]]]:
    fnum = float(gp.global_params.get('fnum'))
    # fnum = 1e0

    cmap = {
        '#ff531f': f'N <  {2 * fnum:.1e}',
        '#00A2E8': f'{2 * fnum:.1e} ≤ N < {4 * fnum:.1e}',
        '#0000ff': f'{4 * fnum:.1e} ≤ N < {6 * fnum:.1e}',
        '#000000': f'{6 * fnum:.1e} ≤ N < {8 * fnum:.1e}',
        '#8fbc8f': f'{8 * fnum:.1e} ≤ N < {10 * fnum:.1e}',
        '#808000': f'{10 * fnum:.1e} ≤ N < {12 * fnum:.1e}',
        '#9acd32': f'{12 * fnum:.1e} ≤ N < {14 * fnum:.1e}',
        '#ffffff': f'{14 * fnum:.1e} ≤ N < {16 * fnum:.1e}',
        '#00ff00': f'N ≥ {16 * fnum:.1e}',
    }

    out_graph = [{
        'color': key,
        'legend': cmap[key],
        'points': [[], []]
    } for key in cmap]

    count_set = 0
    count_errors = 0
    uniq_cells = set()
    for point in timeframe:
        if len(point) == 0:
            continue

        if int(point[3]) in uniq_cells:
            continue
        else:
            uniq_cells.add(int(point[3]))

        try:
            cell_param = gp.grid_params[int(point[3])]
            po = int(cell_param[0] / 2) if int(cell_param[0] / 2) < len(out_graph) else len(out_graph) - 1
            count_set += 1
        except:
            po = 0
            count_errors += 1
        out_graph[po]['points'][0].append(point[0])
        out_graph[po]['points'][1].append(point[1])

    if count_errors > 0:
        print(f'Count points with (without) density: {count_set} ({count_errors})')

    return out_graph


def separate_points_by_temp(timeframe: list[list[float]]) -> list[dict[str, list[list[float]]]]:
    cmap = {
        '#ff531f': f'N <  {50}',
        '#00A2E8': f'{50} ≤ N < {100}',
        '#0000ff': f'{100} ≤ N < {150}',
        '#000000': f'{150} ≤ N < {200}',
        '#8fbc8f': f'{200} ≤ N < {250}',
        '#808000': f'{250} ≤ N < {300}',
        '#9acd32': f'{300} ≤ N < {400}',
        '#ffffff': f'{400} ≤ N < {500}',
        '#00ff00': f'N ≥ {500}',
    }

    out_graph = [{
        'color': key,
        'legend': cmap[key],
        'points': [[], []]
    } for key in cmap]

    uniq_cells = set()
    separators = [0, 50, 100, 150, 200, 250, 300, 400, 500, 1e6]
    for point in timeframe:
        if len(point) == 0:
            continue

        if int(point[3]) in uniq_cells:
            continue
        else:
            uniq_cells.add(int(point[3]))

        try:
            cell_param = gp.grid_params[int(point[3])]
            for i in range(len(separators) - 1):
                if separators[i] <= cell_param[1] < separators[i + 1]:
                    out_graph[i]['points'][0].append(point[0])
                    out_graph[i]['points'][1].append(point[1])
                    break
        except:
            out_graph[0]['points'][0].append(point[0])
            out_graph[0]['points'][1].append(point[1])

    return out_graph


def adjust_points_by_temp(timeframe: list[list[float]]) -> list[list[float]]:
    for point in timeframe:
        if len(point) == 0:
            continue
        try:
            cell_param = gp.grid_params[int(point[3])]
            point.append(cell_param[1])
        except:
            point.append(0)

    return timeframe


def exclude_points(timeframe: list[list[float]],
                   min_x: int,
                   max_x: int,
                   min_y: int,
                   max_y: int,
                   multiply: int):
    for i in range(len(timeframe) - 1, -1, -1):
        if len(timeframe[i]) == 0:
            continue
        if min_x <= timeframe[i][0] <= max_x and min_y <= timeframe[i][1] <= max_y:
            timeframe[i][0] = (timeframe[i][0] - min_x) * multiply
            timeframe[i][1] = (timeframe[i][1] - min_y) * multiply
            continue
        del timeframe[i]
