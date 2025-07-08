import math
import traceback
import uuid

import animation.globparams as gp


def calculate_density(width: int,
                      height: int,
                      timeframe: list[list[float]],
                      labels: list[float],
                      densities: list[int],
                      point_ids: dict[int, dict[str, int]],
                      smoothing: int) -> (list[float], list[int], dict[int, dict[str, int]]):
    if len(labels) < 1:
        labels = [i for i in range(int(height / smoothing) + 1)]
    if len(densities) < 1:
        densities = [0 for _ in range(int(height / smoothing) + 1)]

    seed = uuid.uuid4()
    for point in timeframe:
        # if width - 1 >= point[0] >= width - 2:
        #    densities[int(point[1])] += 1
        if point[0] > width - 10:
            point_ids[int(point[2])] = {'y': int(point[1]), 'seed': seed}

    keys = list(point_ids.keys())
    for key in keys:
        try:
            if point_ids[key]['seed'] != seed:
                densities[int(point_ids[key]['y'] / smoothing)] += 1
                del point_ids[key]
        except Exception:
            print(traceback.format_exc())
            print(f'key = {key}')

    return labels, densities, point_ids


def calculate_density2(width: int,
                       height: int,
                       timeframe: list[list[float]],
                       labels: list[float],
                       densities: list[int],
                       point_ids: dict[int, dict[str, int]],
                       smoothing: int) -> (list[float], list[int], dict[int, dict[str, int]]):
    if len(labels) < 1:
        labels = [i for i in range(int(height / smoothing) + 1)]
    if len(densities) < 1:
        densities = [0 for _ in range(int(height / smoothing) + 1)]

    seed = uuid.uuid4()
    for point in timeframe:
        # if width - 1 >= point[0] >= width - 2:
        #    densities[int(point[1])] += 1
        if point[0] > width - 10:
            point_ids[int(point[2])] = {'y': int(point[1]), 'seed': seed}

    keys = list(point_ids.keys())
    for key in keys:
        try:
            if point_ids[key]['seed'] != seed:
                densities[int(point_ids[key]['y'] / smoothing)] += 1
                del point_ids[key]
        except Exception:
            print(traceback.format_exc())
            print(f'key = {key}')

    return labels, densities, point_ids


def calculate_density_diameter(densities: list[int], percentile: float, reducer: int) -> (float, float, float):
    min_y = 0
    max_y = len(densities)

    if len(densities) == 0:
        return 0.0, 0.0, 0.0

    densities_wo_zeros = [x for x in densities if x != 0]
    min_value = min(densities_wo_zeros) if len(densities_wo_zeros) > 0 else 0
    if percentile < 1:
        min_value = max(densities) * (1 - percentile)
    if min_value == 0:
        return 0.0, 0.0, 0.0

    for i in range(1, int(len(densities) / 2)):
        if densities[i] > min_value:
            break
        else:
            min_y = i
    for i in range(len(densities) - 3, int(len(densities) / 2), -1):
        if densities[i] > min_value:
            break
        else:
            max_y = i
    if min_y < max_y:
        return (max_y - min_y) * reducer, min_y * reducer, max_y * reducer
    else:
        return 0.0, 0.0, 0.0


def calculate_angel(length: float, width: float, min_y: float, max_y: float) -> float:
    if min_y == 0.0 and max_y == 0.0:
        return 0.0
    max_r = max(abs(width * 0.5 - min_y), abs(width * 0.5 - max_y))
    return math.degrees(math.atan(max_r / length))


def calculate_kn(cell_points: list[list[float]]) -> float:
    fnum = float(gp.global_params.get('fnum'))
    sigma = 2.33e-8
    L = 5e-3
    n = (len(cell_points) * fnum) / (L * L * 1)
    Kn = 1 / (math.sqrt(2) * sigma * sigma * n) / L
    return Kn
