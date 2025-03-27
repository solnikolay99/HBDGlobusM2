import matplotlib
from PIL import Image

from animation.graphdata.data import *
from animation.graphdata.calculate_params import *


def show_textor_graph(ax: matplotlib.axes, data: list[list[float]]):
    values, labels = [], []
    for point in data:
        values.append(point[0])
        labels.append(point[1])
    ax.clear()
    ax.plot(values, labels, color='green', label='Textor')


def show_density_graph(ax: matplotlib.axes,
                       frame: int,
                       timeframe: list[list[float]],
                       mask: Image):
    ax.clear()

    timestep = float(gp.global_params.get('timestep'))

    ax.set_title(f"Плотность\nТаймфрейм:{(timestep * 1e6 * frame): 6.2f} мкс")
    ax.set_xticks(
        [0, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000],
        labels=[0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])
    ax.set_yticks([0, 100, 200, 300, 400], labels=[0, 1, 2, 3, 4])
    ax.set_xlabel('см')
    ax.set_ylabel('см')

    graphs = separate_points_by_density(timeframe)

    for graph in graphs:
        if len(graph['points']) > 0:
            ax.scatter(graph['points'][0], graph['points'][1],
                       marker='.', s=0.5, color=graph['color'], label=graph['legend'])

    ax.legend(bbox_to_anchor=(0, -0.5, 1, -0.1), loc="lower left",
              mode="expand", borderaxespad=0, ncol=3, facecolor="darkgreen")

    ax.imshow(mask, extent=[0, mask.width, 0, mask.height])


def show_temperature_graph(ax: matplotlib.axes,
                           frame: int,
                           timeframe: list[list[float]],
                           mask: Image):
    ax.clear()

    timestep = float(gp.global_params.get('timestep'))

    graphs = separate_points_by_temp(timeframe)

    ax.set_title(f"Температура\nТаймфрейм:{(timestep * 1e6 * frame): 6.2f} мкс")
    ax.set_xticks(
        [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000],
        labels=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])
    ax.set_yticks([0, 100, 200, 300, 400], labels=[0, 1, 2, 3, 4])
    ax.set_xlabel('см')
    ax.set_ylabel('см')

    for graph in graphs:
        if len(graph['points']) > 0:
            ax.scatter(graph['points'][0], graph['points'][1],
                       marker='.', s=0.5, color=graph['color'], label=graph['legend'])

    ax.legend(bbox_to_anchor=(0, -0.5, 1, -0.1), loc="lower left",
              mode="expand", borderaxespad=0, ncol=3, facecolor="darkgreen")

    ax.imshow(mask, extent=[0, mask.width, 0, mask.height])


def shift_labels_by_smoothing(in_values: list[int], in_labels: list[float], smoothing: int) -> (list[int], list[float]):
    out_values, out_labels = [], []
    if len(in_labels) > 0:
        out_values.append(0)
        out_labels.append(in_labels[0] * smoothing)
        for i in range(1, len(in_labels) - 1):
            out_values.append(in_values[i])
            out_labels.append(in_labels[i] * smoothing)
        out_values.append(0)
        out_labels.append(in_labels[len(in_labels) - 1] * smoothing)
    return out_values, out_labels


def show_target_graph(ax: matplotlib.axes):
    ax.clear()

    fnum = float(gp.global_params.get('fnum'))
    width, height = gp.global_params['width'], gp.global_params['height']

    density_diameter, min_y, max_y = calculate_density_diameter(gp.density_values, 1.0, gp.density_smoothing)
    angel = calculate_angel(width - gp.last_surf_x, height, min_y, max_y)
    density_diameter_90, min_y_90, max_y_90 = calculate_density_diameter(gp.density_values, 0.5, gp.density_smoothing)
    angel_percentile = calculate_angel(width - gp.last_surf_x, height, min_y_90, max_y_90)
    sum_count_points = sum(gp.density_values)

    total_points = (len(gp.uniq_points) * fnum)
    out_points = (sum_count_points * fnum)
    if len(gp.uniq_points) == 0:
        out_percent = 0
    else:
        out_percent = (out_points / total_points) * 100

    values, labels = shift_labels_by_smoothing(gp.density_values, gp.density_labels, gp.density_smoothing)

    ax.set_title(f"Общее число частиц: {total_points:.2e} ({len(gp.uniq_points)} модельных)\n"
                 f"Плотность на выходе: {out_points:.2e} ({sum_count_points} модельных) {out_percent:.2f}%\n"
                 f"Диаметр потока: {density_diameter * 0.1:.1f} ({density_diameter_90 * 0.1:.1f}) мм / {angel:.3f} ({angel_percentile:.3f}) градусов")
    ax.plot(labels, values, label='Modeling')

    max_density_value = 0
    if len(values) != 0:
        max_density_value = max(values)
    ax.plot([min_y_90, min_y_90], [0, max_density_value])
    ax.plot([max_y_90, max_y_90], [0, max_density_value])

    ax.legend()

    ax.set_xlabel('y, мм')
    ax.set_ylabel('x, ед.')
    ax.set_xlim(0, height / gp.multiplayer * 10)


def show_nozzle_graph(ax: matplotlib.axes,
                      frame: int,
                      timeframe: list[list[float]],
                      mask: Image):
    height = gp.global_params['height']
    multiply = 20
    min_x = 0
    max_x = 150
    min_y = (height / 2) - 15
    max_y = (height / 2) + 15

    ax.clear()

    timestep = float(gp.global_params.get('timestep'))

    exclude_points(timeframe, min_x, max_x, min_y, max_y, multiply)
    graphs = separate_points_by_density(timeframe)

    ax.set_title(f"Плотность\nТаймфрейм:{(timestep * 1e6 * frame): 6.2f} мкс")
    ax.set_xticks(
        [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000],
        labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    ax.set_yticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                  labels=[0, 0.25, 0.50, 0.75, 1, 1.25, 1.50, 1.75, 2, 2.25, 2.50])
    ax.set_xlabel('мм')
    ax.set_ylabel('мм')

    for graph in graphs:
        if len(graph['points']) > 0:
            ax.scatter(graph['points'][0], graph['points'][1],
                       marker='.', s=0.5, color=graph['color'], label=graph['legend'])

    ax.legend(bbox_to_anchor=(0, -1.2, 1, -0.1), loc="lower left",
              mode="expand", borderaxespad=0, ncol=2, facecolor="darkgreen")

    ax.imshow(mask, extent=[0, mask.width, 0, mask.height])
