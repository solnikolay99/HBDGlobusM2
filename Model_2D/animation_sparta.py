# ******************************************************************************
# @author: L. I. Nurtdinova
# ******************************************************************************
import glob
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from animation.graphs import *
from animation.mask import *

if sys.platform == 'win32':
    matplotlib.rcParams[
        'animation.ffmpeg_path'] = "C:\\Program Files\\ffmpeg-2024-09-26-git-f43916e217-essentials_build\\bin\\ffmpeg.exe"
matplotlib.rcParams['legend.markerscale'] = 3


def name_reader(dir_path, pattern):
    return sorted(glob.glob(os.path.join(dir_path, pattern)))


def get_frame_data(frame: int):
    start_gfd = adl.start_timer()

    if frame == 0:
        return [], [], []

    if len(gp.density_labels) < 1:
        gp.density_labels = [i for i in range(int(height / gp.density_smoothing) + 1)]
    if len(gp.density_values) < 1:
        gp.density_values = [0 for _ in range(int(height / gp.density_smoothing) + 1)]

    timeframe = pars_data2(data_files[frame])
    frame_name = re.sub(r"(.+\.)([0-9]+)(\.txt)", r"\2", data_files[frame])
    '''
    density_file_name = ''
    for density_file in density_files:
        if f'{frame_name}' in density_file:
            density_file_name = density_file
            break
    gp.density_values = pars_density(density_file_name, gp.density_values)
    '''

    '''
    for i in range(100):
        target_file_name = target_files.pop(0)
        target_points = pars_data(target_file_name)
        gp.density_labels, gp.density_values, gp.density_point_ids = calculate_density(width,
                                                                                       height,
                                                                                       target_points,
                                                                                       gp.density_labels,
                                                                                       gp.density_values,
                                                                                       gp.density_point_ids,
                                                                                       gp.density_smoothing)
    '''

    if frame != 0:
        gp.density_labels, gp.density_values = pars_target(target_files.pop(0))

    grid_params_file_name = ''
    for grid_file in grid_files:
        if f'.{frame_name}' in grid_file:
            grid_params_file_name = grid_file
            break
    gp.grid_params = pars_grid_params(grid_params_file_name)

    for point in timeframe:
        gp.uniq_points.add(int(point[2]))

    adl.release_timer('Get frame data', start_gfd)

    print(f"Stored data from file '{data_files[frame]}'")

    return timeframe


def get_only_nonzero_labels(in_values: list[int], in_labels: list[float]) -> (list[int], list[float]):
    out_values, out_labels = [], []
    for i in range(len(in_values)):
        if in_values[i] > 0:
            out_values.append(in_values[i])
            out_labels.append(in_labels[i])
    return out_values, out_labels


def update(frame):
    # start_update = adl.start_timer()

    try:
        timeframe = get_frame_data(frame)
        show_density_graph(ax11, frame, timeframe, mask)
        show_temperature_graph(ax21, frame, timeframe, mask)
        show_target_graph(ax12)
        show_nozzle_temperature_graph(ax22, frame, timeframe, small_mask)
    except Exception:
        print(traceback.format_exc())

    frames = [10, 20, 50, 200, 240]
    for i in range(len(frames)):
        if frames[i] == frame:
            plt.savefig(os.path.join(os.getcwd(), 'data', f'last_frame_{i + 1}.png'))

    # adl.release_timer("Update iteration", start_update)
    print(f'{frame} frame')


if __name__ == '__main__':
    # pick_every_timeframe = 100
    pick_every_timeframe = 1

    start_time = adl.start_timer()

    gp.global_params = pars_in_file(os.path.join(gp.main_dir_path, 'in.step' if len(sys.argv) < 2 else sys.argv[1]))
    width, height = gp.global_params['width'], gp.global_params['height']
    print(f'width = {width}, height = {height}')

    mask = create_mask(width, height)
    small_mask = create_small_mask(width, height)

    # data_fig9a = pars_data(os.path.join(main_dir_path, 'dump.fig_9a.txt'))

    data_files = name_reader(gp.parts_dir_path, 'dump.*.txt')
    # density_files = name_reader(gp.parts_dir_path, 'density.*.txt')
    # target_files = name_reader(gp.parts_dir_path, 'target.*.txt')
    target_files = name_reader(gp.parts_dir_path, 'target_sum.*.txt')
    grid_files = name_reader(gp.parts_dir_path, 'grid.*.txt')

    #data_files = data_files[:3]

    px = 1 / plt.rcParams['figure.dpi']
    fig = plt.figure(layout="constrained", figsize=(1900 * px, 1000 * px))
    axd = fig.subplot_mosaic(
        """
        AC
        BD
        """,
        # height_ratios=[1, 2],
        width_ratios=[2, 1],
    )
    ax11 = axd['A']
    ax21 = axd['B']
    ax12 = axd['C']
    ax22 = axd['D']

    ani = FuncAnimation(fig, update, frames=len(data_files), interval=1)
    FFwriter = animation.FFMpegWriter(fps=10)
    ani.save(os.path.join(os.getcwd(), 'data', 'animation.mp4'), writer=FFwriter)

    adl.release_timer('Animation time', start_time)

    print(f'Max used cells = {gp.max_used_cells}')
