#main_dir_path = '\\\\wsl.localhost\\Ubuntu\\home\\c\\sparta_git\\textor'
main_dir_path = 'E:\\dumps_mgiv_ci_1atm_wo_control_P\\'
#parts_dir_path = '\\\\wsl.localhost\\Ubuntu\\home\\c\\cache\\'
parts_dir_path = 'E:\\dumps_mgiv_ci_1atm_wo_control_P\\'

unit_system_CGS = True
multiplayer = 100
kB1 = 1 / 1.38067e-16
global_params = dict[str, any]
max_used_cells = 0
last_surf_x = 0

grid_params: dict[int, list[float]] = dict()

density_smoothing = 0.8
density_labels, density_values, density_point_ids = [], [], dict()
uniq_points = set()
data_fig9a = []
