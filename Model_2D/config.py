debug = True  # debug flag

# Задание параметров
shape_x = 1000  # высота расчетной области
shape_y = 240  # ширина расчетной области

time_steps = 2000  # количество временных шагов
size = int(1e3)  # количество частиц
full_size = int(2e3)  # количество частиц с учетом дублирования частиц
wall_tube = 1
out_folder = 'calculation/001'

ln = size
di = [0, 0, 0]
prob = 0.5
max_points_per_cell = 7  # максимальное количество частиц в ячейке
height = 0.5  # полувысота капилляра
Vx = 1e7 / 2.5  # скорость по x при начальном заполнении капилляра
Vy = height  # скорость по y при начальном заполнении капилляра
capillary_length = 80  # длина капилляра
x_min_lim = 40  # координата по x где частицы уходят из расчета
m = 1
bias = 40  # отступ по y где частицы уходят из расчета
t_step = 0.25 * 1e-6  # временной шаг
V_pot = 0.01 * 2.5 * 1e7 / 2  # скорость потока в капилляре

max_velocity: float = 1.5 * 1e7 / 2.5
min_velocity: float = -1.5 * 1e7 / 2.5

# multithreading params
use_multithread = False
thread_count = 6

# dump params
dump_every = 100
cur_time_frame = [0]

# pre-calculations for check_walls
shape_y_top: float = shape_y // 2 + height - 0.5
shape_y_bottom: float = shape_y // 2 - height - 0.5

# pre-calculations for inverse_maxwell_distribution
m_hel: float = 6.6464764e-27
k: float = 1.380649e-23
T: int = 300
k_t_m_hel: float = (-2 * k * T / m_hel)
denominator: float = 1e4 / 2.5
