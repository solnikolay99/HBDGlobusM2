import copy
import glob
import os

from PyQt6 import QtWidgets

import animation.globparams as gp
from animation.dump.data import pars_in_file, pars_surf_file, pars_data2
from desktop.models.density_graph_model import DensityGraphModel
from desktop.views.main_window import MainWindow
from animation.debug import logger


class MainController:
    def __init__(self):
        self.density_graph_model = DensityGraphModel()
        self.main_view = MainWindow(self, self.density_graph_model)

        self.in_dir_path = ""
        self.in_file_path = ""
        self.dump_dir_path = ""
        self.data_files = []
        self.target_files = []
        self.grid_files = []
        self.cur_timeframe = 0

        self.flg_play = False

        self.logger = logger

        self.main_view.show()

    def on_click_find_in_file(self):
        textInFile = self.main_view.findChild(QtWidgets.QLineEdit, 'textInFile')
        in_file_path = QtWidgets.QFileDialog.getOpenFileName(None, 'Выберите файл настроек:', '')
        self.in_file_path = in_file_path[0]
        self.in_dir_path = os.path.dirname(self.in_file_path)
        textInFile.setText(self.in_file_path)

        self.get_global_params()
        self.get_surf_data()

    def on_click_find_dump_dir(self):
        textDumpDir = self.main_view.findChild(QtWidgets.QLineEdit, 'textDumpDir')
        self.dump_dir_path = QtWidgets.QFileDialog.getExistingDirectory(None, 'Выберите папку с дампом:', '')
        textDumpDir.setText(self.dump_dir_path)

        self.data_files = self.name_reader('dump.*.txt')
        self.target_files = self.name_reader('target.*.txt')
        self.grid_files = self.name_reader('grid.*.txt')

    def on_click_play_start(self):
        self.flg_play = True
        self._run_animation()

    def on_click_play_stop(self):
        self.flg_play = False

    def get_global_params(self):
        gp.global_params = pars_in_file(self.in_file_path)
        self.density_graph_model.set_main_box(0, 0, gp.global_params['width'], gp.global_params['height'])

    def get_surf_data(self):
        surfs = []
        for surf_file in gp.global_params['surfs']:
            _, _, polygons = pars_surf_file(os.path.join(self.in_dir_path, surf_file))
            surfs.append(copy.deepcopy(polygons))
        self.density_graph_model.mask_points = surfs

    def name_reader(self, pattern):
        return sorted(glob.glob(os.path.join(self.dump_dir_path, pattern)))

    def _run_animation(self):
        start_run_animation = self.logger.start_timer()

        data_file = self.data_files[self.cur_timeframe]
        self.cur_timeframe = self.cur_timeframe + 1 if self.cur_timeframe + 1 < len(self.data_files) else 0
        self.density_graph_model.points = pars_data2(data_file)

        self.logger.release_timer(f'Run animation for {self.cur_timeframe - 1}', start_run_animation)
