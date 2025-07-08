import pyqtgraph as pg
from PyQt6 import QtWidgets, uic

from desktop.models.density_graph_model import DensityGraphModel
from animation.debug import logger


class MainWindow(QtWidgets.QMainWindow):

    def __add_density_graph__(self):
        self.tabWidget: QtWidgets.QTabWidget
        self.tabWidget = self.findChild(QtWidgets.QTabWidget, 'tabWidget')

        self.density_graph = pg.PlotWidget()
        self.scatterPlot = pg.ScatterPlotItem()

        self.tabWidget.addTab(self.density_graph, "Плотность")
        self.density_graph.setBackground((193, 219, 225, 255))
        self.density_graph.setLabel("left", "y, mm")
        self.density_graph.setLabel("bottom", "x, mm")

        self.density_graph.addItem(self.scatterPlot)

        self.logger = logger

    def __add_temperatura_graph__(self):
        self.tabWidget: QtWidgets.QTabWidget
        self.tabWidget = self.findChild(QtWidgets.QTabWidget, 'tabWidget')

        self.temp_graph = pg.PlotWidget()

        self.tabWidget.addTab(self.temp_graph, "Температура")
        self.temp_graph.setBackground((193, 219, 225, 255))
        self.temp_graph.setLabel("left", "y, mm")
        self.temp_graph.setLabel("bottom", "x, mm")

    def __add_file_settings__(self):
        self.buttonFindInFile: QtWidgets.QPushButton
        self.buttonFindDumpDir: QtWidgets.QPushButton
        self.textInFile: QtWidgets.QLineEdit
        self.textDumpDir: QtWidgets.QLineEdit

        self.buttonFindInFile = self.findChild(QtWidgets.QPushButton, 'buttonFindInFile')
        self.buttonFindInFile.clicked.connect(self.controller.on_click_find_in_file)
        self.buttonFindDumpDir = self.findChild(QtWidgets.QPushButton, 'buttonFindDumpDir')
        self.buttonFindDumpDir.clicked.connect(self.controller.on_click_find_dump_dir)

    def __add_play_buttons__(self):
        self.buttonPlay: QtWidgets.QPushButton
        self.buttonStop: QtWidgets.QPushButton

        self.buttonPlay = self.findChild(QtWidgets.QPushButton, 'buttonPlay')
        self.buttonPlay.clicked.connect(self.controller.on_click_play_start)
        self.buttonStop = self.findChild(QtWidgets.QPushButton, 'buttonStop')
        self.buttonStop.clicked.connect(self.controller.on_click_play_stop)

    def __init__(self, controller, model: DensityGraphModel):
        self.tabWidget: QtWidgets.QTabWidget

        super(MainWindow, self).__init__()

        self.controller = controller
        self.model = model

        uic.loadUi('desktop/views/ui/main.ui', self)

        self.__add_density_graph__()
        self.__add_temperatura_graph__()

        self.__add_file_settings__()
        self.__add_play_buttons__()

        self.model.add_observer(self)

    def model_is_changed(self):
        start_model_is_changed = self.logger.start_timer()

        main_rect_x = [self.model.main_box[0][0],
                       self.model.main_box[1][0],
                       self.model.main_box[2][0],
                       self.model.main_box[3][0],
                       self.model.main_box[0][0]]
        main_rect_y = [self.model.main_box[0][1],
                       self.model.main_box[1][1],
                       self.model.main_box[2][1],
                       self.model.main_box[3][1],
                       self.model.main_box[0][1]]
        pen = pg.mkPen(color=(0, 0, 0), width=2)
        self.density_graph.plot(main_rect_x, main_rect_y, pen=pen)

        self.density_graph.setXRange(self.model.main_box[0][0], self.model.main_box[2][0])
        self.density_graph.setYRange(self.model.main_box[0][1], self.model.main_box[2][1])

        scatterData = []
        '''
        for surfs in self.model.mask_points:
            for surf in surfs:
                for point in surf:
                    scatterData.append(
                        {
                            'pos': (point[0], point[1]),
                            'size': 3,
                            'pen': {'color': 'g', 'width': 2},
                            'brush': pg.mkBrush(127, 127, 127, 255)}
                    )
        '''

        for point in self.model.points:
            scatterData.append(
                {
                    'pos': (point[0], point[1]),
                    #'size': 1,
                    #'pen': {'color': 'r', 'width': 1},
                    #'brush': pg.mkBrush(255, 0, 0, 255)
                }
            )

        self.logger.release_timer('Convert data in model_is_changed', start_model_is_changed)

        start_plot_data = self.logger.start_timer()
        self.scatterPlot.setPen({'color': 'r', 'width': 1})
        self.scatterPlot.setSize(1)
        self.scatterPlot.setData(scatterData)
        self.logger.release_timer(f'Plot data for {len(self.model.points)} points', start_plot_data)
