import sys

from PyQt6 import QtWidgets

from desktop.controllers.main_controller import MainController

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    controller = MainController()
    sys.exit(app.exec())
