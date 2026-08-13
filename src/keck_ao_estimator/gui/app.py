"""Zero-argument console entry point (see pyproject [project.scripts]
keck-ao-estimator-gui)."""
import sys

from qtcompat import QtWidgets

from .._version import APP_NAME, __version__
from .mainwindow import MainWindow


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    win = MainWindow()
    win.show()
    win._show_about(at_startup=True)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
