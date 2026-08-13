"""Qt binding shim for the ao_strehl GUI.

The GUI is written to the **PyQt6** API per the build spec (scoped enums like
``Qt.AlignmentFlag.AlignRight``, ``widget.exec()``, ``QAction`` from QtGui).
This module is the single place the binding is chosen: it imports PyQt6 if
present and otherwise falls back to PyQt5 (which, at 5.15, supports the same
scoped-enum and ``exec()`` idioms), papering over the two differences that
actually matter — the package name and where ``QAction`` lives.

To move to PyQt6 on a box that has it, nothing needs editing: this file already
prefers it. Every other GUI module imports Qt names FROM HERE, never directly.
"""
import os

BINDING = None
try:                                    # spec's mandated toolkit, if installed
    from PyQt6 import QtCore, QtGui, QtWidgets
    BINDING = "PyQt6"
except ImportError:                     # working fallback on this dev box
    from PyQt5 import QtCore, QtGui, QtWidgets
    BINDING = "PyQt5"

# Tell matplotlib's qtagg backend to use the same binding we picked, so the
# canvas and the app share one Qt. (setdefault: respect an explicit env choice.)
os.environ.setdefault("QT_API", "pyqt6" if BINDING == "PyQt6" else "pyqt5")

# --- common names, binding-independent ---------------------------------------
Qt = QtCore.Qt
Signal = QtCore.pyqtSignal
Slot = QtCore.pyqtSlot
QThread = QtCore.QThread
QObject = QtCore.QObject
QTimer = QtCore.QTimer

# QAction moved QtWidgets -> QtGui between PyQt5 and PyQt6.
QAction = getattr(QtGui, "QAction", None) or QtWidgets.QAction

__all__ = ["BINDING", "QtCore", "QtGui", "QtWidgets", "Qt", "Signal", "Slot",
           "QThread", "QObject", "QTimer", "QAction"]
