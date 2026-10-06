#!/usr/bin/env python3
"""Measured SR numbered OSIRIS runs (Eduardo 2026-10-06, kept in sync with
PyAO's strehl_tool): a PREFIX field (per instrument, saved in the config)
names numbered frames <PREFIX><number> -- NIRC2 n + 4 digits, OSIRIS
e.g. i260723_a + 6 digits. For OSIRIS an empty prefix fills in from the
frames in PATH; GO! and a double-click on a numbered OSIRIS frame drive
FIRST IMAGE. With $OSIRIS_STREHL_DATA set, the real test frames are
measured by number too. Offline; QT_QPA_PLATFORM=offscreen.
"""
import importlib.util
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
import warnings
warnings.filterwarnings("ignore")

from qtcompat import QtCore, QtWidgets

import keck_ao_estimator.gui as gui


def pump(cond, timeout=180):
    app = QtWidgets.QApplication.instance()
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents()
        QtCore.QThread.msleep(10)


def _phase29():
    spec = importlib.util.spec_from_file_location(
        "gui_phase29", os.path.join(HERE, "gui_phase29.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(win, im1):
    win.n2_im1.setValue(im1)
    win.n2_nim.setValue(1)
    win._n2_loaded_files = None
    win._on_nirc2_go()
    pump(lambda: win.n2_go.isEnabled())


def main():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow()
    win._nirc2_prefetch_guide_stars = lambda seq, cont: cont()
    QtWidgets.QMessageBox.exec = lambda self: 0
    assert win.n2_prefix.text() == "n", win.n2_prefix.text()

    tmp = tempfile.mkdtemp(prefix="gui51_osiris_")
    _phase29().make_osiris(tmp)                  # i260723_a000001.fits
    win.n2_instrument.setCurrentText("OSIRIS")
    assert win.n2_prefix.text() == "", "OSIRIS starts with no prefix"
    win.n2_path.setText(tmp)
    win._nirc2_refresh_files()
    assert win.n2_prefix.text() == "i260723_a", win.n2_prefix.text()
    assert win._nirc2_numbered_name(1) == "i260723_a000001"
    print("  [ok] OSIRIS prefix fills in from PATH")

    win.n2_autofind.setChecked(True)
    win.n2_log.clear()
    run(win, 1)
    log = win.n2_log.toPlainText()
    assert "Image 1  SR" in log, log
    print("  [ok] numbered OSIRIS GO! measures i260723_a000001")

    # double-click on the numbered OSIRIS frame drives FIRST IMAGE
    win.n2_im1.setValue(5)
    win._on_nirc2_file_dclick(win.n2_files.item(0))
    pump(lambda: win.n2_go.isEnabled())
    assert win.n2_im1.value() == 1, win.n2_im1.value()
    print("  [ok] double-click drives FIRST IMAGE for OSIRIS")

    # per-instrument prefixes, and a config round trip
    win.n2_instrument.setCurrentText("NIRC2")
    assert win.n2_prefix.text() == "n"
    win.n2_instrument.setCurrentText("OSIRIS")
    assert win.n2_prefix.text() == "i260723_a"
    c = win._collect_config()
    assert c["nirc2"]["instrument_prefixes"]["OSIRIS"] == "i260723_a", c
    win.n2_prefix.setText("zzz")
    win._apply_config(c)
    assert win.n2_prefix.text() == "i260723_a", win.n2_prefix.text()
    print("  [ok] prefixes are per instrument and saved in the config")

    data = os.environ.get("OSIRIS_STREHL_DATA")
    frames = os.path.join(data or "", "test_images")
    if data and os.path.isdir(frames):
        win.n2_path.setText(frames)
        win._nirc2_refresh_files()
        win.n2_idl_version.setCurrentText("legacy")
        win.n2_log.clear()
        run(win, 3002)
        line = next(l for l in win.n2_log.toPlainText().splitlines()
                    if l.startswith("Image 3002  SR"))
        sr = float(line.split("SR")[1].split()[0])
        assert abs(sr - 0.3425) < 0.002, line   # IDL golden 0.341
        print(f"  [ok] real OSIRIS frame 3002 by number: {line[:40]}")
    else:
        print("  [skip] $OSIRIS_STREHL_DATA not set: real-frame check")
    print("gui_phase51: all checks passed")


if __name__ == "__main__":
    main()
