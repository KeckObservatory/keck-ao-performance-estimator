#!/usr/bin/env python3
"""Measured SR, failed measurement (Eduardo 2026-10-06): a frame whose
automatic measurement fails (e.g. "centroid failed; try a bigger
aperture") is still SHOWN, titled with the failure, the reason in the
warning line, and a click on it measures there even with AUTOFIND on.
The next frame clears the state. Offline; QT_QPA_PLATFORM=offscreen.
"""
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from qtcompat import QtWidgets

import keck_ao_estimator as engine
import keck_ao_estimator.gui as gui
from keck_ao_estimator.image_strehl import _failed
from keck_ao_estimator.nirc2 import nirc2_frame_params


def main():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow()
    params = nirc2_frame_params({"CAMNAME": "wide", "PMSNAME": "largehex",
                                 "EFFWAVE": 2.124, "PMRANGL": 0.0,
                                 "COADDS": 1})
    img = np.random.default_rng(7).normal(100.0, 5.0, (256, 256))
    win.n2_autofind.setChecked(True)

    bad = _failed(params, "centroid failed; try a bigger aperture")
    win._on_nirc2_frame_done(7, bad, params, img, None)
    assert win._n2_image is img
    title = win._n2_last_draw[0]
    assert "measurement failed" in title, title
    ax = win.n2_fig.axes[0]
    assert len(ax.images) == 1, "the failed frame must still be drawn"
    assert "centroid failed" in win.n2_warn.text(), win.n2_warn.text()
    assert "Image 7: centroid failed" in win.n2_log.toPlainText()
    print("  [ok] failed frame is displayed with the reason")

    hits = []
    win._nirc2_measure_at = lambda x, y: hits.append((x, y)) or bad
    ev = SimpleNamespace(xdata=120.0, ydata=130.0, inaxes=ax)
    win._on_nirc2_click(ev)
    assert hits == [(120.0, 130.0)], hits
    print("  [ok] click measures on a failed frame with AUTOFIND on")

    # an ordinary (successful) frame restores AUTOFIND's click-ignore
    win._on_nirc2_frame_done(8, None, params, img.copy(), None)
    win.n2_autofind.setChecked(True)
    win._n2_frame_failed = False
    win._on_nirc2_click(ev)
    assert len(hits) == 1, "AUTOFIND ignores clicks on a normal frame"
    print("gui_phase50: all checks passed")


if __name__ == "__main__":
    main()
