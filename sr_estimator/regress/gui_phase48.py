#!/usr/bin/env python3
"""Measured-SR tab, 2026-10-05 (Eduardo): (1) a NIRC2 directory listing
skips the unprocessed `_unp` files (raw n_unp_NNNN.fits, KOA *_unp.fits);
(2) the EMBEDDED field map gets its own "Image" toggle -- the frame drawn
underneath with hollow markers, and 'Add star by click' works on it --
independent of the pop-out's toggle (reverses the 2026-07-26 pop-out-only
call; gui_phase33 still pins the default-off look). Fully offline; run
headless (QT_QPA_PLATFORM=offscreen).
"""
import os
import sys
import tempfile
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

import keck_ao_estimator.gui as gui
from keck_ao_estimator.frame_watch import is_unp_frame
from keck_ao_estimator.nirc2 import nirc2_frame_params


def mk(x, y, sr, params):
    return SimpleNamespace(strehl=sr, fwhm_mas=60.0, x=x, y=y,
                           sr_err=0.004, edge=False, crowded=False,
                           params=params)


def kept_artist(ax):
    return next(c for c in ax.collections
               if getattr(c, "_n2_pool", None) == "kept")


def is_hollow(collection):
    fc = collection.get_facecolor()
    return len(fc) == 0 or np.allclose(fc[:, 3], 0.0)


def main():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow()

    # ---- (1) _unp files are skipped in a NIRC2 directory -----------------
    for name, unp in (("n_unp_0017.fits", True),
                      ("N2.20210821_42637_unp.fits", True),
                      ("N2.20210821_42637_UNP.fits.gz", True),
                      ("n0017.fits", False),
                      ("N2.20210821_42637.fits", False),
                      ("unpacked_n0001.fits", False)):
        assert is_unp_frame(name) is unp, (name, unp)
    with tempfile.TemporaryDirectory() as d:
        for name in ("n0001.fits", "n_unp_0001.fits", "n0002.fits",
                     "N2.20210821_42637_unp.fits", "notes.txt"):
            open(os.path.join(d, name), "w").close()
        win.n2_instrument.setCurrentText("NIRC2")
        win.n2_path.setText(d)
        win._nirc2_refresh_files()
        listed = [win.n2_files.item(i).text()
                  for i in range(win.n2_files.count())]
        assert listed == ["n0001.fits", "n0002.fits"], listed

    # ---- (2) embedded map image toggle -----------------------------------
    hdr = {"CAMNAME": "narrow", "PMSNAME": "largehex", "EFFWAVE": 2.124,
           "PMRANGL": 0.0, "COADDS": 1}
    params = nirc2_frame_params(hdr)
    win._n2_image = np.linspace(0, 1000, 300 * 300).reshape(300, 300)
    win._n2_imno = 42
    win._n2_params = params
    win._n2_field = [mk(100.0, 120.0, 0.25, params),
                     mk(180.0, 160.0, 0.31, params)]
    win._n2_field_dropped = []
    win._nirc2_clear_selection()

    assert not win.n2_map_show_image.isChecked(), "default must stay OFF"
    win.n2_map_show_image.setChecked(True)          # redraws via toggled
    ax = win.n2_map_fig.axes[0]
    assert len(ax.images) == 1, "embedded toggle must draw the frame"
    assert is_hollow(kept_artist(ax)), "embedded markers must go hollow"

    # the pop-out keeps its own toggle, default off
    win._on_nirc2_map_popout()
    fig_ext, _ = win._n2_map_ext
    assert len(fig_ext.axes[0].images) == 0, \
        "the embedded toggle must not switch the pop-out's image on"
    win._n2_map_dialog.close()

    # Add star by click works on the embedded map only while it shows
    # the image; the click arrives in arcsec and maps back to pixels
    hits = []
    win._nirc2_measure_at = lambda px, py: hits.append((px, py)) or "R"
    win._nirc2_field_accept = lambda r: None
    win._nirc2_add_to_field = lambda r: None
    win.n2_add_star.setChecked(True)
    ps = params.plate_scale_mas / 1000.0
    ev = SimpleNamespace(canvas=win.n2_map_canvas, inaxes=ax,
                         xdata=(120.0 - 150.0) * ps,
                         ydata=(80.0 - 150.0) * ps)
    win._on_nirc2_map_ext_click(ev)
    assert len(hits) == 1, hits
    assert abs(hits[0][0] - 120.0) < 1e-6 and abs(hits[0][1] - 80.0) < 1e-6, \
        hits
    win.n2_map_show_image.setChecked(False)
    win._on_nirc2_map_ext_click(ev)
    assert len(hits) == 1, "no add-by-click on the plain embedded map"
    win.n2_add_star.setChecked(False)

    print("gui_phase48: all checks passed")


if __name__ == "__main__":
    main()
