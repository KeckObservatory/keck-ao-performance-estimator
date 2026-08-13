#!/usr/bin/env python3
"""Measured-SR tab usability pass (2026-07-26, Eduardo): (1) the EE
checkbox label makes clear it's field-map-only; (2) the EE aperture
correction's calibrated h is shown prominently next to the field map
(embedded + popped-out), not only in the scrolling log, and clears
when the field does; (3) the log has its own Pop-out button, opening a
second view onto the SAME QTextDocument so it always mirrors the
embedded log live; (4) the POPPED-OUT field map only (never the small
embedded one -- too small for it to help) has a "Show image" toggle:
off (default) reproduces the original plain-background, filled-marker
map exactly; on draws the actual frame as a background with HOLLOW
(open-face) markers so the star underneath a measurement is visible
through the ring. Fully offline; run headless
(QT_QPA_PLATFORM=offscreen).
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

import keck_ao_estimator.gui as gui


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

    # ---- (1) checkbox label ---------------------------------------------
    assert "(field only)" in win.n2_ee_corr.text(), win.n2_ee_corr.text()

    # ---- (2) h shown prominently, mirrored popped-out, clears -----------
    # (the window is never shown() in this offscreen test, so isVisible()
    # would read False regardless -- isHidden() checks the widget's OWN
    # explicit show/hide state, independent of the ancestor chain)
    assert win.n2_ee_out.isHidden() or not win.n2_ee_out.text()
    win._nirc2_set_ee_readout("EE h = 0.51 — 21 pairs, 6 full-radius + 9 corrected")
    assert "h = 0.51" in win.n2_ee_out.text()
    assert not win.n2_ee_out.isHidden()
    win._on_nirc2_map_popout()          # open the popped-out map window
    assert win._n2_map_ext_ee is not None
    assert "h = 0.51" in win._n2_map_ext_ee.text(), win._n2_map_ext_ee.text()
    win._n2_map_dialog.close()

    win._nirc2_set_ee_readout("")
    assert win.n2_ee_out.isHidden()

    # field-clear and a new-frame both clear the readout
    win._nirc2_set_ee_readout("EE h = 0.44 — stale")
    win._on_nirc2_field_clear()
    assert not win.n2_ee_out.text()
    win._nirc2_set_ee_readout("EE h = 0.44 — stale again")
    params = SimpleNamespace(plate_scale_mas=20.0, effwave_um=2.124,
                             lgs=True, utc=None)
    win._nirc2_display = lambda r: None
    win._n2_image = np.zeros((300, 300))
    win._n2_params = params
    win._on_nirc2_frame_done(1, object(), params, np.ones((300, 300)),
                             None, header={})
    assert not win.n2_ee_out.text(), \
        "a new frame must clear the stale EE readout"

    # ---- (3) log pop-out: second view, same document, live-synced -------
    assert getattr(win, "_n2_log_dialog", None) is None
    win._on_nirc2_log_popout()
    assert win._n2_log_dialog is not None
    popped = win._n2_log_dialog.findChild(QtWidgets.QPlainTextEdit)
    assert popped is not None and popped is not win.n2_log
    assert popped.document() is win.n2_log.document(), \
        "pop-out must share the SAME document, not a copy"
    win.n2_log.appendPlainText("a fresh log line")
    assert "a fresh log line" in popped.toPlainText(), \
        "the popped-out view must live-update with the embedded log"
    win._n2_log_dialog.close()
    assert win._n2_log_dialog is None

    # ---- (4) image-background toggle: pop-out only, default OFF ---------
    win._n2_image = np.linspace(0, 1000, 300 * 300).reshape(300, 300)
    win._n2_imno = 42
    win._n2_params = params
    win._n2_field = [mk(100.0, 120.0, 0.25, params),
                     mk(180.0, 160.0, 0.31, params)]
    win._n2_field_dropped = []
    win._nirc2_clear_selection()
    win._nirc2_draw_map()

    embedded_ax = win.n2_map_fig.axes[0]
    assert len(embedded_ax.images) == 0, \
        "the embedded map must NEVER show the image background"
    assert not is_hollow(kept_artist(embedded_ax)), \
        "the embedded map must keep the original FILLED markers"

    # open the pop-out: default (untoggled) must match the embedded look
    assert not getattr(win, "_n2_map_ext_show_image", False)
    win._on_nirc2_map_popout()
    fig_ext, _canvas_ext = win._n2_map_ext
    ax_ext = fig_ext.axes[0]
    assert len(ax_ext.images) == 0, \
        "pop-out must default to the plain background (no image)"
    assert not is_hollow(kept_artist(ax_ext)), \
        "pop-out must default to FILLED markers, same as embedded"

    # toggle it on: pop-out gets the image + hollow markers; embedded
    # map is redrawn too (same _nirc2_draw_map() call) and must be
    # UNAFFECTED
    win._on_nirc2_map_ext_image_toggle(True)
    fig_ext, _canvas_ext = win._n2_map_ext
    ax_ext = fig_ext.axes[0]
    assert len(ax_ext.images) == 1, \
        "toggled-on pop-out must draw the frame as a background"
    ext_kept = kept_artist(ax_ext)
    assert is_hollow(ext_kept), \
        f"toggled-on pop-out markers must be hollow, got {ext_kept.get_facecolor()}"
    assert ext_kept.get_edgecolor().shape[0] >= 1, \
        "hollow markers must still carry an edge colour (the SR/FWHM value)"
    embedded_ax = win.n2_map_fig.axes[0]
    assert len(embedded_ax.images) == 0, \
        "toggling the pop-out's image mode must not affect the embedded map"
    assert not is_hollow(kept_artist(embedded_ax)), \
        "embedded map markers must stay filled regardless of the pop-out toggle"

    # toggle back off
    win._on_nirc2_map_ext_image_toggle(False)
    fig_ext, _canvas_ext = win._n2_map_ext
    assert len(fig_ext.axes[0].images) == 0, "toggling off must remove the image"

    win._n2_map_dialog.close()

    print("gui_phase33: all checks passed")


if __name__ == "__main__":
    main()
