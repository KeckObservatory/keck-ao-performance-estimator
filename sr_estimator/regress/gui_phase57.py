#!/usr/bin/env python3
"""Measured SR viewer, in sync with PyAO's strehl_tool (Eduardo 2026-10-25:
mirror the PyAO GUI into KAPE before the 1.6.0 release): the Plane row
steps through a NIRC2 frame's coadds (its _unp file) and an OSIRIS IFU
cube's wavelength bins ('λ bin' in the slider's place), and Play shows them
in the stretch each has when measured; the wheel zooms around the cursor,
a left drag pans, Fit unzooms; the pixel under the mouse is read out over
the image, yellow / red near and at saturation; a right drag sets a DS9
colour map and a right click opens the stretch menu; a click measures with
AUTOFIND on and the PICK ZOOM follows the cursor until it leaves the image;
the frame list is newest first. Offline; QT_QPA_PLATFORM=offscreen.
"""
import dataclasses
import os
import sys
import tempfile
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
import warnings
warnings.filterwarnings("ignore")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.pop("KECK_AO_INSTRUMENTS", None)

import numpy as np
from astropy.io import fits
from qtcompat import QtCore, QtWidgets

import keck_ao_estimator.gui as gui
from keck_ao_estimator import osiris_ifs
from keck_ao_estimator.gui.tabs.nirc2_strehl import FILE_SORTS, NIRC2_CSV_COLUMNS
from gui_phase56 import ifs_cube, ifs_header, nirc2_header, star_image, write_cube

RNG = np.random.default_rng(57)
TMP = tempfile.mkdtemp(prefix="gui57_")


def pump(cond, timeout=120):
    app = QtWidgets.QApplication.instance()
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents()
        QtCore.QThread.msleep(10)
    return cond()


def settle(win):
    """Wait for the measurement (and any re-measure of a plane) to end."""
    QtWidgets.QApplication.instance().processEvents()
    assert pump(lambda: win.n2_go.isEnabled()
                and not win._n2_plane_timer.isActive()
                and not (win._n2_worker is not None
                         and win._n2_worker.isRunning())), "measurement hung"


def make_nirc2(d, n=3):
    """n0007.fits (the sum) and n_unp_0007.fits (CDS reads per coadd) of a
    1024^2 narrow-camera star."""
    frames = []
    for k in range(n):
        img = star_image(1024, 512.3, 511.6, scale=60.0 * (1 + 0.1 * k))
        frames.append(np.round(img + RNG.normal(0, 2.0, img.shape)))
    hdr = nirc2_header(n)
    hdus = [fits.PrimaryHDU(header=hdr)]
    for img in frames:
        reset = RNG.integers(-60, -40, size=img.shape).astype(float)
        hdus.append(fits.ImageHDU(np.stack([reset, reset + img]).astype(np.int16)))
    fits.HDUList(hdus).writeto(os.path.join(d, "n_unp_0007.fits"))
    fits.writeto(os.path.join(d, "n0007.fits"),
                 np.sum(frames, axis=0).astype(np.float32), header=hdr)


def item_named(win, name):
    return next(win.n2_files.item(i) for i in range(win.n2_files.count())
                if win.n2_files.item(i).text() == name)


def mouse(win, button, x, y, xdata, ydata):
    return SimpleNamespace(button=button, x=x, y=y, xdata=xdata, ydata=ydata,
                           inaxes=win.n2_fig.axes[0], step=0)


def check_coadds(win):
    d = os.path.join(TMP, "nirc2")
    os.makedirs(d)
    make_nirc2(d)
    win.n2_instrument.setCurrentText("NIRC2")
    win.n2_prefix.setText("n")
    win.n2_path.setText(d)
    names = [win.n2_files.item(i).text() for i in range(win.n2_files.count())]
    assert names == ["n0007.fits"], names       # the _unp copy is not listed
    win._on_nirc2_file_dclick(item_named(win, "n0007.fits"))
    settle(win)
    assert win._n2_plane_kind == "coadd", win._n2_plane_kind
    assert win.n2_plane.maximum() == 3 and win.n2_plane.value() == 0
    assert win.n2_plane.specialValueText() == "sum"
    assert win.n2_play.isEnabled() and win._n2_shown_path.endswith("n0007.fits")
    print("  [ok] a NIRC2 frame with an _unp file: Plane = sum, coadds 1..3")

    n_rows = len(win._n2_csv_rows)
    win.n2_plane.setValue(2)            # re-measures after a short pause
    assert win._n2_plane_timer.isActive()
    settle(win)
    assert len(win._n2_csv_rows) == n_rows + 1
    title = win.n2_fig.axes[0].get_title()
    assert "coadd 2/3" in title, title
    assert "COADD" in win.n2_warn.text(), win.n2_warn.text()
    row = win._n2_csv_rows[-1]
    assert "coadd-short-exposure-sr" in row["flags"], row["flags"]
    assert row["effwave_um"] == win._n2_params.effwave_um
    assert row["sky_mode"], row
    log = win.n2_log.toPlainText()
    assert "[coadd: frame flux / N]" in log
    assert "predicted: not compared — a coadd's SR is short-exposure" in log
    print("  [ok] Plane 2 measures coadd 2 (title, COADD tag, CSV flags, "
          "no prediction compared)")

    for stretch in ("IDL ±5σ", "asinh"):
        win.n2_stretch.setCurrentText(stretch)
        measured = win._n2_main_im
        clim, pixels = measured.get_clim(), np.array(measured.get_array())
        win.n2_play.setChecked(True)
        assert win._nirc2_is_playing()
        win._n2_play_timer.stop()
        win._n2_play_next = 2
        win._nirc2_play_tick()
        win._n2_play_timer.stop()
        played = win._n2_play_im
        assert played is not None and played is not measured
        assert np.allclose(played.get_clim(), clim, rtol=1e-9, atol=1e-9), \
            (stretch, played.get_clim(), clim)
        arr = np.array(played.get_array())
        assert arr.shape == pixels.shape and np.allclose(arr, pixels), stretch
        assert "playing (not measured)" in played.axes.get_title()
        n_rows = len(win._n2_csv_rows)
        win.n2_play.setChecked(False)   # Pause: measures the plane shown
        settle(win)
        assert not win._nirc2_is_playing()
        assert len(win._n2_csv_rows) == n_rows + 1
        assert win.n2_play.text() == "▶ Play"
    win.n2_stretch.setCurrentText("IDL ±5σ")
    print("  [ok] a played coadd has the measured coadd's pixels and limits "
          "(IDL ±5σ, asinh); Pause measures it")

    # pixel readout, with the saturation colours
    ax = win.n2_fig.axes[0]
    win._on_nirc2_motion(mouse(win, None, 50, 50, 512.2, 511.4))
    lbl = win.n2_cursor_label
    assert not lbl.isHidden() and lbl.text().startswith("x 512  y 511"), lbl.text()
    assert "SATURATED" not in lbl.text()
    assert win.n2_cap_star.text() == "PICK ZOOM", "hover zoom with AUTOFIND on"
    p = win._n2_params
    win._n2_params = dataclasses.replace(p, max_counts=1.0)
    win._on_nirc2_motion(mouse(win, None, 50, 50, 512.2, 511.4))
    assert lbl.text().endswith("SATURATED") and "#e08276" in lbl.styleSheet()
    v = win._nirc2_cursor_value(512, 511)
    win._n2_params = dataclasses.replace(p, max_counts=v / 0.9)
    win._on_nirc2_motion(mouse(win, None, 50, 50, 512.2, 511.4))
    assert lbl.text().endswith("near sat") and "#d4a437" in lbl.styleSheet()
    win._n2_params = p
    win._on_nirc2_image_leave(None)
    assert lbl.isHidden() and win.n2_cap_star.text() == "MEASURED STAR"
    print("  [ok] readout x, y, value; yellow near / red at saturation; the "
          "PICK ZOOM follows the cursor and gives the panel back on leave")

    # wheel zoom around the cursor, kept across redraws, Fit
    win._on_nirc2_scroll(SimpleNamespace(inaxes=ax, xdata=512.0, ydata=512.0,
                                         step=3))
    view = win._n2_view
    assert view is not None and view[0] == (1024, 1024)
    span = view[1][1] - view[1][0]
    assert abs(span - 1024 / 1.25 ** 3) < 1e-6, span
    win._nirc2_draw_main()
    assert win.n2_fig.axes[0].get_xlim() == view[1], "zoom kept on redraw"
    # a left drag pans; it is not a click (no measurement)
    n_rows = len(win._n2_csv_rows)
    ax = win.n2_fig.axes[0]
    x0 = ax.get_xlim()[0]
    win._on_nirc2_press(mouse(win, 1, 200, 200, 500.0, 500.0))
    win._on_nirc2_motion(mouse(win, 1, 240, 200, 520.0, 500.0))
    win._on_nirc2_release(mouse(win, 1, 240, 200, 520.0, 500.0))
    assert ax.get_xlim()[0] < x0, (ax.get_xlim(), x0)
    assert len(win._n2_csv_rows) == n_rows, "a drag must not measure"
    win._nirc2_zoom_fit()
    assert win._n2_view is None and ax.get_xlim() == (-0.5, 1023.5)
    print("  [ok] wheel zoom kept across redraws; left drag pans without "
          "measuring; Fit shows the whole frame")

    # a right drag sets the colour map; Reset; a right click opens the menu
    base = win._n2_main_im.get_clim()
    win._on_nirc2_press(mouse(win, 3, 10, 10, 100.0, 100.0))
    win._on_nirc2_motion(mouse(win, 3, 40, 300, 120.0, 300.0))
    win._on_nirc2_release(mouse(win, 3, 40, 300, 120.0, 300.0))
    assert (win._n2_bias, win._n2_contrast) != (0.5, 1.0)
    assert win._n2_main_im.get_clim() != base and win.n2_cmap_lbl.text()
    assert "display: IDL ±5σ, bias" in win.n2_log.toPlainText()
    win._nirc2_reset_colour_map()
    assert np.allclose(win._n2_main_im.get_clim(), base)
    assert win.n2_cmap_lbl.text() == ""
    popped = []
    win.n2_stretch_menu.popup = lambda pos: popped.append(pos)
    win._on_nirc2_press(mouse(win, 3, 10, 10, 100.0, 100.0))
    win._on_nirc2_release(mouse(win, 3, 10, 10, 100.0, 100.0))
    assert popped, "a right click opens the stretch menu"
    win._n2_stretch_actions["asinh"].trigger()
    assert win.n2_stretch.currentText() == "asinh"
    win.n2_stretch.setCurrentText("IDL ±5σ")
    print("  [ok] right drag = DS9 colour map (Reset restores); right click "
          "= stretch menu")

    # a left click measures with AUTOFIND on
    assert win.n2_autofind.isChecked()
    n_rows = len(win._n2_csv_rows)
    win._on_nirc2_press(mouse(win, 1, 300, 300, 512.0, 511.0))
    win._on_nirc2_release(mouse(win, 1, 300, 300, 512.0, 511.0))
    assert len(win._n2_csv_rows) == n_rows + 1, "click measures, AUTOFIND on"
    print("  [ok] a click measures with AUTOFIND on")

    # a click on a playing cube pauses it (and measures the plane shown)
    win.n2_play.setChecked(True)
    win._n2_play_timer.stop()
    win._nirc2_play_tick()
    win._n2_play_timer.stop()
    win._on_nirc2_click(mouse(win, 1, 300, 300, 512.0, 511.0))
    assert not win._nirc2_is_playing()
    settle(win)
    # the window closing stops playback
    win.n2_play.setChecked(True)
    assert win._nirc2_is_playing()
    win._nirc2_shutdown()
    assert not win._nirc2_is_playing() and not win._n2_play_timer.isActive()
    print("  [ok] a click pauses playback; closing stops it")

    assert win._nirc2_frame_key("7 p2") == "n0007 p2"
    assert win._nirc2_frame_key("7") == "n0007"
    labels = [lab for _k, lab in NIRC2_CSV_COLUMNS]
    assert {"Wavelength (um)", "Sky", "Flags"} <= set(labels), labels
    print("  [ok] frame keys with planes; CSV wavelength, sky and flags")


def check_ifs(win):
    d = os.path.join(TMP, "osiris6", "261020", "SPEC", "ORP")
    os.makedirs(d)
    data, dq = ifs_cube(35, 24.3, 20.6, sigma_mas=10.0)
    write_cube(os.path.join(d, "s261020_a010002_Kn3_035.fits"), data, dq,
               ifs_header())
    win.n2_instrument.setCurrentText("OSIRIS IFU")
    win.n2_path.setText(d)
    win._on_nirc2_file_dclick(item_named(win, "s261020_a010002_Kn3_035.fits"))
    settle(win)
    assert win._n2_plane_kind == "ifs", win._n2_plane_kind
    assert win.n2_plane.specialValueText() == "all λ"
    nb = len(osiris_ifs.channel_bins(40, win.n2_ifs_bin.value()))
    assert win.n2_plane.maximum() == nb, (win.n2_plane.maximum(), nb)
    assert not win.n2_ifs_bin.isHidden() and win.n2_plane_slider.isHidden()
    assert "ORP: s261020_a010002, nothing subtracted" in win.n2_log.toPlainText()
    row = win._n2_csv_rows[-1]
    assert "ifs-relative-sr" in row["flags"] and "orp-none" in row["flags"], row
    assert "IFU" in win.n2_warn.text() and "SR RELATIVE" in win.n2_warn.text()
    assert "predicted: not compared — an IFU SR is relative" in win.n2_log.toPlainText()
    print(f"  [ok] an IFU cube: Plane = all λ + {nb} bins, λ bin shown, ORP "
          "line logged, flags, no prediction compared")
    win.n2_ifs_bin.setValue(10)         # re-measures the cube on display
    settle(win)
    assert win.n2_plane.maximum() == 4, win.n2_plane.maximum()
    win.n2_plane.setValue(3)
    settle(win)
    title = win.n2_fig.axes[0].get_title()
    assert "λc" in title and "(10 ch)" in title, title
    win.n2_log.clear()
    win._on_nirc2_measure_field()
    assert "Measure field works on whole frames" in win.n2_log.toPlainText()
    print("  [ok] λ bin 10 -> 4 bins; Plane 3 titled with its λc; Measure "
          "field refused on a cube")
    # click measures the IFU way
    n_rows = len(win._n2_csv_rows)
    win._on_nirc2_click(SimpleNamespace(xdata=24.0, ydata=20.0,
                                        inaxes=win.n2_fig.axes[0]))
    assert len(win._n2_csv_rows) == n_rows + 1
    assert "ifs-relative-sr" in win._n2_csv_rows[-1]["flags"]
    print("  [ok] a click on an IFU bin measures it the IFU way")


def check_file_order(win):
    d = os.path.join(TMP, "order")
    os.makedirs(d)
    t = time.time() - 1000
    for i, name in enumerate(["n0003.fits", "n0001.fits", "n0002.fits"]):
        fits.writeto(os.path.join(d, name), np.zeros((4, 4), np.float32))
        os.utime(os.path.join(d, name), (t + 10 * i, t + 10 * i))
    win.n2_instrument.setCurrentText("NIRC2")
    win.n2_file_sort.setCurrentText(FILE_SORTS[0])
    win.n2_path.setText(d)

    def names():
        return [win.n2_files.item(i).text() for i in range(win.n2_files.count())]
    assert names() == ["n0002.fits", "n0001.fits", "n0003.fits"], names()
    item_named(win, "n0001.fits").setSelected(True)
    win.n2_file_sort.setCurrentText("Name")
    assert names() == ["n0001.fits", "n0002.fits", "n0003.fits"], names()
    assert [it.text() for it in win.n2_files.selectedItems()] == ["n0001.fits"]
    win.n2_file_sort.setCurrentText(FILE_SORTS[0])
    assert names()[0] == "n0002.fits"
    print("  [ok] frame list newest first by default; Name re-orders in "
          "place, selection kept")


def main():
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow()
    assert win.n2_file_sort.currentText() == FILE_SORTS[0]
    win.n2_autofind.setChecked(True)
    check_file_order(win)
    check_coadds(win)
    check_ifs(win)
    win._nirc2_shutdown()
    app.processEvents()
    print("gui_phase57: PASS")


if __name__ == "__main__":
    main()
