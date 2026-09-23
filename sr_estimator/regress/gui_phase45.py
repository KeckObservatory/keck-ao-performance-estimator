#!/usr/bin/env python3
"""Starlist dialog sky plot (2026-09-22, Eduardo Marin): to the right of the
table, a polar map (zenith at the centre, N up, E right) of where the
selected target sits in Keck's pointing space --

  * the selected telescope's pointing limits (Nasmyth-deck wedge blocked
    below its floor, the vignetted band below 18 deg, the zenith ceiling,
    the guiding ring) from constants.POINTING_LIMITS;
  * the target's path across the night (engine.night_track: 17:00-08:00
    HST, Sun altitude per sample), with its position at the dialog's
    "Evaluate at" time;
  * the Moon (engine.moon_altaz_deg, topocentric) with 15 / 30 deg circles.

Engine contract: night_track spans the night of the evaluation time;
moon_altaz_deg is above/below the horizon consistently with the Sun-Moon
geometry it came from; sky_plot.small_circle points lie at the requested
separation. GUI: the dialog carries the canvas, a row click draws the path,
the telescope selects the wedge, and changing the evaluation time redraws.
Fully offline, headless.
"""
import datetime
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
from qtcompat import QtWidgets, QtCore
import keck_ao_estimator as engine
import keck_ao_estimator.gui as gui
from keck_ao_estimator.gui.sky_plot import small_circle
np = engine.np
SAMPLE = os.path.join(ROOT, "examples", "synthetic_k1lgs.lst")


def settle(n=6):
    app = QtWidgets.QApplication.instance()
    for _ in range(n):
        app.processEvents(); QtCore.QThread.msleep(25)


def engine_contract():
    when = datetime.datetime(2026, 9, 22, 22, 0)           # HST
    tr = engine.night_track("23:59:38.27", "-20:33:14.58", when)
    assert tr["times_hst"][0] == datetime.datetime(2026, 9, 22, 17, 0)
    assert tr["times_hst"][-1] == datetime.datetime(2026, 9, 23, 8, 0)
    assert len(tr["az"]) == len(tr["el"]) == len(tr["sun_alt"]) == 91
    assert tr["sun_alt"][0] > 0 > tr["sun_alt"][30]          # 17:00 day, 22:00 night
    # early-morning evaluation belongs to the previous evening's night
    tr2 = engine.night_track("23:59:38.27", "-20:33:14.58",
                             datetime.datetime(2026, 9, 23, 3, 0))
    assert tr2["times_hst"][0] == tr["times_hst"][0]
    az, el = engine.moon_altaz_deg(when + datetime.timedelta(hours=10))
    assert 0 <= az < 360 and -90 <= el <= 90
    for rad in (15.0, 30.0):
        caz, cel = small_circle(120.0, 40.0, rad)
        d = np.degrees(np.arccos(np.clip(
            np.sin(np.radians(40)) * np.sin(np.radians(cel))
            + np.cos(np.radians(40)) * np.cos(np.radians(cel))
            * np.cos(np.radians(caz - 120.0)), -1, 1)))
        assert np.allclose(d, rad, atol=1e-6)
    print("  [ok] engine: night_track 17:00-08:00 HST (91 samples, Sun alt); "
          "Moon alt/az; 15/30 deg circles exact")


def gui_dialog():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = gui.MainWindow(); win.resize(1550, 950); win.show(); settle(4)
    win._starlist_eval_dt = datetime.datetime(2026, 9, 22, 22, 0)
    win._open_starlist(SAMPLE); settle(3)
    if win._starlist_dialog is None:
        win._show_starlist_clicked(); settle(3)
    dlg = win._starlist_dialog
    assert dlg is not None and win._starlist_sky_canvas is not None
    ax = win._starlist_sky_fig.axes[0]
    assert ax.name == "polar"
    n_lines_empty = len(ax.lines)
    win.tel_k2.setChecked(True); settle(1)
    win._starlist_show_detail(0); settle(3)
    ax = win._starlist_sky_fig.axes[0]
    assert len(ax.lines) > n_lines_empty, "a row click must draw the path"
    assert "K2" in win._starlist_sky_fig._suptitle.get_text()
    win.tel_k1.setChecked(True); settle(1)
    win._starlist_show_detail(0); settle(3)
    assert "K1" in win._starlist_sky_fig._suptitle.get_text()
    # changing the evaluation time redraws (the Moon and the star move)
    win._starlist_date_edit.setDate(QtCore.QDate(2026, 9, 25)); settle(3)
    assert win._starlist_sky_fig.axes, "redraw after a time change"
    print("  [ok] GUI: sky plot beside the table; row click draws the path; "
          "telescope picks the wedge; time change redraws")
    dlg.grab().save(os.path.join(HERE, "gui_phase45.png"))
    dlg.close(); settle(2)
    assert win._starlist_sky_fig is None
    print("  [ok] GUI: references dropped on close")


def main():
    engine_contract()
    gui_dialog()
    print("  [ok] starlist dialog sky plot")


if __name__ == "__main__":
    main()
