#!/usr/bin/env python3
"""Field-map catalogue clicks and Rank in a Prediction scenario with NO night
Run (2026-09-28, Eduardo Marin: "in the predicted mode, when I load the
catalog I lose the ability to right click on stars even though the circles
are displayed"; Rank's table did not open either).

Since 2026-08-12 the field map renders a Prediction scenario without a Run
(no-run surrogate args/prep), but the canvas click handler still returned on
`self.res is None` and Rank on `self.prep is None or self.res is None`, both
silently. Contract:
  * no Run + Prediction on: right-click offers "Set <star> as TT/NGS star"
    and the action sets it; left-click inspects; Rank ranks and opens the
    table (against the same surrogate prep the map is drawn with);
  * no Run + Prediction off: clicks stay inert and Rank SAYS why instead of
    doing nothing.
Headless, offline (synthetic catalogue stars).
"""
import os, sys, time
from types import SimpleNamespace
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
from qtcompat import QtWidgets, QtCore
import astropy.units as u
from astropy.coordinates import SkyCoord
import keck_ao_estimator.gui as gui


def pump(cond, timeout=60):
    """Nested-QEventLoop wait (see gui_phase12.pump for why not msleep)."""
    loop = QtCore.QEventLoop()
    t0 = time.time()
    done = False

    def check():
        nonlocal done
        if done:
            return
        if cond() or time.time() - t0 > timeout:
            done = True
            loop.quit()
        else:
            QtCore.QTimer.singleShot(10, check)
    QtCore.QTimer.singleShot(0, check)
    loop.exec()
    done = True


def main():
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.resize(1500, 950); win.show(); app.processEvents()
    assert win.res is None and win.prep is None, "must start with no Run"

    def idle():
        return (not win._fm_debounce.isActive()
                and not win._fm_settle.isActive())
    win.plot_tabs.setCurrentIndex(1); app.processEvents(); pump(idle)

    fc = win._field_center_deg()
    assert fc is not None, "the default target defines the field centre"
    cen = SkyCoord(fc[0] * u.deg, fc[1] * u.deg)
    a = cen.spherical_offsets_by(5 * u.arcsec, 3 * u.arcsec)     # 5"E 3"N
    b = cen.spherical_offsets_by(-10 * u.arcsec, 0 * u.arcsec)   # 10"W
    stars = [{"id": "A", "ra": float(a.ra.deg), "dec": float(a.dec.deg),
              "mags": {"R": 11.0}},
             {"id": "B", "ra": float(b.ra.deg), "dec": float(b.dec.deg),
              "mags": {"R": 12.5}}]
    win.tt_sensor.setCurrentText("STRAP (R)")

    # capture the right-click menu instead of blocking on QMenu.exec
    menus = []
    orig_exec = QtWidgets.QMenu.exec

    def fake_exec(menu, *a, **k):
        menus.append({act.text(): act for act in menu.actions()})
        return None
    QtWidgets.QMenu.exec = fake_exec
    ax = win._fm_holder["canvas"].figure.axes[0]

    def click(x, y, button):
        win._on_fm_canvas_click(SimpleNamespace(inaxes=ax, xdata=x, ydata=y,
                                                button=button))
        app.processEvents()

    try:
        # ---- no Run, Prediction OFF: inert, and Rank says why ----------------
        win.pred_enable.setChecked(False); app.processEvents(); pump(idle)
        win._on_catalog_loaded("Gaia DR2", stars, ""); pump(idle)
        click(-5.0, 3.0, 3)
        assert not menus, "no Run + no prediction: the map is a placeholder"
        win._rank_guide_stars(); app.processEvents()
        assert win._gs_rank_dialog is None
        assert "Run first" in win.fm_catalog_status.text(), \
            win.fm_catalog_status.text()
        print("  [ok] no Run, prediction off: clicks inert, Rank explains")

        # ---- no Run, Prediction ON: right-click, inspect, Rank all work ------
        win.pred_enable.setChecked(True); app.processEvents(); pump(idle)
        assert win.res is None, "prediction must not need a Run"
        # plot frame: x = West+, so A (5"E) sits at x = -5
        click(-5.0, 3.0, 3)
        assert menus, "right-click menu must open in prediction mode"
        items = menus[-1]
        assert "Set “A” as TT star" in items and "Set “A” as NGS star" in items, \
            list(items)
        items["Set “A” as TT star"].trigger(); pump(idle)
        txy = win.tt_offset.offset_xy(win._sky_field_center())
        assert abs(txy[0] + 5.0) < 0.1 and abs(txy[1] - 3.0) < 0.1, txy
        assert abs(win.tt_mag.value() - 11.0) < 1e-6, win.tt_mag.value()
        print("  [ok] prediction, no Run: right-click menu offers the star; "
              "'Set as TT star' sets position + R")

        click(10.0, 0.0, 1)                        # left-click B: inspect
        assert win._catalog_inspected == "B", win._catalog_inspected
        print("  [ok] prediction, no Run: left-click inspects")

        for mode in ("NGS", "single-LGS", "LTAO"):
            win.fm_mode.setCurrentText(mode); app.processEvents(); pump(idle)
            win._gs_rank_dialog = None
            win._rank_guide_stars(); pump(idle)
            assert win._gs_rank_dialog is not None, f"{mode}: table must open"
            ranked = [e for e in win._gs_ranking if e["rank"] is not None]
            assert {e["id"] for e in ranked} == {"A", "B"}, win._gs_ranking
            assert "ranked 2/2" in win.fm_catalog_status.text(), \
                win.fm_catalog_status.text()
        print("  [ok] prediction, no Run: Rank ranks and opens the table "
              "(NGS / single-LGS / LTAO)")
    finally:
        QtWidgets.QMenu.exec = orig_exec
    win.close()
    print("gui_phase47: PASS")


if __name__ == "__main__":
    main()
