#!/usr/bin/env python3
"""Tests for: wind live recompute, common-path-only NGS projection, and the
UT-correct 'last night' preset. Run headless."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
from qtcompat import QtWidgets, QtCore
import keck_ao_estimator as engine
import keck_ao_estimator.gui as gui
np = engine.np
DATA = os.path.join(HERE, "data")


def pump(cond, timeout=90):
    app = QtWidgets.QApplication.instance(); t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents(); QtCore.QThread.msleep(20)


def ngs_proj_lines(win):
    ax0 = win._main_holder["canvas"].figure.axes[0]
    return [ln for ln in ax0.get_lines()
            if ln.get_label().startswith("projected NGS")]


def main():
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.show(); app.processEvents()
    win.mode_local.setChecked(True)
    win.dimm_edit.setText(os.path.join(DATA, "20260525_dimm.dat"))
    win.mass_edit.setText(os.path.join(DATA, "20260525_mass.dat"))
    win.masspro_edit.setText(os.path.join(DATA, "20260525_masspro.dat"))
    win.tel_k1.setChecked(True); win._validate(); win.on_run()
    pump(lambda: win.res is not None)

    # default wind widgets seeded from parser
    assert win.wind_ground.value() == 8.0 and win.wind_free.value() == 25.0
    print("  [ok] wind fields default 8 / 25 m/s")

    # 1) wind live recompute (compute-only)
    s0 = float(np.nanmean(win.res.sr_single)); prev = win.res
    win.wind_free.setValue(60.0)
    pump(lambda: win.res is not prev, timeout=8)
    assert win.res is not prev, "wind change did not recompute"
    s1 = float(np.nanmean(win.res.sr_single))
    assert s1 < s0, f"faster jet should lower single-LGS ({s0:.4f}->{s1:.4f})"
    assert win.args_cached.wind_free == 60.0, "wind not mapped into args"
    print(f"  [ok] live wind recompute: single-LGS {s0:.4f} -> {s1:.4f} (jet 60 m/s)")
    win.wind_free.setValue(25.0); prev = win.res
    pump(lambda: win.res is not prev, timeout=8)

    # 2) NGS projection is COMMON-PATH only
    #    LGS-only term (FA_REF) -> no projected NGS curve
    win.wfe_rows["FA_REF"]["spin"].setValue(win.wfe_rows["FA_REF"]["default"] + 80)
    prev = win.res; pump(lambda: win.res is not prev, timeout=8)
    assert not ngs_proj_lines(win), "FA_REF (LGS-only) must NOT project onto NGS"
    print("  [ok] LGS-only FA_REF change -> no NGS projection")
    #    common-path term (STATIC_CALIB) -> projected NGS curve appears
    win.wfe_rows["STATIC_CALIB"]["spin"].setValue(win.wfe_rows["STATIC_CALIB"]["default"] + 100)
    prev = win.res; pump(lambda: win.res is not prev, timeout=8)
    lines = ngs_proj_lines(win)
    assert lines, "STATIC_CALIB (common-path) should project onto NGS"
    print("  [ok] common-path STATIC_CALIB change -> NGS projection:", lines[0].get_label())
    win._reset_all_wfe(); prev = win.res; pump(lambda: win.res is not prev, timeout=8)

    # 3) 'Last night' preset uses UT yesterday, not local civil date
    win._preset_last_night()
    assert win.mode_fetch.isChecked(), "preset should switch to fetch mode"
    exp = QtCore.QDateTime.currentDateTimeUtc().date().addDays(-1)
    got = win.fetch_date.date()
    assert got == exp, f"last-night date {got.toString('yyyyMMdd')} != UT-1 {exp.toString('yyyyMMdd')}"
    print(f"  [ok] last-night preset = UT yesterday: {got.toString('yyyyMMdd')}")

    # 4) 'Tonight's data' preset uses UT today (complements 'Last night')
    win.mode_local.setChecked(True)                # start from local, like a fresh app
    win._preset_tonight()
    assert win.mode_fetch.isChecked(), "preset should switch to fetch mode"
    exp = QtCore.QDateTime.currentDateTimeUtc().date()
    got = win.fetch_date.date()
    assert got == exp, f"tonight date {got.toString('yyyyMMdd')} != UT today {exp.toString('yyyyMMdd')}"
    print(f"  [ok] tonight preset = UT today: {got.toString('yyyyMMdd')}")

    print("  [ok] all phase-8 checks passed")


if __name__ == "__main__":
    main()
