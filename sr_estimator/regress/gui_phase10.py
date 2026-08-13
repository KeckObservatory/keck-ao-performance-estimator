#!/usr/bin/env python3
"""Corrected NGS-projection rules: NGS is affected by every budget term EXCEPT
the LGS-specific ones (focal aniso, Na focus, LTAO tomography), the inactive
telescope's DM fitting, and angular aniso unless off-axis. Run headless."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
from qtcompat import QtWidgets, QtCore
import keck_ao_estimator.gui as gui
DATA = os.path.join(HERE, "data")


def pump(cond, timeout=90):
    app = QtWidgets.QApplication.instance(); t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents(); QtCore.QThread.msleep(20)


def has_proj(win):
    ax0 = win._main_holder["canvas"].figure.axes[0]
    return any(ln.get_label().startswith("projected NGS")
               for ln in ax0.get_lines())


def set_term(win, name, delta):
    r = win.wfe_rows[name]
    prev = win.res
    r["spin"].setValue(r["default"] + delta)
    pump(lambda: win.res is not prev, timeout=8)
    r2 = win.res
    r["spin"].setValue(r["default"])          # reset for next case
    pump(lambda: win.res is not r2, timeout=8)


def check(win, name, delta, expect, label):
    r = win.wfe_rows[name]; prev = win.res
    r["spin"].setValue(r["default"] + delta)
    pump(lambda: win.res is not prev, timeout=8)
    got = has_proj(win)
    r2 = win.res; r["spin"].setValue(r["default"])
    pump(lambda: win.res is not r2, timeout=8)
    assert got == expect, f"{name}: projection={got}, expected {expect} ({label})"
    print(f"  [ok] {name:16s} -> NGS projection {got}  ({label})")


def main():
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.show(); app.processEvents()
    win.mode_local.setChecked(True)
    win.dimm_edit.setText(os.path.join(DATA, "20260525_dimm.dat"))
    win.mass_edit.setText(os.path.join(DATA, "20260525_mass.dat"))
    win.masspro_edit.setText(os.path.join(DATA, "20260525_masspro.dat"))
    win.tel_k1.setChecked(True); win._validate(); win.on_run()
    pump(lambda: win.res is not None)

    # affects NGS
    check(win, "FITTING_ERR_K1", 40, True,  "DM fitting (active tel) is common")
    check(win, "BW_REF",         40, True,  "bandwidth is common")
    check(win, "SCINT_REF",      30, True,  "scintillation is common")
    check(win, "HOMEAS",         40, True,  "HO measurement is common")
    check(win, "STATIC_CALIB",         40, True,  "static is common")
    check(win, "MARGIN",         40, True,  "margin is common")
    # LGS-specific -> no NGS effect
    check(win, "FA_REF",         60, False, "focal aniso is LGS-only")
    check(win, "NAFOC",          40, False, "Na focus is LGS-only")
    check(win, "TOMO_ERR",       40, False, "LTAO tomography is LGS-only")
    # inactive telescope's fitting -> no effect on a K1 night
    check(win, "FITTING_ERR_K2", 40, False, "inactive-telescope fitting excluded")

    # angular aniso: only when off-axis
    check(win, "ANG_REF",        40, False, "ang aniso excluded on-axis (offset=0)")
    prev = win.res
    win.ngs_offset.setValue(20.0)             # off-axis
    pump(lambda: win.res is not prev, timeout=8)
    check(win, "ANG_REF",        40, True,  "ang aniso projects when off-axis")

    print("  [ok] all NGS-projection rules correct")


if __name__ == "__main__":
    main()
