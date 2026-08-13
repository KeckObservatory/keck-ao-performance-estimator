#!/usr/bin/env python3
"""NGS-projection test: adding budget nm projects onto NGS via Maréchal ⊕.
Run headless."""
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


def main():
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.show(); app.processEvents()
    win.mode_local.setChecked(True)
    win.dimm_edit.setText(os.path.join(DATA, "20260525_dimm.dat"))
    win.mass_edit.setText(os.path.join(DATA, "20260525_mass.dat"))
    win.masspro_edit.setText(os.path.join(DATA, "20260525_masspro.dat"))
    win.tel_k1.setChecked(True); win._validate(); win.on_run()
    pump(lambda: win.res is not None)

    # add exactly 100 nm of static (default 91 -> 191)
    default = win.wfe_rows["STATIC_CALIB"]["default"]
    prev = win.res
    win.wfe_rows["STATIC_CALIB"]["spin"].setValue(default + 100.0)
    pump(lambda: win.res is not prev, timeout=8)
    assert win.current_offsets().get("STATIC_CALIB") == default + 100.0

    # find the projected-NGS line on the NGS panel
    ax0 = win._main_holder["canvas"].figure.axes[0]
    proj = [ln for ln in ax0.get_lines()
            if ln.get_label().startswith("projected NGS")]
    assert proj, "projected NGS curve not drawn"
    print("  [ok] projected NGS curve present:", proj[0].get_label())

    # spot-check the SWAP math: the term is already inside the NGS Strehl, so
    # sigma'^2 = sigma^2 - default^2 + new^2  (NOT sigma^2 + delta^2)
    lam = win.prep.lam_nm
    sr = np.asarray(win.res.ngs_bright, float)
    ok = np.isfinite(sr) & (sr > 0)
    sig = (lam / (2 * np.pi)) * np.sqrt(-np.log(np.clip(sr[ok], 1e-6, 1)))
    dvar = (default + 100.0) ** 2 - default ** 2          # STATIC_CALIB swapped
    exp_proj = np.exp(-(2 * np.pi * np.sqrt(sig ** 2 + dvar) / lam) ** 2)
    got_proj = np.asarray(proj[0].get_ydata(), float)[ok]
    assert np.allclose(got_proj, exp_proj, atol=1e-6), "projection math mismatch"
    eff = np.sqrt(dvar)
    print(f"  [ok] swap STATIC_CALIB {default:g}->{default+100:g} (⊕{eff:.0f} nm eff): "
          f"NGS mean {np.nanmean(sr):.3f} -> {np.nanmean(exp_proj):.3f}")

    win._main_holder["canvas"].draw()
    for _ in range(6):
        app.processEvents(); QtCore.QThread.msleep(20)
    win.grab().save(os.path.join(HERE, "gui_phase7.png"))
    print("  [ok] screenshot saved")


if __name__ == "__main__":
    main()
