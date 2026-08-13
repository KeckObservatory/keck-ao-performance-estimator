#!/usr/bin/env python3
"""Live-update test: NGS/Budget changes recompute live; prep-affecting controls
(zenith without target, tomography) re-prepare live. Run headless."""
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
    print("  [ok] initial run")

    # 1) NGS live recompute (compute-only, no worker) --------------------------
    #    Use an off-axis offset: angular anisoplanatism provably lowers NGS.
    ngs0 = float(np.nanmean(win.res.ngs_bright))
    prev = win.res
    win.ngs_offset.setValue(30.0)                 # 30" off-axis -> aniso loss
    pump(lambda: win.res is not prev, timeout=8)
    assert win.res is not prev, "NGS change did not recompute live"
    ngs1 = float(np.nanmean(win.res.ngs_bright))
    assert ngs1 < ngs0 - 1e-3, f"NGS Strehl did not fall ({ngs0:.3f}->{ngs1:.3f})"
    print(f"  [ok] live NGS recompute: mean {ngs0:.3f} -> {ngs1:.3f} (30\" off-axis)")
    win.ngs_offset.setValue(0.0)                  # restore on-axis
    prev = win.res
    pump(lambda: win.res is not prev, timeout=8)

    # 2) Zenith angle WITHOUT target -> live re-prepare ------------------------
    assert not win.target_enable.isChecked(), "target should be off"
    assert win.za_enable.isEnabled() and win.za_spin.isEnabled() is False
    prev_prep = win.prep
    win.za_enable.setChecked(True)                # ungated: usable w/o target
    win.za_spin.setValue(45.0)
    pump(lambda: win.prep is not prev_prep, timeout=30)
    assert win.prep is not prev_prep, "zenith change did not re-prepare"
    assert win.prep.fixed_zen_factor > 1.0, \
        f"zenith factor not applied: {win.prep.fixed_zen_factor}"
    print(f"  [ok] zenith w/o target re-prepared: seeing x{win.prep.fixed_zen_factor:.3f}")

    # 3) Tomography combo -> live re-prepare -----------------------------------
    assert win.prep.tomography_on, "K1 should default tomography ON"
    prev_prep = win.prep
    win.tomo_combo.setCurrentText("off")
    pump(lambda: win.prep is not prev_prep, timeout=30)
    assert win.prep is not prev_prep and not win.prep.tomography_on, \
        "tomography off did not take effect"
    print("  [ok] live tomography off re-prepared (LTAO panel -> single-beacon)")

    win.grab().save(os.path.join(HERE, "gui_phase6.png"))
    print("  [ok] screenshot saved")


if __name__ == "__main__":
    main()
