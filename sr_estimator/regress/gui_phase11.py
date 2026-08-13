#!/usr/bin/env python3
"""Swap-model NGS projection (K2 fitting -> K1's value) and the K1 NGS fit
knobs (seeing exponent A, quadcell penalty). Run headless."""
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


def proj_curve(win):
    ax0 = win._main_holder["canvas"].figure.axes[0]
    for ln in ax0.get_lines():
        if ln.get_label().startswith("projected NGS"):
            return ln
    return None


def main():
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.show(); app.processEvents()
    win.mode_local.setChecked(True)
    win.dimm_edit.setText(os.path.join(DATA, "20260525_dimm.dat"))
    win.mass_edit.setText(os.path.join(DATA, "20260525_mass.dat"))
    win.masspro_edit.setText(os.path.join(DATA, "20260525_masspro.dat"))
    win.tel_k2.setChecked(True); win._validate(); win.on_run()
    pump(lambda: win.res is not None)

    lam = win.prep.lam_nm
    ngs_k2 = float(np.nanmean(win.res.ngs_bright))
    fit_k2 = win.wfe_rows["FITTING_ERR_K2"]["default"]
    fit_k1 = win.wfe_rows["FITTING_ERR_K1"]["default"]

    # --- swap K2 fitting -> K1's 141 nm; NGS must move substantially ---------
    prev = win.res
    win.wfe_rows["FITTING_ERR_K2"]["spin"].setValue(fit_k1)
    pump(lambda: win.res is not prev, timeout=8)
    ln = proj_curve(win)
    assert ln is not None, "no projected NGS after fitting swap"
    proj_mean = float(np.nanmean(np.asarray(ln.get_ydata(), float)))
    eff = np.sqrt(fit_k1 ** 2 - fit_k2 ** 2)
    # expected mean drop from the Marechal factor of the quadrature swap at
    # the run's wavelength -- tied to the live budget values, NOT a frozen
    # number (the old fixed 0.06 encoded the v3_1_1 swap 60->141; under
    # v3_1_3's 48.1->97 the CORRECT movement is ~0.035, indistinguishable
    # from the historical too-small-response bug by any fixed threshold)
    exp_drop = ngs_k2 * (1.0 - np.exp(
        -((2 * np.pi / lam) ** 2) * (fit_k1 ** 2 - fit_k2 ** 2)))
    print(f"  [ok] K2 fitting {fit_k2:g}->{fit_k1:g}: eff swap {eff:.1f} nm "
          f"(delta model would be {fit_k1-fit_k2:g}); "
          f"NGS {ngs_k2:.4f} -> {proj_mean:.4f} "
          f"(expected drop ~{exp_drop:.4f})")
    assert eff > (fit_k1 - fit_k2), "swap must exceed the naive delta"
    assert (ngs_k2 - proj_mean) > 0.6 * exp_drop, \
        f"NGS moved {ngs_k2 - proj_mean:.4f}, expected ~{exp_drop:.4f} " \
        "from the Marechal factor of the swap"
    assert "⊕" in ln.get_label(), ln.get_label()

    # --- lowering a term below default must IMPROVE the projected NGS --------
    win._reset_all_wfe(); prev = win.res
    pump(lambda: win.res is not prev, timeout=8)
    prev = win.res
    win.wfe_rows["STATIC_CALIB"]["spin"].setValue(win.wfe_rows["STATIC_CALIB"]["default"] - 50)
    pump(lambda: win.res is not prev, timeout=8)
    ln = proj_curve(win)
    assert ln is not None and "⊖" in ln.get_label(), \
        f"reducing a term should show ⊖: {ln and ln.get_label()}"
    better = float(np.nanmean(np.asarray(ln.get_ydata(), float)))
    assert better > ngs_k2, f"lowering static should raise NGS ({better:.4f} vs {ngs_k2:.4f})"
    print(f"  [ok] lowering STATIC_CALIB by 50nm improves NGS {ngs_k2:.4f} -> {better:.4f} "
          f"({ln.get_label()})")
    win._reset_all_wfe(); prev = win.res
    pump(lambda: win.res is not prev, timeout=8)

    # --- NGS Gompertz fit editor: telescope-aware + fully editable -----------
    # on K2 the fields show the K2 fit; the K1-only quadcell is disabled
    assert abs(win.ngs_a.value() - engine.NGS_PARAMS["K2"]["A"]) < 1e-9
    assert abs(win.ngs_s0.value() - engine.NGS_PARAMS["K2"]["S0"]) < 1e-9
    assert not win.k1_quadcell.isEnabled(), "quadcell is K1-only"
    print("  [ok] Gompertz fields seeded from K2 fit; quadcell disabled on K2")

    # switching telescope repopulates the fit to K1's values
    prev_prep = win.prep
    win.tel_k1.setChecked(True)
    pump(lambda: win.prep is not prev_prep, timeout=30)
    assert abs(win.ngs_a.value() - engine.NGS_PARAMS["K1"]["A"]) < 1e-9, \
        "fit fields must repopulate to the active telescope"
    assert win.k1_quadcell.isEnabled()
    print("  [ok] telescope switch repopulated Gompertz fields to K1")

    base = float(np.nanmean(win.res.ngs_bright))
    prev = win.res
    win.k1_quadcell.setValue(0.0)               # remove the flat 0.05 penalty
    pump(lambda: win.res is not prev, timeout=8)
    noquad = float(np.nanmean(win.res.ngs_bright))
    assert abs((noquad - base) - engine.NGS_K1_QUADCELL_PENALTY) < 2e-3, \
        f"removing quadcell should lift NGS by ~{engine.NGS_K1_QUADCELL_PENALTY}"
    print(f"  [ok] quadcell 0.05 -> 0: K1 NGS {base:.4f} -> {noquad:.4f}")

    # every Gompertz term is live: A (seeing), S0 (ceiling), m0 (faint knee)
    prev = win.res
    win.ngs_a.setValue(engine.NGS_PARAMS["K2"]["A"])   # shallower exponent
    pump(lambda: win.res is not prev, timeout=8)
    a_lift = float(np.nanmean(win.res.ngs_bright))
    assert a_lift > noquad, "shallower seeing exponent should raise K1 NGS"
    prev = win.res
    win.ngs_s0.setValue(win.ngs_s0.value() + 0.10)     # raise the ceiling
    pump(lambda: win.res is not prev, timeout=8)
    s0_lift = float(np.nanmean(win.res.ngs_bright))
    assert s0_lift > a_lift, "raising S0 should raise NGS"
    assert win.args_cached.ngs_a == engine.NGS_PARAMS["K2"]["A"]
    assert win.args_cached.ngs_s0 > engine.NGS_PARAMS["K1"]["S0"]
    print(f"  [ok] Gompertz terms live: A -> {a_lift:.4f}, +S0 -> {s0_lift:.4f}")
    assert engine.NGS_PARAMS["K1"]["A"] == 1.00, "module NGS_PARAMS mutated!"
    assert engine.NGS_PARAMS["K1"]["S0"] == 0.61
    print("  [ok] engine NGS_PARAMS not mutated")

    # the NGS fit preview plot must exist and redraw live as terms change
    ax = win.fit_fig.axes[0]
    assert len(ax.get_lines()) >= 3, "fit preview should show seeing curves"
    y0 = ax.get_lines()[0].get_ydata().copy()
    win.ngs_m0.setValue(win.ngs_m0.value() - 2.0)   # brighter faint knee
    QtWidgets.QApplication.instance().processEvents()
    y1 = win.fit_fig.axes[0].get_lines()[0].get_ydata()
    assert (y1 != y0).any(), "fit preview did not redraw on m0 change"
    print("  [ok] NGS fit preview plots and updates live")

    # preview follows the science wavelength (extrapolated from the K-band fit)
    win.band_combo.setCurrentText("K")
    QtWidgets.QApplication.instance().processEvents()
    tK = win.fit_fig.axes[0].get_title()
    yK = win.fit_fig.axes[0].get_lines()[0].get_ydata().copy()
    win.band_combo.setCurrentText("H")
    QtWidgets.QApplication.instance().processEvents()
    axH = win.fit_fig.axes[0]
    assert "extrapolated" in axH.get_title() and "H-band" in axH.get_title()
    assert "extrapolated" not in tK, "K-band should not be flagged as extrapolated"
    assert axH.get_lines()[0].get_ydata()[0] < yK[0], "H Strehl should be < K"
    lamH = engine.PHOTOMETRIC_BANDS["H"]
    exp = engine.ngs_strehl(0.3 / engine.V2K, 6.0, "K1", lamH,
                            ngs_s0=win.ngs_s0.value(), ngs_a=win.ngs_a.value(),
                            ngs_m0=win.ngs_m0.value(), ngs_w=win.ngs_w.value(),
                            k1_quadcell=win.k1_quadcell.value(),
                            seeing_law=win.seeing_law.currentText())
    assert abs(axH.get_lines()[0].get_ydata()[0] - exp) < 1e-9, "preview != engine"
    print(f"  [ok] preview tracks wavelength (K {tK.split('—')[1].strip()} -> H, matches engine)")


if __name__ == "__main__":
    main()
