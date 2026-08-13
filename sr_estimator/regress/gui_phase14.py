#!/usr/bin/env python3
"""FWHM reporting: Report combo drives strehl/fwhm/both live; FWHM figure and
overlay correct; CSV export gains FWHM columns; projection gated off in fwhm
mode. Run headless."""
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
        app.processEvents(); QtCore.QThread.msleep(10)


def main():
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.resize(1500, 950); win.show(); app.processEvents()
    win.mode_local.setChecked(True)
    win.dimm_edit.setText(os.path.join(DATA, "20260525_dimm.dat"))
    win.mass_edit.setText(os.path.join(DATA, "20260525_mass.dat"))
    win.masspro_edit.setText(os.path.join(DATA, "20260525_masspro.dat"))
    win.tel_k1.setChecked(True); win._validate(); win.on_run()
    pump(lambda: win.res is not None)

    # default: Strehl mode, no FWHM arrays computed (fast path untouched)
    assert win.report_combo.currentText() == "Strehl"
    assert win.args_cached.report == "strehl"
    assert win.res.fwhm_ngs_bright is None, "fwhm must not be computed by default"
    print("  [ok] default Strehl mode: no FWHM arrays computed")

    # switch to FWHM: live recompute, panel now in mas
    prev = win.res
    win.report_combo.setCurrentText("FWHM")
    pump(lambda: win.res is not prev, timeout=15)
    assert win.args_cached.report == "fwhm"
    fig = win._main_holder["canvas"].figure
    assert "FWHM" in fig.axes[0].get_ylabel(), fig.axes[0].get_ylabel()
    fw = win.res.fwhm_ngs_bright
    dl = 1.029 * (win.prep.lam_nm * 1e-9) / engine.TEL_DIAMETER_M * 206265e3
    med = float(np.nanmedian(fw))
    assert dl * 0.95 < med < dl * 1.6, f"NGS median FWHM {med:.0f} vs dl {dl:.0f}"
    print(f"  [ok] FWHM mode live: NGS median {med:.0f} mas (diffraction {dl:.0f})")

    # axis contract: linear, mas, seeing disk NOT plotted, bounded to the
    # non-collapsed population so the structure is readable
    for i, ax in enumerate(fig.axes):
        assert ax.get_yscale() == "linear", f"axes[{i}] must be linear"
        assert ax.get_ylabel().endswith("(mas)"), ax.get_ylabel()
        lo, hi = ax.get_ylim()
        assert hi <= engine.FWHM_COLLAPSE_MULT * dl * 1.1, \
            f"axes[{i}] hi={hi:.0f} not bounded below the collapsed population"
    labels = [l.get_label() for ax in fig.axes for l in ax.get_lines()]
    assert not any("seeing disk" in s for s in labels), \
        "seeing disk must be text, not a plotted curve"
    # DEFAULT CHANGED 2026-08-07 (Eduardo): the FWHM figure now defaults to
    # "as the SR tool reads it" -- the convention this app's own Measured-SR
    # tab reports in, and the only one comparable to a measured FWHM without
    # a convention caveat. It is also first in the combo, so it is the
    # default by position, not by a separate setCurrentIndex.
    assert win.fwhm_curves_combo.currentIndex() == 0
    assert win.fwhm_curves_combo.currentText() == "as the SR tool reads it"
    assert any("as the SR tool reads it" in s for s in labels), labels
    assert not any("Gaussian-fit sim" in s for s in labels), \
        "the default must draw ONE convention, not the Gaussian-fit ones"
    assert not any("core+halo model" in s for s in labels), \
        "the default is no longer half-max"
    print("  [ok] default FWHM figure shows the SR-tool convention only")

    # switching the FWHM-curves selector draws the Gaussian-fit convention
    prev = win.res
    win.fwhm_curves_combo.setCurrentText("both curves")
    pump(lambda: win.res is not prev, timeout=15)
    labels = [l.get_label() for ax in win._main_holder["canvas"].figure.axes
              for l in ax.get_lines()]
    assert any("Gaussian-fit sim" in s for s in labels), \
        "'both curves' should add the Gaussian-fit convention"
    print("  [ok] FWHM-curves selector adds the Gaussian-fit convention")

    # third convention (free-background fit) and the box-size control
    assert win.fwhm_box_mas.isEnabled(), "box control should be enabled in FWHM mode"
    prev = win.res
    win.fwhm_curves_combo.setCurrentText("Gaussian-fit (+background)")
    pump(lambda: win.res is not prev, timeout=15)
    labels = [l.get_label() for ax in win._main_holder["canvas"].figure.axes
              for l in ax.get_lines()]
    assert any("+background" in s for s in labels), \
        "'Gaussian-fit (+background)' should draw the free-background convention"
    print("  [ok] FWHM-curves selector offers the free-background convention")
    prev = win.res
    win.fwhm_curves_combo.setCurrentText("all four")
    pump(lambda: win.res is not prev, timeout=15)
    labels = [l.get_label() for ax in win._main_holder["canvas"].figure.axes
              for l in ax.get_lines()]
    assert any("core+halo model" in s for s in labels) and \
           any("+background" not in s and "Gaussian-fit sim" in s for s in labels) and \
           any("+background" in s for s in labels) and \
           any("as the SR tool reads it" in s for s in labels), \
        "'all four' should draw all four conventions"
    print("  [ok] 'all four' draws half-max + both Gaussian-fit conventions "
          "+ the SR-tool convention")

    # the 4th convention on its own (2026-08-07): the one directly
    # comparable to a measured FWHM, so it must be selectable alone
    prev = win.res
    win.fwhm_curves_combo.setCurrentText("as the SR tool reads it")
    pump(lambda: win.res is not prev, timeout=15)
    labels = [l.get_label() for ax in win._main_holder["canvas"].figure.axes
              for l in ax.get_lines()]
    assert any("as the SR tool reads it" in s for s in labels), \
        "the SR-tool convention should draw on its own"
    assert not any("core+halo model" in s for s in labels), \
        "selecting one convention must not also draw half-max"
    assert win.res.fwhm_tool_ngs_bright is not None, \
        "the timeline must carry the SR-tool FWHM series"
    import numpy as _np
    assert _np.nanmedian(win.res.fwhm_tool_ngs_bright) > \
           _np.nanmedian(win.res.fwhm_ngs_bright), \
        "the SR-tool convention reads WIDER than half-max (1-px annulus " \
        "binning lowers the apparent peak) -- see psf.fwhm_srtool_mas"
    print("  [ok] 'as the SR tool reads it' draws alone and reads wider "
          "than half-max")

    # box_mas actually changes the fit-based curves (a real, adjustable knob —
    # the real AO Strehl tool's fit box is hand-drawn, not fixed)
    box_prev = list(win.res.fwhm_gauss_ngs_bright)
    prev = win.res
    win.fwhm_box_mas.setValue(30.0)
    pump(lambda: win.res is not prev, timeout=15)
    assert not np.allclose(win.res.fwhm_gauss_ngs_bright, box_prev, equal_nan=True), \
        "changing the fit box size should change the Gaussian-fit FWHM"
    print("  [ok] fit box-size control changes the Gaussian-fit FWHM values")
    win.fwhm_box_mas.setValue(300.0)

    prev = win.res; win.fwhm_curves_combo.setCurrentText("half-max")
    pump(lambda: win.res is not prev, timeout=15)
    fig = win._main_holder["canvas"].figure
    # each axis must hug its own non-collapsed data -- that is what makes the
    # panel readable, and it adapts as the physics changes
    thr = engine.FWHM_COLLAPSE_MULT * dl
    panels = ((0, [win.res.fwhm_ngs_bright, win.res.fwhm_ngs_faint,
                   win.res.fwhm_gauss_ngs_bright]),
              (1, [win.res.fwhm_single, win.res.fwhm_ltao,
                   win.res.fwhm_gauss_single, win.res.fwhm_gauss_ltao]))
    for i, arrs in panels:
        lo, hi = fig.axes[i].get_ylim()
        cat = np.concatenate([np.asarray(a, float) for a in arrs if a is not None])
        cat = cat[np.isfinite(cat)]
        good = cat[cat <= thr]
        assert good.max() <= hi <= good.max() * 1.15 + 1e-6, \
            f"axes[{i}] hi={hi:.1f} does not hug non-collapsed max {good.max():.1f}"
    lo0, hi0 = fig.axes[0].get_ylim(); lo1, hi1 = fig.axes[1].get_ylim()
    print(f"  [ok] linear mas axes hug the data, no seeing curve, spans "
          f"{hi0-lo0:.0f}/{hi1-lo1:.0f} mas")

    # the 2026-07-10 tilt-as-jitter fix: FWHM must RESPOND to conditions.
    # Before it, NGS sat frozen at 1.029 lam/D (spread 1.01x).
    ngs = np.asarray(win.res.fwhm_ngs_bright, float); ngs = ngs[np.isfinite(ngs)]
    see = (np.asarray(win.res.col_dimm, float) * np.asarray(win.res.col_zf, float))
    assert ngs.max() / ngs.min() > 1.05, \
        f"NGS FWHM frozen ({ngs.max()/ngs.min():.3f}x) — tilt jitter lost?"
    rr = np.corrcoef(see[np.isfinite(np.asarray(win.res.fwhm_ngs_bright,float))], ngs)[0, 1]
    assert rr > 0.5, f"NGS FWHM should track seeing (r={rr:.2f})"
    print(f"  [ok] FWHM responds to conditions: NGS spread "
          f"{ngs.max()/ngs.min():.2f}x, r(seeing)={rr:+.3f}")
    # collapsed samples are labelled rather than silently truncated
    notes = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert any("above axis" in t and "core lost" in t for t in notes), \
        "clipped samples must be annotated"
    print("  [ok] clipped (core-lost) samples annotated on-plot")

    # MODIFIED BUDGET in fwhm mode: box drawn, but no SR-axis NGS projection
    r = win.wfe_rows["STATIC_CALIB"]; prev = win.res
    r["spin"].setValue(r["default"] + 80)
    pump(lambda: win.res is not prev, timeout=15)
    fig = win._main_holder["canvas"].figure
    texts = [t.get_text() for ax in fig.axes for t in ax.texts] + \
            [t.get_text() for t in fig.texts]
    assert any("MODIFIED BUDGET" in t for t in texts), "indicator missing"
    assert not any(ln.get_label().startswith("projected NGS")
                   for ax in fig.axes for ln in ax.get_lines()), \
        "SR-axis projection must be skipped in fwhm mode"
    print("  [ok] fwhm mode: MODIFIED BUDGET shown, SR projection skipped")
    win._reset_all_wfe(); prev = win.res
    pump(lambda: win.res is not prev, timeout=15)

    # both: Strehl figure + right-hand FWHM axes appended
    prev = win.res
    win.report_combo.setCurrentText("Strehl + FWHM")
    pump(lambda: win.res is not prev, timeout=15)
    fig = win._main_holder["canvas"].figure
    assert "Strehl" in fig.axes[0].get_ylabel(), "both must keep the SR panel"
    fwhm_axes = [ax for ax in fig.axes if "FWHM" in ax.get_ylabel()]
    assert len(fwhm_axes) >= 2, f"expected 2 twin FWHM axes, got {len(fwhm_axes)}"
    print("  [ok] both mode: SR panels + twin FWHM axes")

    # CSV export in both mode carries FWHM columns + model provenance
    dest = os.path.join(HERE, "p14_export.csv")
    QtWidgets.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (dest, ""))
    win.on_export_csv()
    txt = open(dest).read()
    assert "ltao_fwhm_mas" in txt, "FWHM columns missing from export"
    assert "ltao_fwhm_gaussfit_mas" in txt and "fwhm_gaussfit=simulated" in txt, \
        "Gaussian-equivalent columns/provenance missing"
    assert "ltao_fwhm_gaussfit_sky_mas" in txt and "fwhm_gaussfit_sky=simulated" in txt, \
        "free-background Gaussian-fit columns/provenance missing"
    assert "fwhm_model=airy_core" in txt and "gauss_tt_jitter" in txt, \
        "provenance must state the tilt-jitter convolution"
    assert "S_ho=S_total/marechal(tt)" in txt, "core-energy split not recorded"
    assert "ngs_tilt_servo=" in txt and "strehl budget unchanged" in txt
    print("  [ok] exported CSV has FWHM columns + tilt-jitter model provenance")
    os.remove(dest)

    # PNG export in fwhm mode
    win.report_combo.setCurrentText("FWHM")
    prev = win.res; pump(lambda: win.res is not prev, timeout=15)
    png = os.path.join(HERE, "p14_export.png")
    QtWidgets.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (png, ""))
    win.on_export_png()
    assert os.path.exists(png) and os.path.getsize(png) > 10000
    print(f"  [ok] fwhm-mode PNG export ({os.path.getsize(png)} bytes)")
    os.remove(png)
    assert "Exported PNG" in win.status.text()

    # --- Export PNG exports the plot you are LOOKING AT (2026-07-21):
    #     Field map / Error terms tabs save their on-screen figure ----------
    for tab_idx, label in ((1, "field map"), (2, "error terms")):
        win.plot_tabs.setCurrentIndex(tab_idx)
        pump(lambda: True, timeout=1)          # let the lazy tab render kick in
        QtWidgets.QApplication.processEvents()
        dest_t = os.path.join(HERE, f"p14_export_tab{tab_idx}.png")
        QtWidgets.QFileDialog.getSaveFileName = staticmethod(
            lambda *a, dest_t=dest_t, **k: (dest_t, ""))
        win.on_export_png()
        assert os.path.exists(dest_t) and os.path.getsize(dest_t) > 5000, \
            f"{label} export missing/empty"
        assert label in win.status.text(), \
            f"status must say WHICH plot was exported: {win.status.text()}"
        os.remove(dest_t)
        print(f"  [ok] Export PNG on the {label} tab exports that tab's figure")
    win.plot_tabs.setCurrentIndex(0)

    win.grab().save(os.path.join(HERE, "gui_phase14.png"))
    print("  [ok] screenshot saved")


if __name__ == "__main__":
    main()
