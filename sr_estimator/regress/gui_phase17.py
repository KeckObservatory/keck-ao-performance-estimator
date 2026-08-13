#!/usr/bin/env python3
"""Prediction tab: hypothetical-conditions scenario for the field map.
Engine contract (synthetic snapshot physics: profile round-trip, theta0
solve, LOS projection, clamping) + GUI (enable swaps the field-map snapshot,
worse seeing degrades the map, WFE overrides carry over, presets, config
round-trip). Run headless."""
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


def settle(n=6):
    app = QtWidgets.QApplication.instance()
    for _ in range(n):
        app.processEvents(); QtCore.QThread.msleep(25)


def engine_contract():
    # reference scenario: prior-shaped profile -> zero layer mismatch, and
    # the profile's own theta0 is what the snapshot reports
    s = engine.synthetic_field_snapshot(engine.REF_TOTAL, engine.REF_FREEATM)
    # the base shape is the physical MK free-atmosphere profile (tropopause-
    # dominated: the 8 km bin is the strongest free-atm layer, 4 km is weak),
    # NOT the reconstructor prior -- so a median night carries a real, nonzero
    # layer mismatch against the reconstructor
    assert s["synthetic"] and abs(s["alpha"]) < 1e-9
    assert 0.1 < s["m"] < 0.35, f"reference layer-mismatch {s['m']}"
    Jr = s["cn2_bins"]; h = engine.MASS_HEIGHTS_M / 1e3
    assert h[np.argmax(Jr)] == 8.0, "free-atm must peak at the 8 km tropopause"
    assert Jr[h == 4.0][0] < Jr[h == 8.0][0], "4 km layer must be weaker than 8 km"
    th_prof = engine.theta0_d0_from_profile(s["cn2_bins"], 0.0,
                                            engine.LAMBDA_K_NM)[0]
    assert abs(th_prof - s["theta0_k_zenith"]) < 1e-6
    # sum(J) round-trips the free-atm seeing through the Kolmogorov relation
    k2 = (2 * np.pi / 500e-9) ** 2
    r0 = (0.423 * k2 * s["cn2_bins"].sum()) ** (-3.0 / 5.0)
    eps = np.degrees(0.98 * 500e-9 / r0) * 3600
    assert abs(eps - engine.REF_FREEATM) < 1e-6, eps
    print(f"  [ok] engine: MK free-atm shape peaks at 8 km (m={s['m']:.2f} vs "
          f"reconstructor prior), theta0_K={th_prof:.1f}\", J round-trips seeing")

    # theta0 solve: a smaller requested theta0 tilts turbulence aloft
    # (alpha > 0), raises the layer mismatch, and the profile reproduces it
    t_req = 0.6 * s["theta0_k_zenith"]
    s2 = engine.synthetic_field_snapshot(0.5, 0.3, theta0_k_zenith=t_req)
    th2 = engine.theta0_d0_from_profile(s2["cn2_bins"], 0.0,
                                        engine.LAMBDA_K_NM)[0]
    assert abs(th2 - t_req) < 1e-3 and s2["alpha"] > 0 and s2["m"] > 0.1
    print(f"  [ok] engine: theta0 solve (req {t_req:.1f}\" -> profile "
          f"{th2:.1f}\", alpha={s2['alpha']:+.2f}, m={s2['m']:.2f})")

    # line-of-sight projection at ZA=60 (airmass 2): seeing x 2^(3/5),
    # theta0 x 2^(-8/5)
    s3 = engine.synthetic_field_snapshot(0.5, 0.3, zenith_angle_deg=60.0)
    assert abs(s3["airmass"] - 2.0) < 1e-9
    assert abs(s3["eps_tot_los"] - 0.5 * 2 ** 0.6) < 1e-9
    assert abs(s3["theta0_los"]
               - s["theta0_k_zenith"] * 0.5 ** 1.6) < 1e-6
    # free-atm seeing is clamped to the total
    s4 = engine.synthetic_field_snapshot(0.4, 0.9)
    assert abs(s4["eps_fa_los"] - s4["eps_tot_los"]) < 1e-9
    print("  [ok] engine: LOS projection at ZA=60 exact; free-atm clamped")

    # theta0 decoupling reaches the LGS aniso terms: at the profile-derived
    # theta0 the factor is 1 (budget unchanged); a smaller theta0 scales the
    # angular-aniso + TT-aniso WFE by (theta0_prior/theta0)^(5/6)
    assert abs(s["aniso_scale"] - 1.0) < 1e-9
    assert abs(s2["aniso_scale"] - (1.0 / 0.6) ** (5.0 / 6.0)) < 1e-9
    kw = dict(tt_offset=19.3, lgs_offset=7.0)
    base = engine.lgs_strehl(0.5, 0.3, "K1", "single", **kw)
    worse = engine.lgs_strehl(0.5, 0.3, "K1", "single",
                              aniso_scale=s2["aniso_scale"], **kw)
    same = engine.lgs_strehl(0.5, 0.3, "K1", "single", aniso_scale=1.0, **kw)
    assert worse < base - 1e-4 and same == base
    t1 = engine.lgs_budget_terms(0.5, 0.3, "K1", "single", **kw)
    t2 = engine.lgs_budget_terms(0.5, 0.3, "K1", "single",
                                 aniso_scale=s2["aniso_scale"], **kw)
    assert abs(t2["ang"] / t1["ang"] - s2["aniso_scale"]) < 1e-9
    assert t2["tt"] > t1["tt"] and t2["fit"] == t1["fit"]
    print(f"  [ok] engine: aniso_scale x{s2['aniso_scale']:.2f} hits ang "
          f"({t1['ang']:.0f}->{t2['ang']:.0f} nm) + TT, Strehl "
          f"{base:.3f}->{worse:.3f}; default path untouched")

    # TT ceiling: a loop cannot do worse than no correction. At extreme
    # offsets the TT term saturates at the uncorrected image motion of a
    # seeing-limited spot (x s_tot); neither offset nor the aniso
    # re-weighting can push past it.
    # RANGE UPDATED 2026-08-07 (anchored to KAON 1318 Table 1, 48.7-88.7
    # mas one-axis over L0 = 10-100 m) and DEFAULT MOVED 2026-08-09: L0 is
    # now 50 m (74.4 mas @ ref), the Mauna Kea median both KAON 1318
    # (Fig. 5) and KAON 1303 (sect. 5.5) state; 25 m (60.5 mas) was the
    # two-day first cut. The bound applies exactly as before.
    cap1 = engine.OPEN_LOOP_TILT_ONEAXIS_MAS * engine.NM_PER_MAS   # s_tot=1
    assert 40 < engine.OPEN_LOOP_TILT_ONEAXIS_MAS < 95, \
        engine.OPEN_LOOP_TILT_ONEAXIS_MAS
    assert (engine.OPEN_LOOP_TILT_ONEAXIS_MAS
            < engine.OPEN_LOOP_TILT_ONEAXIS_MAS_KOLMOGOROV), \
        "the outer-scale ceiling must sit BELOW the old Kolmogorov one"
    # and the public re-export must not go stale when L0 moves (psf
    # rebinds a module global; see psf.set_outer_scale) -- exercise a
    # NON-default value and restore the default afterwards
    engine.set_outer_scale(25.0)
    cap25 = engine.tt_wfe_nm(1.0, 15.2, 900.0)
    assert abs(engine.OPEN_LOOP_TILT_ONEAXIS_MAS
               - cap25 / engine.NM_PER_MAS) < 1e-9, \
        "engine.OPEN_LOOP_TILT_ONEAXIS_MAS went stale vs engine.tt_wfe_nm"
    assert cap25 < cap1, "smaller L0 must lower the ceiling"
    engine.set_outer_scale(50.0)                     # restore the default
    far  = engine.tt_wfe_nm(1.0, 15.2, 500.0)
    far2 = engine.tt_wfe_nm(1.0, 15.2, 900.0)
    assert abs(far - cap1) < 1e-9 and abs(far2 - cap1) < 1e-9, (far, far2)
    assert engine.tt_wfe_nm(1.0, 15.2, 60.0, aniso_scale=50.0) <= cap1 + 1e-9
    # The nominal budget must still sit clear of the ceiling: the default
    # TT star is ~244 nm against a 931 nm ceiling (26%; it was 32% at the
    # 25 m first cut, 18% under the old infinite-outer-scale bound). This
    # is the number to watch if the TT rows are ever recalibrated.
    assert engine.tt_wfe_nm(1.0) < 0.40 * cap1, "nominal budget must be far below"
    assert abs(engine.tt_wfe_nm(2.0, 15.2, 900.0) - 2.0 * cap1) < 1e-9, \
        "ceiling must scale with s_tot"
    assert engine.ngs_tt_nm(1.0, 8.0, 500.0) <= cap1 + 1e-9, "NGS path capped too"
    print(f"  [ok] engine: TT saturates at the open-loop tilt "
          f"({engine.OPEN_LOOP_TILT_ONEAXIS_MAS:.0f} mas one-axis @ ref, "
          f"scaling with seeing); nominal budget untouched")


def _target_val(win):
    fig = win._fm_holder["canvas"].figure
    imgax = next(ax for ax in fig.axes if ax.images)
    Z = np.asarray(imgax.images[0].get_array())
    return float(Z[Z.shape[0] // 2, Z.shape[1] // 2])


def gui_tab():
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.resize(1550, 950); win.show(); app.processEvents()
    assert win.tabs.tabText(win.tabs.count() - 1) == "Prediction"

    win.mode_local.setChecked(True)
    win.dimm_edit.setText(f"{DATA}/20260525_dimm.dat")
    win.mass_edit.setText(f"{DATA}/20260525_mass.dat")
    win.masspro_edit.setText(f"{DATA}/20260525_masspro.dat")
    win.tel_k1.setChecked(True); win._validate(); win.on_run()
    pump(lambda: win.res is not None)

    # night-based field map value first
    win.fm_mode.setCurrentText("LTAO")
    win.fm_metric.setCurrentText("Strehl")
    win.plot_tabs.setCurrentIndex(1); settle()
    t_night = _target_val(win)

    # enabling the prediction jumps to the Field map tab and swaps the
    # snapshot: the reference scenario is NOT the night's median conditions
    win.plot_tabs.setCurrentIndex(0); app.processEvents()
    win.pred_enable.setChecked(True); settle()
    assert win.plot_tabs.currentIndex() == 1, "enable should show the field map"
    assert not win.fm_cond.isEnabled(), "Conditions combo must gate off"
    t_ref = _target_val(win)
    assert abs(t_ref - t_night) > 1e-4, "prediction should change the map"
    fig = win._fm_holder["canvas"].figure
    ttl = next(ax for ax in fig.axes if ax.images).get_title()
    assert "PREDICTED SCENARIO" in ttl, ttl
    print(f"  [ok] GUI: prediction swaps the snapshot "
          f"(night {t_night:.3f} -> reference {t_ref:.3f}); title flagged")

    # worse seeing -> worse prediction
    win.pred_dimm.setValue(1.2); win.pred_mass.setValue(0.7); settle()
    t_bad = _target_val(win)
    assert t_bad < t_ref - 0.02, f"bad seeing should cut Strehl ({t_bad:.3f})"
    print(f"  [ok] GUI: 1.2\"/0.7\" scenario drops target Strehl to {t_bad:.3f}")

    # WFE-slider overrides carry over into the predicted map
    r = win.wfe_rows["STATIC_CALIB"]; prev = win.res
    r["spin"].setValue(r["default"] + 100)
    pump(lambda: win.res is not prev, timeout=20); settle()
    t_wfe = _target_val(win)
    assert t_wfe < t_bad - 1e-3, "WFE override must carry into the prediction"
    print(f"  [ok] GUI: STATIC_CALIB +100 nm carries over ({t_bad:.3f} -> {t_wfe:.3f})")
    win._reset_all_wfe(); prev = win.res
    pump(lambda: win.res is not prev, timeout=20); settle()

    # presets: seeing pairs land, theta0 is profile-derived (FA aloft ->
    # small theta0; ground-layer -> large)
    th = {}
    for name, dimm, mass in win.PRED_PRESETS:
        for b in win.tabs.widget(win.tabs.count() - 1).widget() \
                    .findChildren(QtWidgets.QPushButton):
            if b.text().startswith(name):
                b.click(); settle(3)
                break
        assert abs(win.pred_dimm.value() - dimm) < 1e-6
        assert abs(win.pred_mass.value() - mass) < 1e-6
        th[name] = win.pred_theta0.value()
    assert th["Free-atm dominated"] < th["Reference"] < th["Ground-layer dominated"]
    assert "m=" in win.pred_readout.text()
    print(f"  [ok] GUI: presets (theta0_K FA {th['Free-atm dominated']:.1f}\" "
          f"< ref {th['Reference']:.1f}\" < GL "
          f"{th['Ground-layer dominated']:.1f}\"); readout reports m")

    # zenith angle is STICKY: set it, then switch preset -> ZA unchanged
    win.pred_za.setValue(40.0); settle(2)
    for b in win.tabs.widget(win.tabs.count() - 1).widget() \
                .findChildren(QtWidgets.QPushButton):
        if b.text().startswith("Reference"):
            b.click(); settle(2); break
    assert abs(win.pred_za.value() - 40.0) < 1e-9, \
        "zenith angle must survive a preset change"
    win.pred_za.setValue(0.0); settle(2)
    print("  [ok] GUI: zenith angle is sticky across preset changes")

    # Cn2 profile plot: 6 free-atm bins + a ground-layer point, and the
    # ground layer dominates the ground-layer-dominated preset
    for b in win.tabs.widget(win.tabs.count() - 1).widget() \
                .findChildren(QtWidgets.QPushButton):
        if b.text().startswith("Ground-layer"):
            b.click(); settle(2); break
    pax = win.pred_prof_fig.axes[0]
    labels = [ln.get_label() for ln in pax.get_lines()]
    assert any("free-atm" in s for s in labels) and \
           any("ground layer" in s for s in labels), labels
    fa = next(ln for ln in pax.get_lines() if "free-atm" in ln.get_label())
    gl = next(ln for ln in pax.get_lines() if "ground layer" in ln.get_label())
    assert len(fa.get_xdata()) == 6, "free-atm profile must have the 6 MASS bins"
    assert gl.get_xdata()[0] > fa.get_xdata().max(), \
        "ground layer must dominate the free-atm bins in a GL scenario"
    assert pax.get_xscale() == "log"
    print("  [ok] GUI: Cn2 profile plots 6 free-atm bins + dominant ground layer")

    # theta0 auto-track: while 'auto' is on the row is read-only and follows
    # the prior-shape profile at the current free-atm seeing
    assert win.pred_theta0_auto.isChecked() and not win.pred_theta0.isEnabled()
    win.pred_mass.setValue(0.30); settle(3)       # GL preset had 0.15
    assert abs(win.pred_theta0.value() - th["Reference"]) < 0.1, \
        "auto theta0 must re-derive when the seeing changes"
    print("  [ok] GUI: theta0 auto-tracks the profile as the seeing changes")

    # dragging theta0 down (turbulence aloft) must degrade the LGS map even
    # in single-beacon mode -- the off-axis TT star (19.3" default) and the
    # 7" K1 laser offset both feel the aniso re-weighting. Requires the
    # explicit override (auto off frees the row).
    win.fm_mode.setCurrentText("single-LGS"); settle()
    t_hi = _target_val(win)
    win.pred_theta0_auto.setChecked(False); settle(3)
    assert win.pred_theta0.isEnabled(), "unchecking auto must free theta0"
    win.pred_theta0.setValue(5.0); settle()
    t_lo = _target_val(win)
    assert t_lo < t_hi - 0.01, f"theta0 5\" must cut LGS Strehl ({t_lo:.3f})"
    assert "anisoplanatism ×" in win.pred_readout.text()
    print(f"  [ok] GUI: single-LGS + off-axis TT: theta0 override "
          f"{win.pred_theta0.value():g}\" drops target {t_hi:.3f}->{t_lo:.3f}")
    win.fm_mode.setCurrentText("LTAO"); settle()

    # disable -> back to the night's snapshot (reference preset is active,
    # WFE reset above, so the map must reproduce the original night value)
    win.pred_enable.setChecked(False); settle()
    assert win.fm_cond.isEnabled(), "Conditions combo must re-enable"
    assert abs(_target_val(win) - t_night) < 1e-9, "night map must be restored"
    print("  [ok] GUI: disable restores the night-based field map exactly")

    # config round-trip (incl. the manual theta0 override state)
    win.pred_enable.setChecked(True)
    win.pred_theta0_auto.setChecked(False)
    win.pred_dimm.setValue(0.9); win.pred_theta0.setValue(6.5)
    win.pred_za.setValue(45.0); settle(3)
    cfg = win._collect_config()
    win.pred_enable.setChecked(False)
    win.pred_theta0_auto.setChecked(True)
    win.pred_dimm.setValue(0.5); win.pred_za.setValue(0.0); settle(3)
    win._apply_config(cfg); settle()
    assert win.pred_enable.isChecked()
    assert abs(win.pred_dimm.value() - 0.9) < 1e-6
    assert abs(win.pred_za.value() - 45.0) < 1e-6
    assert (not win.pred_theta0_auto.isChecked()
            and abs(win.pred_theta0.value() - 6.5) < 1e-6
            and win.pred_theta0.isEnabled()), "theta0 override must survive"
    print("  [ok] GUI: prediction settings survive a config round-trip")
    win.grab().save(os.path.join(HERE, "gui_phase17.png"))


def main():
    engine_contract()
    gui_tab()
    print("  [ok] prediction tab: engine contract + GUI")


if __name__ == "__main__":
    main()
