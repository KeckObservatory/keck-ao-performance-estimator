#!/usr/bin/env python3
"""Data-tab summary-stats panel (SummaryStatsMixin): mean NGS(bright)/single-
LGS/LTAO Strehl for K1 AND K2, mean DIMM/MASS seeing, mean r0, mean theta0,
over the same "observing window / whole night / specific time / time of last
pull" period selector the field map already offers. Cross-checked against
DIRECT engine calls (night_stats.time_selection_mask/masked_mean,
compute_timeline for the non-selected telescope) so the panel can never
silently drift from what those functions actually compute. Run headless."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
from qtcompat import QtWidgets, QtCore
import keck_ao_estimator as engine
import keck_ao_estimator.gui as gui
DATA = os.path.join(HERE, "data")


def pump(cond, timeout=90):
    app = QtWidgets.QApplication.instance()
    loop = QtCore.QEventLoop()
    t0 = time.time()

    done = False

    def check():
        # A stale re-arm MUST be inert. The last singleShot(10, check) is
        # armed BEFORE cond() goes true, so it survives loop.quit() and
        # fires after pump() has returned -- by which time `loop` may be
        # garbage-collected, and calling quit() on a deleted C++
        # QEventLoop segfaults. That is the SIGSEGV (-11) that failed
        # gui_phase24 on CI 2026-07-28: intermittent, load-dependent, and
        # not reproducible on an idle box.
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
    done = True          # neutralise any callback still in flight


def main():
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.resize(1500, 950); win.show(); app.processEvents()

    # --- before any run: placeholders, never a crash ------------------------
    assert win.res is None
    win._refresh_summary_stats()
    assert all(lbl.text() == "—" for lbl in win._stats_val.values())
    print("  [ok] before any run: every stat shows the placeholder, no crash")

    # --- period combo matches the field map's exactly -----------------------
    assert ([win.stats_cond.itemText(i) for i in range(win.stats_cond.count())]
            == [win.fm_cond.itemText(i) for i in range(win.fm_cond.count())])
    print("  [ok] Period combo offers the same 4 options as the field map's")

    win.mode_local.setChecked(True)
    win.dimm_edit.setText(f"{DATA}/20260525_dimm.dat")
    win.mass_edit.setText(f"{DATA}/20260525_mass.dat")
    win.masspro_edit.setText(f"{DATA}/20260525_masspro.dat")
    win.tel_k2.setChecked(True)
    win.on_run()
    pump(lambda: win.res is not None)
    app.processEvents()

    # --- values cross-checked directly against the engine -------------------
    when, t_hst = win._when_time_from(win.stats_cond, win.stats_time)
    # default combo text is "observing window" -> when="window" literally;
    # show_target is off, so time_selection_mask itself falls back to the
    # whole night (matching field_cn2_profile's own fallback convention)
    assert when == "window" and t_hst is None
    sel_t = engine.time_selection_mask(win.res.times, when, t_hst, win.prep)
    sel_p = engine.time_selection_mask(win.res.p_times, when, t_hst, win.prep)
    want_ngs_k2 = engine.masked_mean(win.res.ngs_bright, sel_t)
    want_lgs_k2 = engine.masked_mean(win.res.sr_single, sel_p)
    want_dimm = engine.masked_mean(win.res.col_dimm, sel_t)
    want_mass = engine.masked_mean(win.res.col_mass, sel_p)
    want_r0 = engine.masked_mean(win.res.col_r0_cm, sel_t)
    want_theta0 = engine.masked_mean(win.res.col_theta0, sel_p)
    # tolerances match each field's DISPLAY precision (3.dp Strehl, 2.dp
    # arcsec, 1.dp cm) -- these compare the rendered label text, not the
    # raw float, so a rounding-sized gap is expected, not a bug
    assert abs(float(win._stats_val["ngs_k2"].text()) - want_ngs_k2) < 6e-4
    assert abs(float(win._stats_val["lgs_k2"].text()) - want_lgs_k2) < 6e-4
    assert abs(float(win._stats_val["dimm"].text().rstrip('"')) - want_dimm) < 6e-3
    assert abs(float(win._stats_val["mass"].text().rstrip('"')) - want_mass) < 6e-3
    assert abs(float(win._stats_val["r0"].text().split()[0]) - want_r0) < 6e-2
    assert abs(float(win._stats_val["theta0"].text().rstrip('"')) - want_theta0) < 6e-3
    print(f"  [ok] K2-current-telescope stats match direct engine masked_mean "
          f"calls exactly (NGS {want_ngs_k2:.3f}, LGS {want_lgs_k2:.3f}, "
          f"DIMM {want_dimm:.2f}\", MASS {want_mass:.2f}\", r0 {want_r0:.1f} cm, "
          f"theta0 {want_theta0:.2f}\")")

    # --- the OTHER telescope (K1) comes from a genuine second compute_timeline
    # -- with K1's OWN Gompertz-fit defaults, NOT the fit values collect_args
    # carries (those belong to the currently-selected telescope, K2): reusing
    # them was the 2026-07-22 bug where both NGS columns shifted with the
    # Telescope radio while LGS/LTAO (fit-independent) stayed put
    other_args = win.args_cached
    import copy
    k1_args = copy.copy(other_args); k1_args.telescope = "K1"
    _p = engine.NGS_PARAMS["K1"]
    k1_args.ngs_s0, k1_args.ngs_a = _p["S0"], _p["A"]
    k1_args.ngs_m0, k1_args.ngs_w = _p["m0"], _p["w"]
    with engine.budget_overrides(**(win.last_offsets or {})):
        res_k1 = engine.compute_timeline(k1_args, win.prep)
    want_ngs_k1 = engine.masked_mean(res_k1.ngs_bright, sel_t)
    want_lgs_k1 = engine.masked_mean(res_k1.sr_single, sel_p)
    want_ltao_k1 = engine.masked_mean(res_k1.sr_ltao, sel_p)
    assert abs(float(win._stats_val["ngs_k1"].text()) - want_ngs_k1) < 6e-4
    assert abs(float(win._stats_val["lgs_k1"].text()) - want_lgs_k1) < 6e-4
    assert abs(float(win._stats_val["ltao_k1"].text()) - want_ltao_k1) < 6e-4
    # K1 and K2 NGS must NOT match by coincidence (different Gompertz fits) --
    # a vacuous-test guard, same idea as gui_phase20's "must actually differ"
    assert abs(want_ngs_k1 - want_ngs_k2) > 0.01, \
        "this check is vacuous unless K1/K2 genuinely differ"
    print(f"  [ok] the OTHER telescope's (K1) numbers match a direct SECOND "
          f"compute_timeline call with K1's OWN fit (NGS {want_ngs_k1:.3f} vs "
          f"K2's {want_ngs_k2:.3f}, LGS {want_lgs_k1:.3f}, LTAO {want_ltao_k1:.3f})")

    # LTAO has no K2 column -- permanently "—", never overwritten
    assert None not in win._stats_val   # the LTAO/K2 cell was never registered
    print("  [ok] LTAO has no K2 entry (K1-only hardware)")

    # --- telescope-flip invariance: the SAME night's per-telescope stats must
    #     NOT depend on which telescope the Data tab has selected (the exact
    #     2026-07-22 symptom: NGS K1 0.463->0.599 / K2 0.513->0.649 on flip)
    before = {k: win._stats_val[k].text()
             for k in ("ngs_k1", "ngs_k2", "lgs_k1", "lgs_k2", "ltao_k1")}
    prev = win.res
    win.tel_k1.setChecked(True)
    pump(lambda: win.res is not prev, timeout=30)
    app.processEvents()
    after = {k: win._stats_val[k].text()
            for k in ("ngs_k1", "ngs_k2", "lgs_k1", "lgs_k2", "ltao_k1")}
    assert before == after, \
        f"per-telescope stats must not move with the Telescope radio:\n" \
        f"  K2 selected: {before}\n  K1 selected: {after}"
    prev = win.res
    win.tel_k2.setChecked(True)
    pump(lambda: win.res is not prev, timeout=30)
    app.processEvents()
    print("  [ok] flipping the Telescope radio leaves every per-telescope "
          "stat unchanged (each column always uses its own telescope's fit)")

    # --- band annotations (2026-07-23): every row states its band ------------
    # SR/theta0 at the SCIENCE band (live: tracks the band combo), the
    # DIMM/MASS/r0/tau0 monitor quantities at 500 nm
    assert win._stats_caption["ngs"].text() == "SR NGS (bright, K):"
    assert win._stats_caption["theta0"].text() == "theta0 (K):"
    print("  [ok] captions carry the bands (science K for SR/theta0; the "
          "500 nm rows are static text)")

    # --- geometry: common rows sit ON the K1/K2 midline ----------------------
    # everything in the two telescope columns is centred WITHIN its column,
    # which makes the common rows' col-1..2 span-centre equal the midpoint
    # of the two column centres by construction (two red-pen rounds taught
    # that left-aligned values park the span centre ~half a column off).
    # Self-relative pixels, so font-independent.
    def _xc(lbl):
        return lbl.mapToGlobal(lbl.rect().center()).x()
    mid = (_xc(win._stats_val["ngs_k1"]) + _xc(win._stats_val["ngs_k2"])) / 2
    for key in ("dimm", "mass", "r0", "theta0", "tau0"):
        off = _xc(win._stats_val[key]) - mid
        assert abs(off) <= 3, \
            f"common row '{key}' sits {off:+.0f}px off the K1/K2 midline"
    print("  [ok] common rows centred on the K1/K2 value midline (<=3px)")

    # --- switching Period recomputes -----------------------------------------
    prev_ngs = win._stats_val["ngs_k2"].text()
    win.stats_cond.setCurrentText("specific time")
    app.processEvents()
    assert win.stats_time.isEnabled()
    win.stats_time.setTime(QtCore.QTime(23, 30))
    app.processEvents()
    when_t, t_hst_t = win._when_time_from(win.stats_cond, win.stats_time)
    assert when_t == "time" and t_hst_t is not None
    sel_t2 = engine.time_selection_mask(win.res.times, when_t, t_hst_t, win.prep)
    assert sel_t2.sum() == 1, "'specific time' must select exactly one sample"
    want_ngs_t = engine.masked_mean(win.res.ngs_bright, sel_t2)
    assert abs(float(win._stats_val["ngs_k2"].text()) - want_ngs_t) < 6e-4
    assert win._stats_val["ngs_k2"].text() != prev_ngs, \
        "this check is vacuous unless the period switch actually changed it"
    print(f"  [ok] 'specific time' (23:30) selects exactly one sample and "
          f"the panel updates to match ({want_ngs_t:.3f}, was {prev_ngs})")
    win.stats_cond.setCurrentText("whole night")
    app.processEvents()

    # --- TRICK (K1-only) makes the K2 columns n/a, STRAP restores them -------
    win.tel_k1.setChecked(True)
    pump(lambda: True, timeout=1); app.processEvents()
    prev = win.res
    win.tt_sensor.setCurrentText("TRICK (K)")
    pump(lambda: win.res is not prev, timeout=15)
    app.processEvents()
    assert win._stats_val["ngs_k2"].text() == "—"
    assert win._stats_val["lgs_k2"].text() == "—"
    assert win._stats_val["ngs_k1"].text() != "—", \
        "the CURRENT (K1) telescope's own numbers must still show"
    print("  [ok] TRICK selected: the OTHER telescope's (K2) columns show "
          "n/a, the current (K1) telescope's own numbers still show")

    prev = win.res
    win.tt_sensor.setCurrentText("STRAP (R)")
    pump(lambda: win.res is not prev, timeout=15)
    app.processEvents()
    assert win._stats_val["ngs_k2"].text() != "—"
    assert win._stats_val["lgs_k2"].text() != "—"
    print("  [ok] switching back to STRAP repopulates the K2 columns")

    # --- config round-trip ----------------------------------------------------
    win.stats_cond.setCurrentText("specific time")
    win.stats_time.setTime(QtCore.QTime(22, 15))
    app.processEvents()
    cfg = win._collect_config()
    assert cfg["stats_cond"] == "specific time" and cfg["stats_time"] == "22:15"

    win2 = gui.MainWindow()
    win2._apply_config(cfg)
    assert win2.stats_cond.currentText() == "specific time"
    assert win2.stats_time.time().toString("HH:mm") == "22:15"
    assert win2.stats_time.isEnabled()
    win2.close()
    print("  [ok] Period selection (combo + time) round-trips through a config")

    win.grab().save(os.path.join(HERE, "gui_phase25.png"))
    print("  [ok] screenshot saved")


if __name__ == "__main__":
    main()
