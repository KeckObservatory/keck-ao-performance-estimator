#!/usr/bin/env python3
"""GFS winds in the GUI: the Data tab's "Fetch winds (GFS)" button
(GfsWindsWorker -> winds.night_winds) fills the wind spinboxes with the
night's real representative winds and the estimate re-runs; a FAILED fetch
warns and changes nothing (Eduardo's explicit requirement: bounded, no
hang, no silent damage). Plus the summary panel's tau0 row. Fully offline:
the worker's night_winds is monkeypatched. Run headless."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
from qtcompat import QtWidgets, QtCore
import keck_ao_estimator as engine
import keck_ao_estimator.gui as gui
import keck_ao_estimator.gui.workers as workers
DATA = os.path.join(HERE, "data")


def pump(cond, timeout=90):
    app = QtWidgets.QApplication.instance(); t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents(); QtCore.QThread.msleep(10)


def main():
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.resize(1500, 950); win.show(); app.processEvents()

    orig_nw = workers.night_winds
    seen = {}

    def fake_winds(ymd, cache_dir, fa_weights=None):
        seen.update(ymd=ymd, cache_dir=cache_dir, fa_weights=fa_weights)
        return dict(v_ground=3.4, v_free=22.9,
                    per_bin=[(0.5, 3.5), (1.0, 4.1), (2.0, 5.2),
                             (4.0, 6.9), (8.0, 40.9), (16.0, 6.7)],
                    bins_full=[(500.0, 3.5, 220.0), (1000.0, 4.1, 225.0),
                               (2000.0, 5.2, 230.0), (4000.0, 6.9, 227.0),
                               (8000.0, 40.9, 259.0), (16000.0, 6.7, 90.0)],
                    n_hours=13, hours="04-16 UT")

    # --- layout guards (the nighttime-status lesson, third time): the winds
    #     status and the FA advisory hold LONG strings -- each must sit on
    #     its own full-width form row with word-wrap and a zero width floor,
    #     never inline after another widget where it clips ------------------
    for lbl, name in ((win.winds_status, "winds_status"),
                      (win.fa_advisory, "fa_advisory")):
        assert lbl.wordWrap(), f"{name} must word-wrap, not clip"
        assert lbl.minimumWidth() == 0, f"{name} must stay width-shrinkable"
        assert lbl.parentWidget() is not win.winds_fetch_btn.parentWidget(), \
            f"{name} must sit on its own form row (inline after the button " \
            f"it only gets the leftover width and clips)"
    print("  [ok] winds status + FA advisory: own wrapped rows, zero width "
          "floors (no inline clipping)")

    # --- no night, local mode: refuses with the reason, no worker ------------
    win.mode_local.setChecked(True)
    win._fetch_gfs_winds(); app.processEvents()
    assert win._winds_worker is None
    assert "no night" in win.winds_status.text(), win.winds_status.text()
    print("  [ok] no loaded night + local mode: refused with the reason, "
          "no network worker at all")

    # --- load a real night, fetch (stubbed) -----------------------------------
    win.dimm_edit.setText(f"{DATA}/20260525_dimm.dat")
    win.mass_edit.setText(f"{DATA}/20260525_mass.dat")
    win.masspro_edit.setText(f"{DATA}/20260525_masspro.dat")
    win.tel_k2.setChecked(True); win._validate(); win.on_run()
    pump(lambda: win.res is not None); app.processEvents()
    assert win.wind_ground.value() == 8.0 and win.wind_free.value() == 25.0

    workers.night_winds = fake_winds
    try:
        prev = win.res
        win._fetch_gfs_winds()
        pump(lambda: win._winds_worker is None, timeout=15)
        pump(lambda: win.res is not prev, timeout=15)   # throttled recompute
        app.processEvents()
        # the ymd came from the PREPARED night (ut_stamp), not the date field
        assert seen["ymd"] == "20260525", seen
        assert seen["cache_dir"] == engine.DEF_CACHE_DIR
        # the free-atm collapse got the night's OWN median masspro weights
        assert seen["fa_weights"] is not None and len(seen["fa_weights"]) == 6
        assert win.wind_ground.value() == 3.4 and win.wind_free.value() == 22.9
        assert "GFS 20260525" in win.winds_status.text()
        assert "free-atm 22.9" in win.winds_status.text()
        assert "8→40.9" in win.winds_status.toolTip(), \
            "tooltip must carry the per-bin medians"
        assert "night-median" in win.winds_status.toolTip()
        print(f"  [ok] fetch fills the wind fields (8/25 -> 3.4/22.9), "
              f"ymd from the prepared night, night's own Cn2 weights, "
              f"estimate re-ran ({win.winds_status.text()})")

        # --- tau0 row: matches the engine formula for the shown period -------
        when, t_hst = win._when_time_from(win.stats_cond, win.stats_time)
        sel_t = engine.time_selection_mask(win.res.times, when, t_hst, win.prep)
        sel_p = engine.time_selection_mask(win.res.p_times, when, t_hst, win.prep)
        want = engine.tau0_seconds(
            engine.masked_mean(win.res.col_dimm, sel_t),
            engine.masked_mean(win.res.col_mass, sel_p), 3.4, 22.9)
        shown = float(win._stats_val["tau0"].text().split()[0])
        assert abs(shown - want * 1e3) < 0.06, (shown, want * 1e3)
        print(f"  [ok] summary tau0 row matches engine.tau0_seconds for the "
              f"period + CURRENT winds ({shown} ms)")

        # --- a failed fetch: warns, changes NOTHING ---------------------------
        def dead_winds(ymd, cache_dir, fa_weights=None):
            raise engine.WindsError("GFS winds fetch failed (timeout)")
        workers.night_winds = dead_winds
        win._fetch_gfs_winds()
        pump(lambda: win._winds_worker is None, timeout=15)
        app.processEvents()
        assert "winds unchanged" in win.winds_status.text(), \
            win.winds_status.text()
        assert win.wind_ground.value() == 3.4 and win.wind_free.value() == 22.9, \
            "a failed fetch must leave the winds exactly as they were"
        assert win.winds_fetch_btn.isEnabled(), "button must re-enable"
        print("  [ok] failed fetch: explicit warning, winds untouched, "
              "button re-enabled")

        # --- FA advisory (display-only): trailing stats + lead/lag line ------
        # point the target at the ZENITH at the reference time so the
        # lead/lag geometry is guaranteed valid regardless of which night's
        # data is loaded (the advisory itself would just omit the line for
        # a below-horizon target -- graceful, but not what we're testing)
        from datetime import timedelta
        import astropy.units as u
        from astropy.time import Time
        from keck_ao_estimator.constants import KECK_LAT_DEG, KECK_LON_DEG
        t_zen = win.res.p_times[-1]
        lst = Time(t_zen + timedelta(hours=10)).sidereal_time(
            "apparent", longitude=KECK_LON_DEG * u.deg)
        win.target_enable.setChecked(True)
        win.tname_edit.setText("ZenithTest")
        win.ra_edit.setText(lst.to_string(unit=u.hourangle, sep="hms",
                                          precision=1))
        win.dec_edit.setText(f"{KECK_LAT_DEG:.4f}")
        win.pmra_spin.setValue(0.0); win.pmdec_spin.setValue(0.0)
        app.processEvents()
        # after a run + winds, the advisory must show the trailing-window
        # summary and the per-layer lead/lag -- and it must NOT have touched
        # the timeline: same res object
        res_before = win.res
        win._update_fa_advisory()
        adv = win.fa_advisory.text()
        assert "FA" in adv and "last 40 min" in adv or "40 min" in adv, adv
        assert "+ = Keck first" in adv, \
            f"with winds fetched + target on, the lead/lag line must show: {adv}"
        assert "16 km" in adv
        assert win.res is res_before, "advisory must not trigger/alter a compute"
        # cross-check the trailing numbers against the engine helper directly
        t_ref = win.res.p_times[-1]
        st = engine.trailing_fa_stats(win.res.p_times, win.res.col_mass, t_ref)
        assert f"{st['med']:.2f}" in adv, (st, adv)
        print(f"  [ok] FA advisory: trailing stats + per-layer lead/lag, "
              f"display-only ({adv[:95]}…)")

        # no MASS-window samples near the reference -> says so, no crash
        from datetime import timedelta
        win._nighttime_last_pull = win.res.p_times[0] - timedelta(hours=3)
        win._update_fa_advisory()
        assert "no MASS samples" in win.fa_advisory.text()
        win._nighttime_last_pull = None
        win._update_fa_advisory()
        print("  [ok] FA advisory: out-of-window reference degrades gracefully")

        # --- FA geometry dialog (the live Figure-3 remake) --------------------
        res_before2 = win.res
        win._show_fa_geometry(); app.processEvents()
        assert win._fa_geo_dialog is not None
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        canvas = win._fa_geo_dialog.findChild(FigureCanvasQTAgg)
        assert canvas is not None
        axes = canvas.figure.axes
        assert len(axes) == 2, "plan + side view panels"
        # the summit-zoom inset is a CHILD axes of the plan view (inset_axes
        # children do not appear in figure.axes)
        insets = axes[0].child_axes
        assert len(insets) == 1, "plan view must carry the summit-zoom inset"
        assert insets[0].get_title().startswith("summit zoom"), \
            insets[0].get_title()
        # the plan view must carry the wind arrows/labels (bins_full present)
        texts = [t.get_text() for ax in axes for t in ax.texts]
        assert any("m/s" in t for t in texts), texts
        assert any("km out" in t for t in texts), texts
        # the catalog-model candidate box: 3 ranked stars with probabilities,
        # dated by the reference time (the stay-near-zenith model)
        cand_box = [t for t in texts if "monitor candidates" in t]
        assert cand_box and cand_box[0].count("P=") == 3, cand_box
        assert win.res is res_before2, "geometry dialog is display-only"
        win._fa_geo_dialog.close(); app.processEvents()
        assert win._fa_geo_dialog is None, "close must drop the reference"
        print("  [ok] FA geometry dialog: plan+side+zoom inset, wind vectors "
              "+ pierce labels, 3 catalog-model monitor candidates, "
              "display-only, close drops the reference")

        # --- the reference time follows the Period selector: 'specific
        #     time' must re-time the dialog (the stuck-at-end-of-night bug)
        prev_cond = win.stats_cond.currentText()
        win.stats_cond.setCurrentText("specific time")
        win.stats_time.setTime(QtCore.QTime(23, 30))
        win._show_fa_geometry(); app.processEvents()
        canvas2 = win._fa_geo_dialog.findChild(FigureCanvasQTAgg)
        sup = canvas2.figure._suptitle.get_text()
        assert "23:30" in sup, sup
        texts2 = [t.get_text() for a in canvas2.figure.axes for t in a.texts]
        assert any("monitor candidates" in t for t in texts2), \
            "candidates must recompute for the selected time"
        win._fa_geo_dialog.close(); app.processEvents()
        win.stats_cond.setCurrentText(prev_cond)
        print(f"  [ok] FA geometry reference time follows the Period "
              f"selector ({sup.split('·')[0].strip()})")

        # --- single-flight: a second click while one runs is a no-op ---------
        import threading
        release = threading.Event()

        def slow_winds(ymd, cache_dir, fa_weights=None):
            release.wait(5)
            return fake_winds(ymd, cache_dir, fa_weights)
        workers.night_winds = slow_winds
        win._fetch_gfs_winds()
        first = win._winds_worker
        assert first is not None
        win._fetch_gfs_winds()
        assert win._winds_worker is first, "second click must not stack workers"
        release.set()
        pump(lambda: win._winds_worker is None, timeout=15)
        app.processEvents()
        print("  [ok] one fetch in flight at a time")
    finally:
        workers.night_winds = orig_nw

    win.grab().save(os.path.join(HERE, "gui_phase27.png"))
    print("  [ok] screenshot saved")


if __name__ == "__main__":
    main()
