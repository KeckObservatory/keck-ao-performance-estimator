#!/usr/bin/env python3
"""Field-map science targets: an explicit field centre (grid origin) with
user-dropped science-target markers plotted at offsets from it, each reading
its own predicted performance via engine.field_metric_at (the exact
single-point evaluator, identical to the map at that point). Plus the
right-click actions to place the laser / TT / NGS star at a clicked field
position, remove/clear targets, and config round-trip. Run headless."""
import os, sys, time, math
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


def engine_single_point():
    """field_metric_at at a point == field_map_grid's value at that same point
    (centre AND an interior grid node), for every mode/metric, to machine
    precision -- the refactor that shares _field_point must not perturb the
    grid, and a dropped target must read exactly what the map shows there."""
    from astropy.utils.iers import conf; conf.auto_max_age = None
    a = engine.build_parser().parse_args([
        "--dimm", f"{DATA}/20260525_dimm.dat", "--mass", f"{DATA}/20260525_mass.dat",
        "--masspro", f"{DATA}/20260525_masspro.dat", "--telescope", "K1",
        "--out", "/tmp/fp.png", "--force"])
    p = engine.prepare_night(a); r = engine.compute_timeline(a, p)
    snap = engine.field_snapshot(a, p, r, "night")
    lgs = 7.0
    laser = (+lgs / math.sqrt(2), -lgs / math.sqrt(2))
    tt = (0.0, a.tt_offset); ngs = (0.0, 0.0)
    for mode in ("single", "ltao", "ngs"):
        for metric in ("strehl", "fwhm", "fwhm_gaussfit"):
            ext, Z, meta = engine.field_map_grid(a, p, snap, mode, metric,
                                                 ngs, tt, laser, n_grid=41)
            c = engine.field_metric_at(a, p, snap, mode, metric, ngs, tt,
                                       laser, (0.0, 0.0))
            assert abs(c - meta["target"]) < 1e-12, (mode, metric, c, meta["target"])
            xs = np.linspace(ext[0], ext[1], 41); ys = np.linspace(ext[2], ext[3], 41)
            v = engine.field_metric_at(a, p, snap, mode, metric, ngs, tt,
                                       laser, (xs[30], ys[12]))
            assert abs(v - Z[12, 30]) < 1e-12, (mode, metric, v, Z[12, 30])
    print("  [ok] engine: field_metric_at == grid centre AND grid node "
          "(all modes/metrics, byte-exact)")


def gui_targets():
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.resize(1500, 950); win.show(); app.processEvents()
    win.mode_local.setChecked(True)
    win.dimm_edit.setText(f"{DATA}/20260525_dimm.dat")
    win.mass_edit.setText(f"{DATA}/20260525_mass.dat")
    win.masspro_edit.setText(f"{DATA}/20260525_masspro.dat")
    win.tel_k1.setChecked(True); win._validate(); win.on_run()
    pump(lambda: win.res is not None)

    def idle():
        return (not win._fm_debounce.isActive()
                and not win._fm_settle.isActive())
    win.plot_tabs.setCurrentIndex(1); app.processEvents(); pump(idle)

    # the NIRC2 use case: two science targets 8" apart, read each one's Strehl
    win._fm_add_target(-4.0, 0.0)          # 4" East of the field centre
    win._fm_add_target(+4.0, 0.0)          # 4" West of the field centre
    pump(idle); app.processEvents()
    assert len(win._fm_markers) == 2, win._fm_markers
    sep = math.hypot(win._fm_markers[0]["x"] - win._fm_markers[1]["x"],
                     win._fm_markers[0]["y"] - win._fm_markers[1]["y"])
    assert abs(sep - 8.0) < 1e-9, sep

    # each dropped target's value == the exact single-point engine evaluation
    snap = engine.field_snapshot(win.args_cached, win.prep, win.res,
                                 *win._fm_when_time())
    mode = {"NGS": "ngs", "single-LGS": "single",
            "LTAO": "ltao"}[win.fm_mode.currentText()]
    fc = win._sky_field_center()
    nx = win.ngs_offset.offset_xy(fc); tx = win.tt_offset.offset_xy(fc)
    lx = win._laser_xy()
    for m in win._fm_markers:
        with engine.budget_overrides(**win.last_offsets):
            want = engine.field_metric_at(win.args_cached, win.prep, snap, mode,
                                          "strehl", nx[:2], tx[:2], lx,
                                          (m["x"], m["y"]))
        assert m["val"] is not None and abs(m["val"] - want) < 1e-9, (m, want)
    print(f"  [ok] two targets 8\" apart, per-target SR = exact eval "
          f"(T1={win._fm_markers[0]['val']:.3f}, T2={win._fm_markers[1]['val']:.3f})")

    # the Targets list widget mirrors the markers; markers drawn as green squares
    assert win.fm_target_list.count() == 2, win.fm_target_list.count()
    ax = next(a for a in win._fm_holder["canvas"].figure.axes if a.images)
    greens = [ln for ln in ax.get_lines() if ln.get_marker() == "s"
              and ln.get_markerfacecolor() == gui.FM_C_MARKER]
    assert len(greens) >= 2, f"expected 2 target markers, got {len(greens)}"
    # the origin marker is the FIELD CENTRE (blue), distinct from the targets
    ctr = next(ln for ln in ax.get_lines()
               if ln.get_label().startswith("field centre"))
    assert ctr.get_markerfacecolor() == gui.FM_C_TARGET
    print("  [ok] targets listed + drawn (green); origin is the blue field centre")

    # right-click 'Put laser here' lands the laser at the click (within the
    # 0.1\"/1° control resolution) and enables the LGS offset
    win._fm_put_laser(3.0, -5.0); pump(idle)
    lxy = win._laser_xy()
    assert math.hypot(lxy[0] - 3.0, lxy[1] + 5.0) < 0.25, lxy
    assert win.lgs_offset_enable.isChecked()
    # 'Put TT star here' -> ΔRA/ΔDec spinboxes (0.1\") land it exactly
    win._fm_put_star(win.tt_offset, -6.0, 2.0); pump(idle)
    txy = win.tt_offset.offset_xy(fc)
    assert abs(txy[0] + 6.0) < 1e-6 and abs(txy[1] - 2.0) < 1e-6, txy
    win._fm_put_star(win.ngs_offset, 1.5, -2.5); pump(idle)
    nxy = win.ngs_offset.offset_xy(fc)
    assert abs(nxy[0] - 1.5) < 1e-6 and abs(nxy[1] + 2.5) < 1e-6, nxy
    print("  [ok] right-click places laser / TT star / NGS star at the click")

    # remove-nearest keeps the far one; clear empties the list
    win._fm_remove_near(-4.0, 0.1); pump(idle)
    assert len(win._fm_markers) == 1 and win._fm_markers[0]["name"] == "T2"
    win._fm_clear_targets(); pump(idle)
    assert len(win._fm_markers) == 0 and win.fm_target_list.count() == 0
    print("  [ok] remove-nearest + clear-all")

    # config round-trips the dropped targets
    win._fm_add_target(2.5, -3.5); win._fm_add_target(-1.0, 7.0)
    cfg = win._collect_config()
    assert len(cfg["fm_markers"]) == 2, cfg["fm_markers"]
    win._fm_clear_targets()
    win._loading = True; win._apply_config(cfg); win._loading = False
    assert len(win._fm_markers) == 2 and abs(win._fm_markers[0]["x"] - 2.5) < 1e-9
    print("  [ok] config round-trip preserves dropped targets")


def main():
    engine_single_point()
    gui_targets()
    print("  [ok] field-map science targets + explicit field centre")


if __name__ == "__main__":
    main()
