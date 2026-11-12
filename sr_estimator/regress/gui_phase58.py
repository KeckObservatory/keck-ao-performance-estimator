#!/usr/bin/env python3
"""All-sky sky coverage (2026-11-10): the whole-sky curve and map, the
STRAP-or-TRICK and fixed laser tip-tilt options, the truth-star requirement.

Engine contract (sky_map.py):
  * the bundled counts cover 821 equal-area sight lines at dec >= -40;
  * at the Strehl a 20-mas tip-tilt residual gives (K1 LTAO, reference
    seeing at ZA 30, TRICK-H science band), the whole-sky fractions match
    the independent per-star calculation of
    keck_ao_experiments/plgs_sky_coverage_20261102 (coverage_ir.py, same
    counts and relations, TT inverted in closed form): STRAP 0.1551,
    STRAP or TRICK with a truth star 0.4114, truth star 0.8879;
  * STRAP or TRICK is never below either sensor alone; the truth-star
    requirement never raises a sight line and leaves STRAP unchanged; the
    chance falls with Strehl; the laser is flat at its own Strehl up to
    the truth-star fraction (everywhere without the requirement);
  * budget: tt_fixed_nm replaces the TT star's rows and None is
    byte-identical; NGS ignores sensor, law and truth star.
GUI: the All-sky field is offered first while the default field stays
b = 60; the tilt law defaults to Hardy; TRICK on the All-sky field keeps
STRAP as a fallback (engine 'R+H'); PolyChroLGS TT or the Map view selects
the All-sky field, a bundled field drops them; the map draws on a Mollweide
axis and follows the Strehl level; the laser residual is live only for the
laser; the truth star only for the whole sky in LGS modes; the tab and
every title say the model is KAPE's engine, not the analytic error-budget
sheet; Hold works; config round-trip; the tab's control rows fit the
regress window. Fully offline, headless.
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
import warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
from qtcompat import QtWidgets
import keck_ao_estimator as engine
import keck_ao_estimator.gui as gui
from keck_ao_estimator.budget import lgs_budget_terms, lgs_strehl
from keck_ao_estimator.constants import REF_FREEATM, REF_TOTAL
from keck_ao_estimator.marechal import marechal_strehl, tilt_strehl
np = engine.np


def pump(ms=500):
    app = QtWidgets.QApplication.instance()
    end = time.time() + ms / 1000.0
    while time.time() < end:
        app.processEvents(); time.sleep(0.01)


def engine_contract():
    sky = engine.load_allsky()
    assert sky["n_cones"] == 821 and sky["dec"].min() >= -40.0
    assert np.all(sky["w"] >= 0) and np.all(np.isfinite(sky["H"]))

    a = engine.build_parser().parse_args([])
    a.telescope = "K1"; a.ngs_wfs = engine.DEF_NGS_WFS
    a.tt_sensor = "trick-h"; engine.resolve_tt_sensor(a)
    lam, lab = engine.resolve_wavelength(a)
    assert abs(lam - 2200.0) < 1e-9
    prep = SimpleNamespace(
        lam_nm=lam, lam_label=lab, show_target=False,
        _ltao_bw_fac=engine.ltao_bw_factor(a.ltao_bw_floor_frac))
    snap = engine.synthetic_field_snapshot(REF_TOTAL, REF_FREEATM,
                                           zenith_angle_deg=30.0)
    s20 = engine.laser_strehl(a, prep, snap, "ltao", laser_mas=20.0)
    run = {s: engine.allsky_coverage(a, prep, snap, "ltao", s)
           for s in ("R", "H", "R+H", "laser")}
    frac = {s: float(engine.sky_probability(r, s20 - 1e-9).mean())
            for s, r in run.items()}
    assert abs(frac["R"] - 0.1551) < 0.003, frac
    assert abs(frac["R+H"] - 0.4114) < 0.005, frac
    assert abs(frac["laser"] - 0.8879) < 0.001, frac
    assert abs(run["R"]["reach"] - 0.8879) < 0.001
    print(f"  [ok] engine: whole sky at the 20-mas Strehl ({s20:.3f}) "
          f"matches the per-star calculation: STRAP {frac['R']:.4f}, "
          f"STRAP or TRICK {frac['R+H']:.4f}, laser {frac['laser']:.4f}")

    P = {s: r["P"] for s, r in run.items()}
    assert np.all(P["R+H"] >= np.maximum(P["R"], P["H"]) - 1e-9)
    for p in P.values():
        assert np.all(np.diff(p, axis=1) <= 1e-12)
    free = {s: engine.allsky_coverage(a, prep, snap, "ltao", s, truth=False)
            for s in ("R", "H", "laser")}
    assert np.allclose(free["R"]["P"], P["R"], atol=1e-12)
    assert np.all(free["H"]["P"] >= P["H"] - 1e-12)
    assert free["H"]["P"].mean() > P["H"].mean()
    las = run["laser"]
    assert las["laser_sr"] == s20 and np.all(las["sr"] == s20)
    assert abs(las["frac"][-1] - las["reach"]) < 1e-3
    assert free["laser"]["reach"] == 1.0
    print("  [ok] engine: STRAP or TRICK >= either; truth star never raises a "
          "sight line and leaves STRAP alone; laser flat at its Strehl")

    # budget: the fixed tip-tilt term
    kw = dict(tt_mag=16.0, tt_offset=30.0)
    assert lgs_strehl(0.6, 0.35, "K1", "ltao", **kw) == lgs_strehl(
        0.6, 0.35, "K1", "ltao", tt_fixed_nm=None, **kw)
    t = lgs_budget_terms(0.6, 0.35, "K1", "ltao", **kw)
    t2 = lgs_budget_terms(0.6, 0.35, "K1", "ltao", tt_fixed_nm=123.0, **kw)
    assert t2["tt"] == 123.0 and all(t2[k] == t[k] for k in t if k != "tt")
    ho = np.sqrt(sum(v ** 2 for k, v in t.items() if k != "tt"))
    assert abs(lgs_strehl(0.6, 0.35, "K1", "ltao", 2200.0, tt_fixed_nm=123.0,
                          **kw) - marechal_strehl(ho, 2200.0)
               * tilt_strehl(123.0, 2200.0)) < 1e-12
    ls = [engine.laser_strehl(a, prep, snap, "ltao", laser_mas=x)
          for x in (10.0, 20.0, 40.0)]
    assert ls[0] > ls[1] > ls[2] > 0
    n1 = engine.allsky_coverage(a, prep, snap, "ngs", "H", "hardy")
    n2 = engine.allsky_coverage(a, prep, snap, "ngs")
    assert n1["sensor"] == "R" and not n1["truth"]
    assert np.array_equal(n1["P"], n2["P"])
    assert n2["sr_at"][0.05] < run["R"]["sr_at"][0.05]
    try:
        engine.allsky_coverage(a, prep, snap, "ltao", "J")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown sensor must raise")
    print("  [ok] engine: tt_fixed_nm replaces only the tip-tilt term "
          "(None byte-identical); NGS ignores sensor / law / truth star")


def gui_tab():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = gui.MainWindow(); win.resize(1550, 950); win.show(); pump()
    tabs = win.plot_tabs
    last = tabs.count() - 1
    assert win.sc_field.itemData(0) == engine.ALLSKY_FIELD
    assert win.sc_field.currentData() == engine.DEF_SKY_FIELD
    assert win.sc_view.currentText() == "Curve"
    assert not win.sc_level.isEnabled() and not win.sc_laser.isEnabled()
    assert not win.sc_truth.isEnabled()          # bundled field
    assert win.sc_law.currentText() == "Hardy"
    assert "not the analytic error-budget sheet" in win.sc_model_label.text()
    win.tel_k1.setChecked(True); pump(300)
    win.sc_mode.setCurrentText("LTAO")
    win.pred_enable.setChecked(True); pump(300)
    tabs.setCurrentIndex(last); pump(300)
    h = win._sc_holder["widget"]
    assert h.minimumSizeHint().width() <= h.width(), (
        h.minimumSizeHint().width(), h.width())

    # TRICK on the whole sky keeps STRAP as a fallback
    win.sc_field.setCurrentIndex(0); pump(300)
    win.sc_sensor.setCurrentText("TRICK (H)"); pump(400)
    assert win.sc_field.currentData() == engine.ALLSKY_FIELD
    label, res = win._sc_last
    assert res["sensor"] == "R+H" and res["truth"] and "All sky" in label
    assert res["tilt_law"] == "hardy"
    title = win._sc_holder["canvas"].figure.axes[0].get_title()
    assert "TRICK H ≤ 17 or STRAP R ≤ 19" in title and "truth star" in title
    assert "not the analytic error-budget sheet" in title
    direct = engine.allsky_coverage(
        win._skycov_inputs()[0], win._skycov_inputs()[1],
        win._pred_snapshot(), "ltao", "R+H", "hardy")
    assert np.array_equal(res["sr"], direct["sr"])
    assert win.sc_truth.isEnabled()
    win.sc_truth.setChecked(False); pump(300)
    assert not win._sc_last[1]["truth"]
    assert win._sc_last[1]["reach"] > res["reach"]
    win.sc_truth.setChecked(True); pump(300)
    win.sc_hold_btn.click(); pump(200)
    assert len(win._sc_held) == 1

    # the map, following the Strehl level
    win.sc_view.setCurrentText("Map"); pump(400)
    assert win.sc_level.isEnabled()
    ax = win._sc_holder["canvas"].figure.axes[0]
    assert ax.name == "mollweide" and "Strehl ≥ 0.30 on" in ax.get_title()
    assert "not the analytic error-budget sheet" in ax.get_title()
    pct = lambda t: int(t.split("Strehl ≥")[1].split(" on ")[1].split(" %")[0])
    at30 = pct(ax.get_title())
    win.sc_level.setValue(0.10); pump(400)
    ax = win._sc_holder["canvas"].figure.axes[0]
    assert "Strehl ≥ 0.10 on" in ax.get_title() and pct(ax.get_title()) > at30

    # laser: its residual control is live only for it
    win.sc_sensor.setCurrentText("PolyChroLGS TT"); pump(400)
    assert win.sc_laser.isEnabled()
    r = win._sc_last[1]
    assert r["sensor"] == "laser" and r["laser_mas"] == 20.0
    win.sc_laser.setValue(40.0); pump(400)
    assert win._sc_last[1]["laser_sr"] < r["laser_sr"]
    assert "40 mas" in win._sc_last[0]

    # a bundled field drops the map and the whole-sky sensor
    win.sc_field.setCurrentIndex(win.sc_field.findData("b60_l90")); pump(400)
    assert win.sc_view.currentText() == "Curve"
    assert win.sc_sensor.currentText() == "STRAP (R)"
    assert not win.sc_truth.isEnabled() and not win.sc_laser.isEnabled()
    assert win._sc_last[1]["field"] == "b60_l90"
    # the Map view selects the All-sky field again
    win.sc_view.setCurrentText("Map"); pump(400)
    assert win.sc_field.currentData() == engine.ALLSKY_FIELD
    # K2: STRAP-or-TRICK unavailable, like TRICK
    win.sc_tel.setCurrentText("K2"); pump(300)
    model = win.sc_sensor.model()
    for i in range(win.sc_sensor.count()):
        t = model.item(i).text()
        assert model.item(i).isEnabled() == ("TRICK" not in t), t
    win.sc_tel.setCurrentText("K1"); pump(300)
    # NGS: no sensor, no truth star
    win.sc_mode.setCurrentText("NGS"); pump(400)
    assert not win.sc_truth.isEnabled() and win._sc_last[1]["mode"] == "ngs"
    win.sc_mode.setCurrentText("LTAO"); pump(300)
    print("  [ok] GUI: All-sky field offered, default field unchanged; "
          "Hardy default; TRICK keeps STRAP as fallback on the whole sky; "
          "PolyChroLGS TT and the map select it, a bundled field drops them; "
          "map follows the level; laser residual live only for the laser; "
          "truth star whole-sky LGS only; engine (not sheet) named; Hold")

    # config round-trip
    win.sc_sensor.setCurrentText("PolyChroLGS TT")
    win.sc_laser.setValue(25.0); win.sc_level.setValue(0.45)
    win.sc_truth.setChecked(False); pump(300)
    cfg = win._collect_config()
    win.sc_view.setCurrentText("Curve")
    win.sc_field.setCurrentIndex(win.sc_field.findData("b20_l45"))
    win.sc_laser.setValue(20.0); win.sc_level.setValue(0.30)
    win.sc_truth.setChecked(True); pump(300)
    win._apply_config(cfg); pump(600)
    assert (win.sc_view.currentText(), win.sc_field.currentData(),
            win.sc_sensor.currentText()) == ("Map", engine.ALLSKY_FIELD,
                                             "PolyChroLGS TT")
    assert (win.sc_laser.value(), win.sc_level.value(),
            win.sc_truth.isChecked()) == (25.0, 0.45, False)
    print("  [ok] GUI: config round-trip; control rows fit the window")


def main():
    engine_contract()
    gui_tab()
    print("  [ok] all-sky sky coverage")


if __name__ == "__main__":
    main()
