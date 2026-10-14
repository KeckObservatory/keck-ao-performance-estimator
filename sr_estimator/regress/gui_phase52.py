#!/usr/bin/env python3
"""Sky coverage tab (2026-10-13): Strehl versus the fraction of sky that
reaches it, for a chosen telescope, AO mode, tip-tilt sensor and tilt-Strehl
law, under the conditions the estimator is using.

Engine contract (sky_coverage.py):
  * the (magnitude, offset) Strehl surface IS the field map's on-axis
    model: at a star's magnitude and offset it equals field_metric_at
    exactly, for NGS, single-LGS and LTAO;
  * tilt_strehl: Marechal exp(-x^2) and Hardy 1/(1+x^2) agree to first
    order and Hardy is the higher one; lgs_strehl's default is Marechal,
    byte-identical to before the option existed;
  * a curve is non-increasing, ends at the star-in-reach fraction, and its
    table is NaN beyond it; a denser star field covers more sky; Hardy
    never reads below Marechal; NGS ignores the tilt law;
  * the synthetic b = 60 field (KAON 1303 counts) has the density, cuts and
    count slope of its recipe, reaches 1 - exp(-rho pi r^2) of the sky, and
    is bright-star poor, so its curve starts below the real field's.
GUI: the tab is the last plot tab; with no night it shows the placeholder,
with the Prediction scenario it draws from the predicted conditions, and
with a night loaded (scenario off) from that night's actuals; the tab's
telescope follows the Data tab and can be set apart from it; TRICK is
K1-only; NGS greys the sensor and the law; K2 NGS offers the 57x57 and
29x29 WFS modes (following the NGS tab's, each with its reference fit);
a held curve identical to the current one takes no second legend entry;
the readout box is headed with the curve it reads; the patrol radius is
60" or 51" (KAON 1318's), a smaller one reaching less sky; the synthetic sky
field is
selectable and named in the title, and the KAON 1318 overlay says it was
computed on synthetic Besancon fields; WFE sliders move the curve;
Hold / Clear; config round-trip; the left dock gains no tab. Fully
offline, headless.
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
from keck_ao_estimator.budget import lgs_strehl
np = engine.np
DATA = os.path.join(HERE, "data")


def pump(ms=500):
    app = QtWidgets.QApplication.instance()
    end = time.time() + ms / 1000.0
    while time.time() < end:
        app.processEvents(); time.sleep(0.01)


def engine_contract():
    a = engine.build_parser().parse_args([])
    a.telescope = "K1"; a.ngs_wfs = engine.DEF_NGS_WFS
    engine.resolve_tt_sensor(a)
    lam, lab = engine.resolve_wavelength(a)
    prep = SimpleNamespace(
        lam_nm=lam, lam_label=lab, show_target=False,
        _ltao_bw_fac=engine.ltao_bw_factor(a.ltao_bw_floor_frac))
    snap = engine.synthetic_field_snapshot(0.50, 0.30, 20.0, lam)
    L = engine.resolve_lgs_offset(a)

    # the surface is the field map's on-axis model
    for mode in ("ngs", "single", "ltao"):
        mags, offs, Z = engine.strehl_surface(a, prep, snap, mode, "R")
        for m, r in ((9.0, 0.0), (13.0, 12.0), (16.5, 40.0)):
            want = engine.field_metric_at(
                a, prep, snap, mode, "strehl", (r, 0.0), (r, 0.0), (L, 0.0),
                (0.0, 0.0), ngs_bright_override=m, tt_mag_override=m)
            got = Z[list(mags).index(m), list(offs).index(r)]
            assert abs(got - want) < 1e-12, (mode, m, r, got, want)
    print("  [ok] engine: Strehl surface == field map on-axis value "
          "(NGS, single-LGS, LTAO)")

    # tilt laws
    x = np.array([1.0, 20.0, 150.0, 600.0])
    mar = engine.tilt_strehl(x, 2200.0)
    har = engine.tilt_strehl(x, 2200.0, "hardy")
    assert np.allclose(mar, engine.marechal_strehl(x, 2200.0))
    assert np.all(har >= mar) and abs(har[0] - mar[0]) < 1e-9
    assert abs(har[2] - 1.0 / (1.0 + (2 * np.pi * 150.0 / 2200.0) ** 2)) < 1e-12
    base = lgs_strehl(0.6, 0.35, "K1", "ltao", tt_mag=16.0, tt_offset=30.0)
    assert base == lgs_strehl(0.6, 0.35, "K1", "ltao", tt_mag=16.0,
                              tt_offset=30.0, tilt_law="marechal")
    assert lgs_strehl(0.6, 0.35, "K1", "ltao", tt_mag=16.0, tt_offset=30.0,
                      tilt_law="hardy") > base
    try:
        engine.tilt_strehl(1.0, 2200.0, "gaussian")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown tilt law must raise")
    print("  [ok] engine: Hardy tilt law >= Marechal, equal to first order; "
          "lgs_strehl default unchanged")

    # curves
    mar = engine.sky_coverage_curve(a, prep, snap, "ltao", "R")
    har = engine.sky_coverage_curve(a, prep, snap, "ltao", "R", "hardy")
    for r in (mar, har):
        assert np.all(np.diff(r["sr"]) <= 0) and np.all(r["sr"] >= 0)
        assert abs(r["frac"][-1] - r["reach"]) < 1e-12 and 0 < r["reach"] < 1
        assert all(np.isnan(v) == (f > r["reach"])
                   for f, v in r["sr_at"].items())
        assert all(0.0 <= v <= r["reach"] for v in r["frac_at"].values())
    assert mar["reach"] == har["reach"] and np.all(har["sr"] >= mar["sr"] - 1e-12)
    assert har["sr_at"][0.30] > mar["sr_at"][0.30] + 0.02
    dense = engine.sky_coverage_curve(a, prep, snap, "ltao", "R",
                                      field="b20_l45")
    pole = engine.sky_coverage_curve(a, prep, snap, "ltao", "R",
                                     field="b90_l00")
    assert dense["reach"] > mar["reach"] > pole["reach"]
    assert dense["sr_at"][0.30] > mar["sr_at"][0.30] > pole["sr_at"][0.30]
    trick = engine.sky_coverage_curve(a, prep, snap, "ltao", "H")
    assert trick["mag_limit"] == 17.0 and mar["mag_limit"] == 19.0
    n1 = engine.sky_coverage_curve(a, prep, snap, "ngs", "H", "hardy")
    n2 = engine.sky_coverage_curve(a, prep, snap, "ngs")
    assert n1["sensor"] == "R" and n1["tilt_law"] == "marechal"
    assert np.array_equal(n1["sr"], n2["sr"])
    assert n2["sr_at"][0.20] < 0.05 < mar["sr_at"][0.20]   # NGS covers little
    # synthetic field (KAON 1303 counts)
    # (statistics, not bit equality: numpy does not promise the same random
    # stream across versions, so the bundled copy is the reference)
    syn, gen = engine.load_sky_field("b60_synth"), engine.synthetic_sky_field()
    want = engine.sky_coverage.SYNTH_STARS_PER_ARCMIN2 * np.pi * 30.0 ** 2
    for cat in (syn, gen):
        for key, band, mlim in (("g", "G", 19.0), ("ir", "H", 17.0)):
            assert abs(len(cat[band]) - want) < 6 * np.sqrt(want)
            assert cat[band].min() >= 8.0 and cat[band].max() <= mlim
            assert np.hypot(cat[f"{key}_x"], cat[f"{key}_y"]).max() <= 1800.0
            # 0.30 dex/mag counts: half the stars in the last magnitude
            assert 0.42 < np.mean(cat[band] > mlim - 1.0) < 0.58
        assert np.array_equal(cat["K"], cat["H"])
    assert engine.is_synthetic_field("b60_synth")
    assert not engine.is_synthetic_field("b60_l90")
    real = engine.load_sky_field("b60_l90")
    for band in ("G", "H"):
        assert 1.5 < np.sum(real[band] <= 14) / np.sum(syn[band] <= 14) < 3.5
    s_r = engine.sky_coverage_curve(a, prep, snap, "ltao", "R",
                                    field="b60_synth")
    rho = len(syn["G"]) / (np.pi * 1800.0 ** 2)
    assert abs(s_r["reach"] - (1 - np.exp(-rho * np.pi * 60.0 ** 2))) < 0.02
    assert s_r["synthetic"] and not mar["synthetic"]
    assert s_r["sr_at"][0.10] < mar["sr_at"][0.10] - 0.03
    print(f"  [ok] engine: synthetic field matches its recipe; bright-star poor "
          f"(SR at 10 % sky {s_r['sr_at'][0.10]:.3f} vs "
          f"{mar['sr_at'][0.10]:.3f} on the real field)")
    print(f"  [ok] engine: curves monotone, end at reach "
          f"({100 * mar['reach']:.0f} % at b=60); LTAO+STRAP SR at 30 % sky "
          f"{mar['sr_at'][0.30]:.3f} Marechal / {har['sr_at'][0.30]:.3f} Hardy")


def gui_tab():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = gui.MainWindow(); win.resize(1550, 950); win.show(); pump()
    tabs = win.plot_tabs
    last = tabs.count() - 1
    assert tabs.tabText(last) == "Sky coverage"
    assert [tabs.tabText(i) for i in range(4)] == [
        "Timeline", "Field map", "Error terms", "Measured SR"]

    # no night, no scenario: placeholder
    tabs.setCurrentIndex(last); pump(200)
    assert win._sc_last is None

    # predicted conditions, no night loaded
    win.tel_k1.setChecked(True); pump(300)
    assert win.sc_tel.currentText() == "K1"       # follows the Data tab
    win.sc_mode.setCurrentText("LTAO")
    win.pred_enable.setChecked(True); pump(300)
    tabs.setCurrentIndex(last); pump(300)
    label, res = win._sc_last
    assert "predicted" in label and res["mode"] == "ltao"
    title = win._sc_holder["canvas"].figure.axes[0].get_title()
    assert "PREDICTED SCENARIO" in title and "K1 LTAO" in title
    direct = engine.sky_coverage_curve(
        win._fm_args(), win._fm_prep(), win._pred_snapshot(), "ltao", "R")
    assert np.array_equal(res["sr"], direct["sr"])
    # worse predicted seeing lowers the curve
    sr30 = res["sr_at"][0.30]
    win.pred_dimm.setValue(win.pred_dimm.value() + 0.3); pump(300)
    assert win._sc_last[1]["sr_at"][0.30] < sr30
    win.pred_dimm.setValue(win.pred_dimm.value() - 0.3); pump(300)
    assert abs(win._sc_last[1]["sr_at"][0.30] - sr30) < 1e-9
    print("  [ok] GUI: last tab; placeholder with nothing loaded; predicted "
          "conditions drive the curve")

    # options
    win.sc_law.setCurrentText("Hardy"); pump(100)
    assert win._sc_last[1]["tilt_law"] == "hardy"
    assert win._sc_last[1]["sr_at"][0.30] > sr30
    win.sc_hold_btn.click(); win.sc_hold_btn.click(); pump(100)
    assert len(win._sc_held) == 1                 # the same curve holds once
    ax = win._sc_holder["canvas"].figure.axes[0]
    # held == current: one line, one legend entry
    assert len(ax.get_legend().get_texts()) == 1 and len(ax.lines) <= 10
    win.sc_sensor.setCurrentText("TRICK (H)"); pump(100)
    assert win._sc_last[1]["sensor"] == "H"
    ax = win._sc_holder["canvas"].figure.axes[0]
    assert len(ax.get_legend().get_texts()) == 2
    box = [t.get_text() for t in ax.texts if "star in reach" in t.get_text()]
    assert len(box) == 1 and box[0].startswith(
        "K1 LTAO\nTRICK (H)\nHardy tilt law\nb = 60°\npredicted DIMM")
    win.sc_kaon.setChecked(True); pump(100)
    ax = win._sc_holder["canvas"].figure.axes[0]
    assert any("KAON 1318" in t.get_text() and "Besançon" in t.get_text()
               for t in ax.get_legend().get_texts())
    win.sc_field.setCurrentIndex(win.sc_field.findData("b60_synth"))
    pump(100)
    assert win._sc_last[1]["synthetic"]
    title = win._sc_holder["canvas"].figure.axes[0].get_title()
    assert "synthetic (KAON 1303 counts)" in title and "Gaia" not in title
    win.sc_field.setCurrentIndex(win.sc_field.findData("b60_l90"))
    pump(100)
    assert "Gaia + 2MASS" in (
        win._sc_holder["canvas"].figure.axes[0].get_title())
    # patrol radius: 60" (default) or KAON 1318's 51"
    r60 = win._sc_last[1]
    assert r60["patrol"] == 60.0 and "patrol" not in win._sc_last[0]
    win.sc_patrol.setCurrentIndex(win.sc_patrol.findData(51.0)); pump(100)
    r51 = win._sc_last[1]
    assert r51["patrol"] == 51.0 and r51["reach"] < r60["reach"]
    assert "51″ patrol" in win._sc_last[0]
    assert "within 51″" in win._sc_holder["canvas"].figure.axes[0].get_title()
    a51, p51, s51 = win._skycov_inputs()
    assert np.array_equal(r51["sr"], engine.sky_coverage_curve(
        a51, p51, s51, "ltao", "H", "hardy", patrol=51.0)["sr"])
    win.sc_patrol.setCurrentIndex(win.sc_patrol.findData(60.0)); pump(100)
    assert np.array_equal(win._sc_last[1]["sr"], r60["sr"])
    win.sc_kaon.setChecked(False); win.sc_clear_btn.click(); pump(100)
    assert win._sc_held == []
    # TRICK is K1-only; the tab's telescope can differ from the Data tab's
    win.sc_tel.setCurrentText("K2"); pump(100)
    assert win.tel_k1.isChecked() and win.sc_sensor.currentText() == "STRAP (R)"
    assert win._sc_last[0].startswith("K2 ")
    win.sc_mode.setCurrentText("NGS"); pump(100)
    assert not win.sc_sensor.isEnabled() and not win.sc_law.isEnabled()
    assert win._sc_last[1]["tilt_law"] == "marechal"
    assert "K2 NGS 57x57" in win._sc_last[0]
    # K2 NGS WFS mode: 57x57 / 29x29, each with its reference fit
    assert win.sc_ngs_wfs.isEnabled()
    sr57 = win._sc_last[1]["sr"].copy()
    win.sc_ngs_wfs.setCurrentText("29x29"); pump(100)
    assert "K2 NGS 29x29 · preliminary fit" in win._sc_last[0]
    sr29 = win._sc_last[1]["sr"]
    a29, p29, s29 = win._skycov_inputs()
    par = engine.ngs_fit_params("K2", "29x29")
    assert (a29.ngs_wfs, a29.ngs_s0, a29.ngs_m0) == (
        "29x29", par["S0"], par["m0"])
    assert np.array_equal(
        sr29, engine.sky_coverage_curve(a29, p29, s29, "ngs")["sr"])
    # 29x29: lower bright-star ceiling, but it holds to fainter stars
    assert sr29[0] < sr57[0] and np.sum(sr29 >= 0.2) > np.sum(sr57 >= 0.2)
    win.ngs_wfs.setCurrentText("57x57"); win.ngs_wfs.setCurrentText("29x29")
    assert win.sc_ngs_wfs.currentText() == "29x29"    # follows the NGS tab
    win.ngs_wfs.setCurrentText("57x57"); pump(100)
    assert np.array_equal(win._sc_last[1]["sr"], sr57)
    win.sc_tel.setCurrentText("K1"); pump(100)
    assert not win.sc_ngs_wfs.isEnabled() and "K1 NGS," in win._sc_last[0]
    win.sc_tel.setCurrentText("K2"); win.sc_mode.setCurrentText("LTAO")
    assert not win.sc_ngs_wfs.isEnabled()
    win.sc_mode.setCurrentText("NGS"); pump(100)
    win.sc_logx.setChecked(True); pump(100)
    assert win._sc_holder["canvas"].figure.axes[0].get_xscale() == "log"
    win.sc_logx.setChecked(False)
    win.sc_tel.setCurrentText("K1"); win.sc_mode.setCurrentText("LTAO")
    win.sc_law.setCurrentText("Maréchal"); pump(100)
    assert win.sc_sensor.isEnabled() and win.sc_law.isEnabled()
    print("  [ok] GUI: Hardy, TRICK, KAON 1318 overlay (Besancon noted), "
          "synthetic / real sky field, Hold/Clear without duplicate legend "
          "entries, headed readout box, 51\" / 60\" patrol, K2 from a K1 "
          "dock, K2 NGS 57x57 / "
          "29x29, NGS greys sensor + law, log axis")

    # WFE sliders move the curve (live, through the control debounce)
    before = win._sc_last[1]["sr_at"][0.10]
    win.wfe_rows["MARGIN"]["spin"].setValue(win.wfe_rows["MARGIN"]["spin"].value() + 80)
    pump(600)
    assert win._sc_last[1]["sr_at"][0.10] < before - 0.01
    win.wfe_rows["MARGIN"]["spin"].setValue(win.wfe_rows["MARGIN"]["spin"].value() - 80)
    pump(600)
    assert abs(win._sc_last[1]["sr_at"][0.10] - before) < 1e-9
    print("  [ok] GUI: WFE sliders reach the curve")

    # config round-trip
    win.sc_tel.setCurrentText("K2"); win.sc_mode.setCurrentText("single-LGS")
    win.sc_law.setCurrentText("Hardy"); win.sc_ngs_wfs.setCurrentText("29x29")
    win.sc_field.setCurrentIndex(win.sc_field.findData("b20_l45"))
    win.sc_patrol.setCurrentIndex(win.sc_patrol.findData(51.0))
    win.sc_kaon.setChecked(True); win.sc_logx.setChecked(True)
    cfg = win._collect_config()
    win.sc_patrol.setCurrentIndex(win.sc_patrol.findData(60.0))
    win.sc_tel.setCurrentText("K1"); win.sc_mode.setCurrentText("NGS")
    win.sc_field.setCurrentIndex(0); win.sc_ngs_wfs.setCurrentText("57x57")
    win.sc_kaon.setChecked(False); win.sc_logx.setChecked(False)
    win._apply_config(cfg); pump(600)
    assert (win.sc_tel.currentText(), win.sc_mode.currentText(),
            win.sc_law.currentText(), win.sc_field.currentData(),
            win.sc_ngs_wfs.currentText()) == (
                "K2", "single-LGS", "Hardy", "b20_l45", "29x29")
    assert win.sc_patrol.currentData() == 51.0
    win.sc_patrol.setCurrentIndex(win.sc_patrol.findData(60.0))
    win.sc_ngs_wfs.setCurrentText("57x57")
    assert win.sc_kaon.isChecked() and win.sc_logx.isChecked()
    win.sc_tel.setCurrentText("K1"); win.sc_mode.setCurrentText("LTAO")
    win.sc_law.setCurrentText("Maréchal")
    win.sc_field.setCurrentIndex(win.sc_field.findData("b60_l90"))
    win.sc_kaon.setChecked(False); win.sc_logx.setChecked(False)
    print("  [ok] GUI: config round-trip")

    # actuals: a loaded night with the scenario off
    win.pred_enable.setChecked(False); pump(200)
    win.mode_local.setChecked(True)
    win.dimm_edit.setText(os.path.join(DATA, "20260525_dimm.dat"))
    win.mass_edit.setText(os.path.join(DATA, "20260525_mass.dat"))
    win.masspro_edit.setText(os.path.join(DATA, "20260525_masspro.dat"))
    win._validate(); win.on_run()
    t0 = time.time()
    while (win.worker is not None or win.res is None) and time.time() - t0 < 60:
        pump(100)
    assert win.res is not None, "night did not load"
    tabs.setCurrentIndex(last); pump(300)
    label, res = win._sc_last
    title = win._sc_holder["canvas"].figure.axes[0].get_title()
    assert "predicted" not in label and "PREDICTED" not in title
    when, t_hst = win._fm_when_time()
    snap = engine.field_snapshot(win.args_cached, win.prep, win.res, when,
                                 t_hst)
    assert f"seeing {snap['eps_tot_los']:.2f}" in title
    direct = engine.sky_coverage_curve(win.args_cached, win.prep, snap,
                                       "ltao", "R")
    assert np.array_equal(res["sr"], direct["sr"])
    # the scenario takes over again when it is switched back on
    win.pred_enable.setChecked(True); pump(300)
    tabs.setCurrentIndex(last); pump(300)
    assert "predicted" in win._sc_last[0]
    win.pred_enable.setChecked(False); pump(300)
    tabs.setCurrentIndex(last); pump(300)
    assert np.array_equal(win._sc_last[1]["sr"], direct["sr"])
    print(f"  [ok] GUI: night actuals drive the curve (seeing "
          f"{snap['eps_tot_los']:.2f}\", SR at 30 % sky "
          f"{res['sr_at'][0.30]:.3f}); the scenario overrides them while on")

    # the tab lives on the right: the left dock gained nothing
    assert [win.tabs.tabText(i) for i in range(win.tabs.count())] == [
        "Data", "Target", "NGS", "LGS", "WFE sliders", "Prediction"]
    tabs.setCurrentIndex(last); pump(200)
    win.grab().save(os.path.join(HERE, "gui_phase52.png"))
    win.close()


def main():
    engine_contract()
    gui_tab()
    print("  [ok] sky coverage tab")


if __name__ == "__main__":
    main()
