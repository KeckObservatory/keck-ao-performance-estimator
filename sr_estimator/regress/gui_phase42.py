#!/usr/bin/env python3
"""Prediction tab: turbulence by LAYER on the reconstructor's altitude grid
(2026-09-15). The Layers sub-tab carries 7 turbulence FRACTIONS -- the
reconstructor's own units (KAON 1542 sect. 3.4 (2): 0/0.5/1/2/4/8/16 km,
[0.4557, 0.1295, 0.0442, 0.0506, 0.1167, 0.0926, 0.1107]) -- summing to
exactly 1 at all times: a slider/spin edit rescales the other rows
proportionally in either direction; an "Exact" text row + Apply sets all
seven verbatim (normalized only if they do not sum to 1). The Scenario
page's total (DIMM) seeing sets the scale (its row is mirrored on the
Layers page); its free-atm (MASS) seeing follows from the aloft share.
"Reset layers to reconstructor prior" loads the table verbatim (and undoes
edits).

Engine contract (layers.py + synthetic_field_snapshot(cn2_layers=...)):
  * seeing <-> integrated Cn2 round-trips; layer strengths add in J, so
    the seeing of two layers is the 5/3-power sum, not the plain sum;
  * fractions_to_layers at eps_tot reproduces eps_tot exactly (a sum
    below 1 is normalized), the prior gives eps_fa = eps_tot * (aloft
    share)^(3/5) and layer mismatch m = 0;
  * a snapshot built from layers uses the aloft 6 as its Cn2 profile
    (theta0 = that profile's own), reports alpha = NaN, and the 2-number
    path is byte-identical to before (no cn2_layers key set).
GUI: Layers is a Prediction SUB-tab (house rule: the dock never scrolls);
the button enables layer mode, MASS becomes derived and the presets gate
off while DIMM stays live; raising the 8 km fraction raises free-atm and
leaves the total alone, editing DIMM in layer mode rescales both; the sum
stays 1 both ways; exact entry; the Layers page repeats the plot live; the
field map (no night loaded) responds; config round-trip; Reset restores
the prior. Fully offline, headless.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
from qtcompat import QtWidgets, QtCore
import keck_ao_estimator as engine
import keck_ao_estimator.gui as gui
np = engine.np


def settle(n=6):
    app = QtWidgets.QApplication.instance()
    for _ in range(n):
        app.processEvents(); QtCore.QThread.msleep(25)


def engine_contract():
    # seeing <-> J inversion, and the 5/3-power combination of layers
    for eps in (0.2, 0.5, 1.37):
        assert abs(engine.integrated_cn2_to_seeing(
            engine.seeing_to_integrated_cn2(eps)) - eps) < 1e-9
    assert engine.integrated_cn2_to_seeing(0.0) == 0.0
    assert engine.layer_from_seeing(0.0) == 0.0
    J = np.zeros(7); J[0] = engine.layer_from_seeing(0.4)
    J[5] = engine.layer_from_seeing(0.3)
    ls = engine.layers_seeing(J)
    want = (0.4 ** (5 / 3) + 0.3 ** (5 / 3)) ** 0.6
    assert abs(ls["eps_tot"] - want) < 1e-9 and ls["eps_tot"] < 0.7
    assert abs(ls["eps_fa"] - 0.3) < 1e-9 and abs(ls["eps_ground"] - 0.4) < 1e-9
    print(f"  [ok] engine: seeing<->J round-trips; 0.4\"+0.3\" layers = "
          f"{ls['eps_tot']:.3f}\" (5/3-power sum, not 0.7\")")

    # the reconstructor prior scaled to a total seeing
    assert engine.RECON_HEIGHTS_M[0] == 0.0 and engine.N_RECON_LAYERS == 7
    assert np.allclose(engine.RECON_HEIGHTS_M[1:], engine.MASS_HEIGHTS_M)
    for eps_tot in (0.5, 0.8):
        Jp = engine.recon_prior_layers(eps_tot)
        lp = engine.layers_seeing(Jp)
        share = engine.RECON_PRIOR_FRAC[1:].sum()
        assert abs(lp["eps_tot"] - eps_tot) < 1e-9
        assert abs(lp["eps_fa"] - eps_tot * share ** 0.6) < 1e-9
        assert np.allclose(lp["frac"], engine.RECON_PRIOR_FRAC)
        assert engine.layer_mismatch(lp["cn2_bins"]) < 1e-12
    print(f"  [ok] engine: prior @0.5\" -> free-atm {lp['eps_fa'] / 0.8 * 0.5:.3f}\" "
          f"(aloft share {share:.4f}), m = 0 exactly")

    # fractions: the reconstructor's units. Normalized when the sum is < 1,
    # the total seeing is always the one asked for; all-zero -> empty
    f = np.array([0.5, 0.1, 0.05, 0.05, 0.1, 0.1, 0.1])
    Jf = engine.fractions_to_layers(f, 0.7)
    assert abs(engine.layers_seeing(Jf)["eps_tot"] - 0.7) < 1e-9
    assert np.allclose(engine.layers_seeing(Jf)["frac"], f)
    Jh = engine.fractions_to_layers(0.5 * f, 0.7)         # sums to 0.5
    assert np.allclose(Jh, Jf), "a sum below 1 must be used as proportions"
    assert not engine.fractions_to_layers(np.zeros(7), 0.7).any()
    assert np.allclose(engine.fractions_to_layers(engine.RECON_PRIOR_FRAC, 0.5),
                       engine.recon_prior_layers(0.5))
    print("  [ok] engine: fractions_to_layers normalizes the sum, keeps eps_tot")

    # snapshot from layers: aloft 6 ARE the profile, theta0 is the
    # profile's own, alpha NaN, m = 0 at the prior; two-number path untouched
    Jp = engine.recon_prior_layers(0.5)
    s = engine.synthetic_field_snapshot(9.9, 9.9, cn2_layers=Jp)
    assert abs(s["eps_tot_zenith"] - 0.5) < 1e-9, "layers must override the pair"
    assert np.allclose(s["cn2_bins"], Jp[1:]) and s["m"] < 1e-12
    assert np.isnan(s["alpha"]) and s["cn2_layers"] is not None
    th = engine.theta0_d0_from_profile(Jp[1:], 0.0, engine.LAMBDA_K_NM)[0]
    assert abs(s["theta0_k_zenith"] - th) < 1e-9
    s0 = engine.synthetic_field_snapshot(0.5, 0.3)
    assert s0["cn2_layers"] is None and s0["alpha"] == 0.0
    # an explicit theta0 still overrides, re-weighting the aniso terms
    s2 = engine.synthetic_field_snapshot(0, 0, cn2_layers=Jp,
                                         theta0_k_zenith=0.5 * th)
    assert abs(s2["theta0_k_zenith"] - 0.5 * th) < 1e-9
    assert s2["aniso_scale"] > s["aniso_scale"] * 1.5
    # the layer-mode pair->layers helper round-trips the pair
    lp2 = engine.layers_seeing(engine.layers_from_seeing_pair(0.8, 0.35))
    assert abs(lp2["eps_tot"] - 0.8) < 1e-9 and abs(lp2["eps_fa"] - 0.35) < 1e-9
    print(f"  [ok] engine: layer snapshot (theta0_K {th:.1f}\", m=0, alpha=NaN); "
          f"two-number path untouched; pair->layers round-trips")


def _target_val(win):
    fig = win._fm_holder["canvas"].figure
    imgax = next(ax for ax in fig.axes if ax.images)
    Z = np.asarray(imgax.images[0].get_array())
    return float(Z[Z.shape[0] // 2, Z.shape[1] // 2])


def gui_tab():
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.resize(1550, 950); win.show(); app.processEvents()
    assert len(win.pred_layers) == 7
    assert not win.pred_layers_enable.isChecked()
    assert win.pred_dimm.isEnabled() and win.pred_mass.isEnabled()
    assert win._pred_presets_box.isEnabled()
    assert "not applied" in win.pred_layers_readout.text()
    # fraction rows: 0..1, and the panel starts from the DIMM/MASS pair's
    # own split (sums to 1)
    for sp in win.pred_layers:
        assert abs(sp.maximum() - 1.0) < 1e-9 and sp.minimum() == 0.0
    assert abs(win._pred_layer_fractions().sum() - 1.0) < 1e-9
    # Mauna Kea seeing is never worse than 2": the seeing sliders stop there
    for sp in (win.pred_dimm, win.pred_mass):
        assert abs(sp.maximum() - 2.0) < 1e-9, sp.maximum()
    print("  [ok] GUI: 7 fraction rows (0..1, Σ = 1 at start); DIMM/MASS cap at 2\"")

    # house rule: the control dock never needs to scroll. The layer panel is
    # a "Layers" SUB-TAB of Prediction (not more rows under the scenario,
    # not another top-level tab on the already-full tab bar), and neither
    # page needs a vertical scrollbar at the regress window size.
    assert win.pred_subtabs.tabText(0) == "Scenario"
    assert win.pred_subtabs.tabText(1) == "Layers"
    scroll = win.tabs.widget(win.tabs.count() - 1)
    win.tabs.setCurrentIndex(win.tabs.count() - 1); settle(3)
    for i in (0, 1):
        win.pred_subtabs.setCurrentIndex(i); settle(3)
        assert not scroll.verticalScrollBar().isVisible(), \
            f"Prediction sub-page {i} must not need a vertical scrollbar"
        assert not scroll.horizontalScrollBar().isVisible()
    win.pred_subtabs.setCurrentIndex(1); settle(2)
    # and the readout is not clipped by the squeezed page: its height fits
    # its wrapped text
    ro = win.pred_layers_readout
    assert ro.height() >= ro.heightForWidth(ro.width()) - 1, \
        (ro.height(), ro.heightForWidth(ro.width()))
    print("  [ok] GUI: Layers is a Prediction sub-tab; no scrollbars on either page; "
          "readout unclipped")

    # the button: prior fractions verbatim -> layer mode on, DIMM unchanged
    # and still live, MASS = the prior's aloft share and read-only, presets
    # off, m = 0, theta0 = the prior profile's own
    win.pred_dimm.setValue(0.6); settle(2)
    win.pred_layers_recon_btn.click(); settle(3)
    share = engine.RECON_PRIOR_FRAC[1:].sum()
    assert win.pred_layers_enable.isChecked()
    assert np.allclose(win._pred_layer_fractions(), engine.RECON_PRIOR_FRAC)
    assert abs(win.pred_dimm.value() - 0.6) < 1e-6, win.pred_dimm.value()
    assert abs(win.pred_mass.value() - 0.6 * share ** 0.6) < 6e-3, \
        win.pred_mass.value()                       # MASS spin shows 0.01" steps
    assert win.pred_dimm.isEnabled() and not win.pred_mass.isEnabled()
    assert not win._pred_presets_box.isEnabled()
    assert "m=0.00" in win.pred_layers_readout.text(), win.pred_layers_readout.text()
    assert "Layers = the reconstructor prior" in win.pred_layers_readout.text()
    assert "driving the scenario" in win.pred_layers_readout.text()
    assert "layer strengths" in win.pred_readout.text()
    assert "m=0.00" in win.pred_readout.text()
    snap = win._pred_snapshot()
    assert np.allclose(snap["cn2_bins"], engine.recon_prior_layers(0.6)[1:])
    th_prior = win.pred_theta0.value()
    assert abs(th_prior - snap["theta0_k_zenith"]) < 0.05
    assert win.pred_layers_prof_fig.axes and "Layer profile" in \
        win.pred_layers_prof_fig._suptitle.get_text()     # Layers page showing
    print(f"  [ok] GUI: reconstructor button -> layer mode, DIMM 0.60\" kept (live), "
          f"MASS {win.pred_mass.value():.3f}\" derived, m=0, theta0_K {th_prior:.1f}\"")

    # editing fractions: the sum stays <= 1. Raising 8 km from 0.0926 to
    # 0.60 would give 1.51, so the OTHER six scale down proportionally to
    # share the remaining 0.40 (their mutual ratios kept): total unchanged
    # (DIMM rules), free-atm up, theta0 down, m up
    prior = engine.RECON_PRIOR_FRAC
    d0, m0 = win.pred_dimm.value(), win.pred_mass.value()
    win.pred_layers[5].setValue(0.60); settle(3)
    f = win._pred_layer_fractions()
    assert abs(f[5] - 0.60) < 1e-9 and abs(f.sum() - 1.0) < 1e-9, f
    others = [k for k in range(7) if k != 5]
    want = prior[others] * 0.40 / prior[others].sum()
    assert np.allclose(f[others], want), (f[others], want)
    assert "Layers differ" in win.pred_layers_readout.text()
    d1, m1 = win.pred_dimm.value(), win.pred_mass.value()
    assert abs(d1 - d0) < 1e-9, "the total is the DIMM value, never moved by layers"
    assert m1 > m0 + 0.05, (m0, m1)
    assert win.pred_theta0.value() < th_prior - 1.0
    # lowering a row scales the others UP proportionally: the sum is
    # always exactly 1 (fractions of the whole), ratios of the others kept
    win.pred_layers[0].setValue(0.05); settle(3)
    f2 = win._pred_layer_fractions()
    assert abs(f2.sum() - 1.0) < 1e-9 and abs(f2[0] - 0.05) < 1e-9
    assert np.allclose(f2[1:] / f2[1:].sum(), f[1:] / f[1:].sum())
    assert np.all(f2[1:] > f[1:])
    win.pred_layers[0].setValue(f[0]); settle(3)      # spin rounds to 3 dp
    assert np.allclose(win._pred_layer_fractions(), f, atol=1e-3)
    f = win._pred_layer_fractions()
    # a row taken to 1 empties the others; lowering it again splits the
    # remainder equally (nothing left to keep in ratio)
    win.pred_layers[3].setValue(1.0); settle(3)
    f3 = win._pred_layer_fractions()
    assert abs(f3[3] - 1.0) < 1e-9 and not f3[[0, 1, 2, 4, 5, 6]].any()
    win.pred_layers[3].setValue(0.4); settle(3)
    f4 = win._pred_layer_fractions()
    assert abs(f4.sum() - 1.0) < 1e-9 and np.allclose(f4[[0, 1, 2, 4, 5, 6]], 0.1)
    # exact entry: seven typed values land verbatim, no redistribution
    win.pred_layers_edit.setText("0.5, 0.1 0.05 0.05 0.1 0.15 0.05")
    win.pred_layers_apply_btn.click(); settle(3)
    assert np.allclose(win._pred_layer_fractions(),
                       [0.5, 0.1, 0.05, 0.05, 0.1, 0.15, 0.05])
    assert "normalized" not in win.pred_layers_readout.text()
    assert win.pred_layers_edit.text() == \
        "0.5000 0.1000 0.0500 0.0500 0.1000 0.1500 0.0500"
    # not summing to 1 -> normalized, and said so; malformed -> refused,
    # rows untouched
    win.pred_layers_edit.setText("1 1 1 1 1 1 2"); win.pred_layers_apply_btn.click()
    settle(3)
    assert np.allclose(win._pred_layer_fractions(), np.array([1, 1, 1, 1, 1, 1, 2]) / 8)
    assert "summed to 8.000; normalized" in win.pred_layers_readout.text()
    win.pred_layers_edit.setText("0.5 0.5 abc"); win.pred_layers_apply_btn.click()
    settle(2)
    assert "Apply refused" in win.pred_layers_readout.text()
    assert np.allclose(win._pred_layer_fractions(), np.array([1, 1, 1, 1, 1, 1, 2]) / 8)
    # a slider edit after that clears the note and the text mirrors again
    win.pred_layers[5].setValue(0.60); settle(3)
    f = win._pred_layer_fractions()
    assert "refused" not in win.pred_layers_readout.text()
    assert win.pred_layers_edit.text().split()[5] == "0.6000"
    print("  [ok] GUI: lowering a row scales the others up (Σ = 1 always); "
          "exact entry lands verbatim, normalizes when needed, refuses junk")
    # DIMM edit in layer mode rescales both (MASS is re-derived) -- from
    # the Layers page's own DIMM row, mirrored with the Scenario row
    assert abs(win.pred_layers_dimm.value() - 0.6) < 1e-9
    m1 = win.pred_mass.value()                # at the fractions as they are now
    win.pred_layers_dimm.setValue(1.2); settle(3)
    assert abs(win.pred_dimm.value() - 1.2) < 1e-9, "mirror -> Scenario row"
    assert abs(win.pred_mass.value() - m1 * 2.0) < 0.02, (win.pred_mass.value(), m1)
    win.pred_dimm.setValue(0.6); settle(3)
    assert abs(win.pred_layers_dimm.value() - 0.6) < 1e-9, "Scenario row -> mirror"
    assert abs(win.pred_layers_dimm.maximum() - 2.0) < 1e-9
    print(f"  [ok] GUI: 8 km -> 0.60 scales the others to share 0.40 (Σ = 1); FA "
          f"{m0:.2f}->{m1:.2f}\" at the same total; Layers-page DIMM row mirrors")

    # the Layers page repeats the Scenario page's Cn2 profile plot and it
    # follows the layers live: same twin panels, the 8 km bin at the value
    # just set, the ground point at the ground layer; the Scenario copy is
    # caught up when that page is switched to
    def _fa_gl(fig):
        ax = fig.axes[0]
        fa = next(ln for ln in ax.get_lines() if "free-atm" in ln.get_label())
        gl = next(ln for ln in ax.get_lines() if "ground layer" in ln.get_label())
        return np.asarray(fa.get_xdata(), float), float(gl.get_xdata()[0])
    assert len(win.pred_layers_prof_fig.axes) == 2, "twin panels expected"
    J_now = win._pred_layer_values()
    fa_x, gl_x = _fa_gl(win.pred_layers_prof_fig)
    assert np.allclose(fa_x, J_now[1:]) and abs(gl_x - J_now[0]) < 1e-20
    win.pred_layers[5].setValue(0.20); settle(3)
    fa_x2, _ = _fa_gl(win.pred_layers_prof_fig)
    assert fa_x2[4] < fa_x[4] * 0.6, "8 km bin must drop on the Layers plot"
    assert np.allclose(fa_x2, win._pred_layer_values()[1:])
    win.pred_subtabs.setCurrentIndex(0); settle(3)          # Scenario catches up
    fa_s, _ = _fa_gl(win.pred_prof_fig)
    assert np.allclose(fa_s, fa_x2)
    win.pred_subtabs.setCurrentIndex(1); settle(2)
    print("  [ok] GUI: Layers page repeats the Cn2 profile plot; it tracks the "
          "layers live and the Scenario copy catches up on switch")

    # the field map (no night loaded) follows the layers; Reset restores
    # the prior verbatim (undoing the edits) at the same total
    win.pred_enable.setChecked(True); settle(4)
    assert win.plot_tabs.currentIndex() == 1
    win.fm_mode.setCurrentText("LTAO"); win.fm_metric.setCurrentText("Strehl")
    settle(4)
    t_edit = _target_val(win)
    win.pred_layers_recon_btn.click(); settle(4)
    assert np.allclose(win._pred_layer_fractions(), engine.RECON_PRIOR_FRAC)
    assert abs(win.pred_dimm.value() - 0.6) < 1e-6
    assert "Layers = the reconstructor prior" in win.pred_layers_readout.text()
    t_prior = _target_val(win)
    assert t_prior != t_edit, "field map must respond to the layers"
    print(f"  [ok] GUI: LTAO field-map target {t_edit:.3f} (edited layers) vs "
          f"{t_prior:.3f} (prior); Reset restored the prior")

    # config round-trip: mode + EXACT fractions (m stays 0 after a reload,
    # MASS re-derived on load)
    win.pred_layers[0].setValue(0.30); win.pred_layers[5].setValue(0.20); settle(3)
    f_saved = win._pred_layer_fractions().copy()
    m_saved = win.pred_mass.value()
    cfg = win._collect_config()
    assert cfg["prediction"]["layers_enabled"] and \
        len(cfg["prediction"]["layer_fractions"]) == 7
    win.pred_layers_enable.setChecked(False); settle(2)
    assert win.pred_mass.isEnabled() and win._pred_presets_box.isEnabled()
    assert "not applied" in win.pred_layers_readout.text()
    win.pred_layers[0].setValue(0.05); win.pred_mass.setValue(0.10); settle(2)
    win._apply_config(cfg); settle(3)
    assert win.pred_layers_enable.isChecked() and not win.pred_mass.isEnabled()
    assert np.allclose(win._pred_layer_fractions(), f_saved)
    assert abs(win.pred_mass.value() - m_saved) < 1e-6, (win.pred_mass.value(), m_saved)
    assert abs(win.pred_layers_dimm.value() - win.pred_dimm.value()) < 1e-9
    print("  [ok] GUI: layer mode + exact fractions survive a config round-trip")

    # a config without the new keys (pre-feature) leaves layer mode off
    old = {k: v for k, v in cfg.items()}
    old["prediction"] = {k: v for k, v in cfg["prediction"].items()
                         if not k.startswith("layer")}
    win._apply_config(old); settle(2)
    assert not win.pred_layers_enable.isChecked() and win.pred_mass.isEnabled()
    print("  [ok] GUI: an older config (no layer keys) loads with layer mode off")
    win.tabs.setCurrentIndex(win.tabs.count() - 1)
    win.pred_subtabs.setCurrentIndex(1); settle(2)
    win.grab().save(os.path.join(HERE, "gui_phase42.png"))


def main():
    engine_contract()
    gui_tab()
    print("  [ok] prediction tab: turbulence by layer (reconstructor grid)")


if __name__ == "__main__":
    main()
