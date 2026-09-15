"""Prediction tab: a what-if seeing/Cn2-profile snapshot (independent of a
Run), its ground-layer split and Cn2-profile preview, and the free-atm/
wind-weighted-bandwidth readouts it drives.
"""
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter
from qtcompat import Qt, QtWidgets

import keck_ao_estimator as engine

from ..constants import FM_C_STAR, NIGHTTIME_FM_COND
from ..widgets import _dspin
from ..theme import set_cue


class PredictionTabMixin:
    def _tab_prediction(self):
        """Hypothetical-conditions scenario for the FIELD MAP only: dial in
        zenith DIMM/MASS seeing, theta0 and zenith angle, and the field map
        predicts performance for that scenario instead of the night's data.
        Everything else set elsewhere (WFE sliders, NGS fit, NGS/TT/laser
        geometry, winds, wavelength) carries over unchanged."""
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        note = QtWidgets.QLabel(
            "<b>Predicted scenario — field map and error terms.</b> When "
            "enabled, the <b>Field map</b> and <b>Error terms</b> tabs use "
            "these hypothetical conditions instead of the night's data (the "
            "timeline plot always shows the real night). Seeing values are "
            "<b>at zenith</b> (as MKWC reports); "
            "the zenith angle below projects them onto the line of sight. "
            "All other settings — WFE sliders, NGS fit, NGS/TT/laser "
            "geometry, winds, wavelength — carry over. No Run is required: "
            "with no night loaded the field map and the Error-terms tab "
            "compute directly from the current controls and this scenario.")
        note.setWordWrap(True)
        note.setStyleSheet("QLabel { color:#333; background:#fdf3e7; "
                           "padding:6px; border:1px solid #e0cdb0; }")
        v.addWidget(note)

        self.pred_enable = QtWidgets.QCheckBox("Use this predicted scenario")
        self.pred_enable.setToolTip(
            "Drives the Field map AND Error terms tabs from this scenario "
            "instead of the night's data.")
        self.pred_enable.setMinimumWidth(120)   # floor, not text width
        v.addWidget(self.pred_enable)

        self._pred_rows = {}
        ref_th0 = engine.synthetic_field_snapshot(
            engine.REF_TOTAL, engine.REF_FREEATM)["theta0_k_zenith"]
        for key, label, lo, hi, step, default, dec, suffix, scale in (
                # 2.00" ceiling: Mauna Kea seeing is never worse than that
                ("dimm",   "DIMM total seeing (zenith)",
                 0.20, 2.00, 0.01, engine.REF_TOTAL,   2, " ″", 100.0),
                ("mass",   "MASS free-atm seeing (zenith)",
                 0.05, 2.00, 0.01, engine.REF_FREEATM, 2, " ″", 100.0),
                ("theta0", "θ₀ (K-band, zenith)",
                 1.0, 60.0, 0.5, ref_th0,              1, " ″", 10.0),
                ("za",     "Zenith angle",
                 0.0, 70.0, 1.0, 0.0,                  0, "°",  1.0)):
            row = QtWidgets.QHBoxLayout()
            lbl = QtWidgets.QLabel(label)
            lbl.setMinimumWidth(200)
            slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
            slider.setRange(int(round(lo * scale)), int(round(hi * scale)))
            slider.setValue(int(round(default * scale)))
            spin = _dspin(lo, hi, step, default, dec, suffix)
            row.addWidget(lbl)
            row.addWidget(slider, 1)
            row.addWidget(spin)
            v.addLayout(row)
            self._pred_rows[key] = dict(slider=slider, spin=spin, scale=scale,
                                        hbox=row)
            slider.valueChanged.connect(
                lambda val, s=spin, sc=scale: s.setValue(val / sc))
            spin.valueChanged.connect(
                lambda val, sl=slider, sc=scale: sl.setValue(int(round(val * sc))))
            spin.valueChanged.connect(self._on_pred_changed)
        self.pred_dimm = self._pred_rows["dimm"]["spin"]
        self.pred_mass = self._pred_rows["mass"]["spin"]
        self.pred_theta0 = self._pred_rows["theta0"]["spin"]
        self.pred_za = self._pred_rows["za"]["spin"]

        # theta0 auto-tracks the prior-shape profile at the current seeing
        # unless the user unchecks this to override it explicitly
        self.pred_theta0_auto = QtWidgets.QCheckBox("auto")
        self.pred_theta0_auto.setChecked(True)
        self.pred_theta0_auto.setToolTip(
            "Derive θ₀ from the synthesized profile (reference shape) at the "
            "current free-atm seeing. Uncheck to set θ₀ yourself — the "
            "deviation re-weights the laser/TT anisoplanatism terms.")
        self._pred_rows["theta0"]["hbox"].addWidget(self.pred_theta0_auto)
        self.pred_theta0_auto.toggled.connect(self._on_pred_changed)

        # stacked vertically: side by side the three buttons need ~560 px,
        # which overflows the 520 px dock and clips the whole tab
        # short title -- a QGroupBox's minimum width is its full title text
        # and the long form forced a panel scrollbar (631045c); the full
        # wording lives in the tooltip
        pbox = QtWidgets.QGroupBox("Presets (same total seeing)")
        self._pred_presets_box = pbox
        pbox.setToolTip("Same total seeing, turbulence moved ground↔aloft: "
                        "the presets change only WHERE the turbulence sits.")
        pv = QtWidgets.QVBoxLayout(pbox)
        for name, dimm, mass in self.PRED_PRESETS:
            b = QtWidgets.QPushButton(
                f"{name}   ({dimm:g}″ / {mass:g}″)")
            b.setMinimumWidth(120)   # floor, not text width (631045c)
            b.clicked.connect(
                lambda _=False, d=dimm, m=mass: self._apply_pred_preset(d, m))
            pv.addWidget(b)
        v.addWidget(pbox)

        self.pred_readout = QtWidgets.QLabel()
        self.pred_readout.setWordWrap(True)
        set_cue(self.pred_readout, "secondary")
        v.addWidget(self.pred_readout)

        # Cn^2 density profile of the synthesized scenario (the profile the
        # LTAO layer-mismatch term uses), in the lower space of the tab.
        self.pred_prof_fig = Figure(figsize=(3.4, 3.0))
        self.pred_prof_canvas = FigureCanvas(self.pred_prof_fig)
        self.pred_prof_canvas.setMinimumHeight(240)
        self.pred_prof_canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding)
        v.addWidget(self.pred_prof_canvas, 1)

        # The dock must never need to scroll (house rule): the layer-strength
        # panel is a SUB-TAB of Prediction rather than more rows under the
        # scenario page, and a top-level tab would push the already-full
        # control tab bar further into its overflow arrows.
        self.pred_subtabs = QtWidgets.QTabWidget()
        self.pred_subtabs.addTab(w, "Scenario")
        self.pred_subtabs.addTab(self._build_pred_layers_page(), "Layers")
        self._pred_prof_stale = [False, False]     # per sub-page plot copy
        self.pred_subtabs.currentChanged.connect(self._on_pred_subtab_changed)

        self.pred_enable.toggled.connect(self._on_pred_toggle)
        self._sync_pred_layers_ui()      # initial: layer mode off
        self._pred_autoset_theta0()      # initial: auto on -> row read-only
        self._update_pred_readout()
        self._update_pred_profile_plot()
        return self._scroll(self.pred_subtabs)

    # ---- turbulence by layer (reconstructor altitudes) -----------------------
    # Per-layer controls are turbulence FRACTIONS -- the reconstructor's own
    # units (KAON 1542 sect. 3.4 (2)): the share of the integrated turbulence
    # (Cn2*dh) in each of the 7 layers, summing to 1. The absolute scale is
    # the Scenario page's total (DIMM) seeing; the free-atm (MASS) seeing
    # follows from the aloft share, eps_fa = eps_tot * (sum aloft)^(3/5).
    # The sum is ALWAYS exactly 1 (fractions of the whole; anything else is
    # unphysical): a slider/spin edit of one row rescales the OTHER rows
    # proportionally to fill 1 - value (turbulence moves between layers,
    # their mutual ratios kept), in both directions. To set all seven
    # exactly, without that redistribution, type them into the text row and
    # Apply (normalized to 1 if they do not already sum to it).
    PRED_LAYER_FRAC_STEP = 0.01
    # The exact fractions live in self._pred_layer_frac; the spins DISPLAY
    # them to 3 decimals. A user edit writes that row's exact value from the
    # spin; the reconstructor button writes the unrounded table, so the prior
    # round-trips to m = 0 exactly.

    def _build_pred_layers_page(self):
        """The Prediction tab's "Layers" sub-page: 7 fraction rows on the
        reconstructor's altitude grid (ground + the 6 MASS bins), a button
        that loads the reconstructor's static prior, and a live readout of
        the free-atm / ground seeing the fractions imply at the Scenario
        page's total seeing. While the mode is on, the Scenario page's MASS
        row is DERIVED (read-only), DIMM stays the scale, the presets are
        gated off; the field map and error terms use the aloft 6 layers as
        the Cn2 profile directly."""
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        note = QtWidgets.QLabel(
            "<b>Turbulence by layer.</b> Fractions of the turbulence per "
            "layer of the LTAO reconstructor's grid (sum = 1, the "
            "reconstructor's units). DIMM sets the scale; MASS follows from "
            "the aloft fractions while layer mode is on.")
        note.setWordWrap(True)
        note.setStyleSheet("QLabel { color:#333; background:#fdf3e7; "
                           "padding:6px; border:1px solid #e0cdb0; }")
        v.addWidget(note)
        box = QtWidgets.QGroupBox("Turbulence fractions by layer (Σ = 1)")
        box.setToolTip(
            "Share of the integrated turbulence (Cn²·dh) in each layer at "
            "0 / 0.5 / 1 / 2 / 4 / 8 / 16 km above the summit. The sum is "
            "always 1: moving one row rescales the other rows proportionally "
            "to make up the difference. To set all seven exactly, type them "
            "in the text row and Apply.")
        lv = QtWidgets.QVBoxLayout(box)

        self.pred_layers_enable = QtWidgets.QCheckBox("Set turbulence by layer")
        self.pred_layers_enable.setToolTip(
            "Drive the scenario from the 7 layer fractions below. Total "
            "seeing stays the Scenario page's DIMM value; free-atm = the "
            "aloft share; \u03b8\u2080 (auto) and the layer mismatch m come "
            "from the aloft layers themselves.")
        self.pred_layers_enable.setMinimumWidth(120)   # floor, not text width
        lv.addWidget(self.pred_layers_enable)

        # short label: the dock is ~400 px wide and a QPushButton clips
        # rather than wraps; the tooltip carries the detail
        self.pred_layers_recon_btn = QtWidgets.QPushButton(
            "Reset layers to reconstructor prior")
        self.pred_layers_recon_btn.setMinimumWidth(120)
        self.pred_layers_recon_btn.setToolTip(
            "Load the K1 tomographic reconstructor's static layer prior "
            "(KAON 1542 \u00a73.4: fractions "
            + ", ".join(f"{f:.4f}" for f in engine.RECON_PRIOR_FRAC)
            + " at 0/0.5/1/2/4/8/16 km) into the rows -- also the way to "
            "undo layer edits. Enables layer mode; the total seeing is "
            "unchanged and the free-atm seeing becomes the prior's aloft "
            "share (layer mismatch m = 0).")
        lv.addWidget(self.pred_layers_recon_btn)

        # the total (DIMM) seeing, mirrored two-way with the Scenario page's
        # row so the scale can be set without leaving this page
        drow = QtWidgets.QHBoxLayout()
        dl = QtWidgets.QLabel("DIMM total seeing (zenith)")
        dl.setMinimumWidth(200)
        dr = self._pred_rows["dimm"]
        self.pred_layers_dimm_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.pred_layers_dimm_slider.setRange(dr["slider"].minimum(),
                                              dr["slider"].maximum())
        self.pred_layers_dimm_slider.setValue(dr["slider"].value())
        self.pred_layers_dimm = _dspin(dr["spin"].minimum(), dr["spin"].maximum(),
                                       dr["spin"].singleStep(), dr["spin"].value(),
                                       dr["spin"].decimals(), dr["spin"].suffix())
        self.pred_layers_dimm.setToolTip(
            "Total seeing at zenith -- the same control as on the Scenario "
            "page; it sets the absolute scale of the layer fractions.")
        drow.addWidget(dl)
        drow.addWidget(self.pred_layers_dimm_slider, 1)
        drow.addWidget(self.pred_layers_dimm)
        lv.addLayout(drow)
        self.pred_layers_dimm_slider.valueChanged.connect(
            lambda val, s=self.pred_layers_dimm, sc=dr["scale"]: s.setValue(val / sc))
        self.pred_layers_dimm.valueChanged.connect(
            lambda val, sl=self.pred_layers_dimm_slider, sc=dr["scale"]:
            sl.setValue(int(round(val * sc))))
        self.pred_layers_dimm.valueChanged.connect(self.pred_dimm.setValue)
        self.pred_dimm.valueChanged.connect(self._sync_pred_layers_dimm)

        self._pred_layer_rows = []
        scale = 1000.0
        # start from the DIMM/MASS pair's own split (reconstructor aloft
        # shape) so the panel is never blank
        J0 = engine.layers_from_seeing_pair(self.pred_dimm.value(),
                                            self.pred_mass.value())
        f0 = engine.layers_seeing(J0)["frac"]
        for i, h_m in enumerate(engine.RECON_HEIGHTS_M):
            row = QtWidgets.QHBoxLayout()
            label = ("0 km (ground)" if i == 0
                     else f"{h_m / 1e3:g} km")
            lbl = QtWidgets.QLabel(label)
            lbl.setMinimumWidth(200)
            slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, int(round(1.0 * scale)))
            slider.setValue(int(round(float(f0[i]) * scale)))
            spin = _dspin(0.0, 1.0, self.PRED_LAYER_FRAC_STEP, float(f0[i]), 3)
            spin.setToolTip(f"fraction of the turbulence in the {label} layer")
            row.addWidget(lbl)
            row.addWidget(slider, 1)
            row.addWidget(spin)
            lv.addLayout(row)
            self._pred_layer_rows.append(dict(slider=slider, spin=spin,
                                              scale=scale, hbox=row,
                                              label=lbl))
            slider.valueChanged.connect(
                lambda val, s=spin, sc=scale: s.setValue(val / sc))
            spin.valueChanged.connect(
                lambda val, sl=slider, sc=scale: sl.setValue(int(round(val * sc))))
            spin.valueChanged.connect(
                lambda val, i=i: self._on_pred_layer_spin(i, val))
        self.pred_layers = [r["spin"] for r in self._pred_layer_rows]
        self._pred_layer_frac = [float(x) for x in f0]
        self._pred_layers_note = ""          # one-shot note under the readout

        # exact entry: all seven at once, no proportional redistribution
        erow = QtWidgets.QHBoxLayout()
        el = QtWidgets.QLabel("Exact:")
        self.pred_layers_edit = QtWidgets.QLineEdit()
        self.pred_layers_edit.setMinimumWidth(60)   # floor, never widens the panel
        self.pred_layers_edit.setToolTip(
            "Type the 7 fractions (ground, 0.5, 1, 2, 4, 8, 16 km), separated "
            "by spaces or commas, then Apply (or Enter). They are set "
            "verbatim -- no proportional redistribution -- and normalized to "
            "sum to 1 only if they do not already.")
        self.pred_layers_apply_btn = QtWidgets.QPushButton("Apply")
        self.pred_layers_apply_btn.setMinimumWidth(60)
        erow.addWidget(el)
        erow.addWidget(self.pred_layers_edit, 1)
        erow.addWidget(self.pred_layers_apply_btn)
        lv.addLayout(erow)
        self.pred_layers_apply_btn.clicked.connect(self._apply_pred_layers_text)
        self.pred_layers_edit.returnPressed.connect(self._apply_pred_layers_text)
        self._refresh_pred_layers_text(force=True)

        self.pred_layers_readout = QtWidgets.QLabel()
        self.pred_layers_readout.setWordWrap(True)
        set_cue(self.pred_layers_readout, "secondary")
        lv.addWidget(self.pred_layers_readout)
        v.addWidget(box)
        self._pred_layers_box = box

        # the Scenario page's Cn2 profile plot, repeated here so the profile
        # is in view while the layers are dragged (same twin panels, same
        # data; each page's copy is drawn when that page is showing)
        self.pred_layers_prof_fig = Figure(figsize=(3.4, 3.0))
        self.pred_layers_prof_canvas = FigureCanvas(self.pred_layers_prof_fig)
        self.pred_layers_prof_canvas.setMinimumHeight(200)
        self.pred_layers_prof_canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding)
        v.addWidget(self.pred_layers_prof_canvas, 1)

        self.pred_layers_enable.toggled.connect(self._on_pred_layers_toggle)
        self.pred_layers_recon_btn.clicked.connect(self._apply_recon_prior_layers)
        self._update_pred_layers_readout()
        return page

    def _pred_layer_fractions(self):
        """The 7 layer fractions as entered (exact values, not the spins'
        3-decimal display); may sum to less than 1."""
        return np.array(self._pred_layer_frac, float)

    def _pred_layer_values(self):
        """The 7 layer strengths J [m^1/3, 500 nm, zenith] the fractions
        imply at the Scenario page's total (DIMM) seeing (fractions
        normalized to 1; an all-zero set falls back to the reconstructor
        shape so the scenario never loses its turbulence)."""
        f = self._pred_layer_fractions()
        if f.sum() <= 0.0:
            f = engine.RECON_PRIOR_FRAC
        return engine.fractions_to_layers(f, self.pred_dimm.value())

    def _on_pred_layer_spin(self, i, val):
        """A fraction row was moved: the sum stays exactly 1. The OTHER rows
        are rescaled proportionally (their mutual ratios kept) to fill
        1 - value, whichever way this row went -- turbulence moves between
        this layer and the rest. If the others are all empty the remainder
        is split equally among them (no ratio to keep)."""
        val = min(max(float(val), 0.0), 1.0)
        f = list(self._pred_layer_frac)
        others = [k for k in range(len(f)) if k != i]
        others_sum = float(sum(f[k] for k in others))
        room = 1.0 - val
        for k in others:
            f[k] = (f[k] * room / others_sum if others_sum > 0.0
                    else room / len(others))
        f[i] = val
        self._pred_layers_note = ""
        self._set_pred_layer_fractions(f)          # writes all rows, blocked
        self._on_pred_layer_changed()

    def _refresh_pred_layers_text(self, force=False):
        """Mirror the current fractions into the exact-entry field, unless
        the user is mid-edit there (the field is modified and not applied)."""
        if not force and self.pred_layers_edit.isModified():
            return
        self.pred_layers_edit.setText(
            " ".join(f"{x:.4f}" for x in self._pred_layer_frac))
        self.pred_layers_edit.setCursorPosition(0)   # show the ground end first
        self.pred_layers_edit.setModified(False)

    def _apply_pred_layers_text(self, *_):
        """'Apply': the 7 typed fractions, verbatim (no redistribution),
        normalized to sum 1 only if they do not already. Enables layer
        mode. A malformed entry is refused with the reason in the readout
        and the rows untouched."""
        import re
        text = self.pred_layers_edit.text()
        parts = [t for t in re.split(r"[\s,;]+", text.strip()) if t]
        try:
            vals = [float(t) for t in parts]
        except ValueError:
            vals = None
        n = len(self.pred_layers)
        if vals is None or len(vals) != n or any(
                not np.isfinite(v) or v < 0 for v in vals):
            self._pred_layers_note = (f"Apply refused: need {n} non-negative "
                                      f"numbers, got {text.strip()!r}.")
            self._update_pred_layers_readout()
            return
        tot = float(sum(vals))
        if tot <= 0.0:
            self._pred_layers_note = "Apply refused: all seven are zero."
            self._update_pred_layers_readout()
            return
        if abs(tot - 1.0) > 5e-4:
            vals = [v / tot for v in vals]
            self._pred_layers_note = (f"Typed values summed to {tot:.3f}; "
                                      f"normalized to 1.")
        else:
            self._pred_layers_note = ""
        self._set_pred_layer_fractions(vals)
        self.pred_layers_edit.setModified(False)
        self._refresh_pred_layers_text(force=True)
        if not self._pred_layers_on():
            self.pred_layers_enable.setChecked(True)   # toggle -> sync + changed
        else:
            self._on_pred_layer_changed()

    def _sync_pred_layers_dimm(self, *_):
        """Mirror the Scenario page's DIMM row into the Layers page's copy
        (signals blocked: the Scenario row is the source of truth)."""
        val = self.pred_dimm.value()
        sc = self._pred_rows["dimm"]["scale"]
        for w in (self.pred_layers_dimm, self.pred_layers_dimm_slider):
            w.blockSignals(True)
        self.pred_layers_dimm.setValue(val)
        self.pred_layers_dimm_slider.setValue(int(round(val * sc)))
        for w in (self.pred_layers_dimm, self.pred_layers_dimm_slider):
            w.blockSignals(False)

    def _set_pred_layer_fractions(self, f):
        """Write 7 fractions into the rows without firing the per-row
        handler 7 times; the caller runs _on_pred_layer_changed() once."""
        for i, (r, x) in enumerate(zip(self._pred_layer_rows,
                                       np.asarray(f, float))):
            x = min(max(float(x), 0.0), 1.0)
            self._pred_layer_frac[i] = x
            for w in (r["spin"], r["slider"]):
                w.blockSignals(True)
            r["spin"].setValue(x)
            r["slider"].setValue(int(round(x * r["scale"])))
            for w in (r["spin"], r["slider"]):
                w.blockSignals(False)

    def _apply_recon_prior_layers(self, *_):
        """'Reset layers to reconstructor prior': the prior's fractions into
        the rows (the reconstructor's table, verbatim), layer mode on so the
        MASS row, theta0, the readouts, the profile plot and the field map
        all follow (free-atm becomes the prior's aloft share; m = 0). Also
        how per-layer edits are undone."""
        self._pred_layers_note = ""
        self._set_pred_layer_fractions(engine.RECON_PRIOR_FRAC)
        if not self.pred_layers_enable.isChecked():
            self.pred_layers_enable.setChecked(True)   # toggle -> sync + changed
        else:
            self._on_pred_layer_changed()

    def _pred_layers_at_recon_prior(self):
        """True while the fractions equal the reconstructor prior."""
        return bool(np.allclose(self._pred_layer_fractions(),
                                engine.RECON_PRIOR_FRAC, rtol=0, atol=5e-4))

    def _sync_pred_layers_ui(self):
        """Layer mode on: the MASS row is derived (read-only) and the presets
        (seeing-pair scenarios) are gated off; DIMM stays the scale. Off:
        restored."""
        on = self.pred_layers_enable.isChecked()
        r = self._pred_rows["mass"]
        r["spin"].setEnabled(not on)
        r["slider"].setEnabled(not on)
        self._pred_presets_box.setEnabled(not on)

    def _on_pred_layers_toggle(self, on):
        self._sync_pred_layers_ui()
        if on:
            self._on_pred_layer_changed()       # derive MASS from the layers
        else:
            self._update_pred_layers_readout()  # "(not applied)" state
            self._on_pred_changed()             # DIMM/MASS rows rule again

    def _sync_pred_mass_from_layers(self):
        """Layer mode: mirror the free-atm seeing the fractions imply (at
        the current DIMM total) into the MASS row, signals blocked -- that
        row is derived now. Called from _on_pred_changed so a DIMM edit in
        layer mode re-derives MASS before anything downstream reads it."""
        ls = engine.layers_seeing(self._pred_layer_values())
        r = self._pred_rows["mass"]
        eps = min(max(ls["eps_fa"], r["spin"].minimum()), r["spin"].maximum())
        for w in (r["spin"], r["slider"]):
            w.blockSignals(True)
        r["spin"].setValue(eps)
        r["slider"].setValue(int(round(eps * r["scale"])))
        for w in (r["spin"], r["slider"]):
            w.blockSignals(False)

    def _pred_layers_on(self):
        box = getattr(self, "pred_layers_enable", None)
        return box is not None and box.isChecked()

    def _on_pred_layer_changed(self, *_):
        """A fraction moved (or layer mode came on): refresh the layer
        readout and run the normal scenario update (which re-derives MASS,
        theta0 auto, readouts, profile plot, field map / terms)."""
        self._update_pred_layers_readout()
        if self._pred_layers_on():
            self._on_pred_changed()

    def _update_pred_layers_readout(self):
        J = self._pred_layer_values()
        ls = engine.layers_seeing(J)
        m = engine.layer_mismatch(ls["cn2_bins"])
        # kept to ~3 lines at the dock width: the page must not scroll
        state = ("driving the scenario" if self._pred_layers_on()
                 else "not applied — enable to use")
        origin = ("Layers = the reconstructor prior"
                  if self._pred_layers_at_recon_prior()
                  else "Layers differ from the reconstructor prior")
        note = f" {self._pred_layers_note}" if self._pred_layers_note else ""
        self.pred_layers_readout.setText(
            f"DIMM {ls['eps_tot']:.2f}″ → free-atm "
            f"{ls['eps_fa']:.2f}″ ({100 * float(ls['frac'][1:].sum()):.0f}% "
            f"aloft), ground {ls['eps_ground']:.2f}″, m={m:.2f}. "
            f"{origin}.{note} ({state})")
        if hasattr(self, "pred_layers_edit"):
            self._refresh_pred_layers_text()

    def _pred_layers_config(self):
        return {"layers_enabled": self._pred_layers_on(),
                "layer_fractions": [float(x) for x in self._pred_layer_frac]}

    def _apply_pred_layers_config(self, pc):
        """Restore the fraction rows + mode from a config dict (called inside
        _apply_config with the bulk signal block active for the enable box,
        so the spins are written directly and the mode synced by hand)."""
        f = pc.get("layer_fractions")
        if f is not None and len(f) == len(self.pred_layers):
            self._set_pred_layer_fractions(f)
        self._pred_layers_note = ""
        self._refresh_pred_layers_text(force=True)
        self.pred_layers_enable.setChecked(bool(pc.get("layers_enabled", False)))
        self._sync_pred_layers_ui()
        self._sync_pred_layers_dimm()      # DIMM was applied with signals blocked
        if self._pred_layers_on():
            self._sync_pred_mass_from_layers()
        self._update_pred_layers_readout()

    def _update_pred_profile_plot(self):
        """Redraw the synthesized scenario's Cn² profile as twin panels sharing
        the altitude axis: LEFT the layer-INTEGRATED Cn²·dh per bin (what MASS
        measures and the model uses — the free-atm peak at the 8 km tropopause
        is clearly visible here), RIGHT the Cn² DENSITY (textbook units, but
        MASS's thick bins smear the tropopause so it reads ground-heavy). The 6
        free-atm bins are the tomographic profile; the ground layer (total −
        free-atm, below MASS sensing) is shown separately."""
        s = self._pred_snapshot()
        j_ground = self._ground_layer_j(s["eps_tot_zenith"], s["eps_fa_zenith"])
        if s.get("cn2_layers") is not None:
            title = (f"Layer profile (reconstructor grid) — θ₀ᴷ "
                     f"{s['theta0_k_zenith']:.1f}″, m={s['m']:.2f}")
        else:
            title = (f"Synthesized MK profile — θ₀ᴷ {s['theta0_k_zenith']:.1f}″, "
                     f"α={s['alpha']:+.2f}, m={s['m']:.2f}")
        # the same profile on both sub-pages (Scenario / Layers); only the
        # showing page's copy is drawn now, the other is marked stale and
        # caught up when it is switched to (_on_pred_subtab_changed)
        copies = ((self.pred_prof_fig, self.pred_prof_canvas),
                  (self.pred_layers_prof_fig, self.pred_layers_prof_canvas))
        cur = self.pred_subtabs.currentIndex() if hasattr(self, "pred_subtabs") else 0
        for i, (fig, canvas) in enumerate(copies):
            if hasattr(self, "pred_subtabs") and i != cur:
                self._pred_prof_stale[i] = True
                continue
            self._draw_cn2_profiles(fig, s["cn2_bins"], j_ground, title)
            canvas.draw_idle()
            if hasattr(self, "_pred_prof_stale"):
                self._pred_prof_stale[i] = False

    def _on_pred_subtab_changed(self, idx):
        if self._pred_prof_stale[idx]:
            self._update_pred_profile_plot()

    @staticmethod
    def _ground_layer_j(eps_tot_zenith, eps_fa_zenith):
        """Integrated ground-layer turbulence J(total) − J(free-atm), m^1/3."""
        return max(engine.seeing_to_integrated_cn2(eps_tot_zenith)
                   - engine.seeing_to_integrated_cn2(eps_fa_zenith), 0.0)

    @staticmethod
    def _draw_cn2_profiles(fig, cn2_bins, j_ground, suptitle):
        """Twin-panel Cn² profile sharing the altitude axis: LEFT the layer-
        INTEGRATED Cn²·dh per bin (what MASS measures and the model uses; the
        free-atm/tropopause peak is visible here), RIGHT the Cn² DENSITY
        (textbook units, but MASS's thick bins smear the tropopause). The 6
        free-atm bins are the tomographic profile; the ground layer (total −
        free-atm, below MASS sensing) is shown separately. Shared by the
        prediction tab (synthetic) and the LGS tab (real night)."""
        J = np.asarray(cn2_bins, float)
        h_km, cn2 = engine.cn2_density_profile(J)
        fig.clear()
        ax0, ax1 = fig.subplots(1, 2, sharey=True)
        for ax, xfa, xgnd, xlab in (
                (ax0, J, j_ground, "Cn²·dh  (m$^{1/3}$)"),
                (ax1, cn2, j_ground / 500.0, "Cn²  (m$^{-2/3}$)")):
            ax.plot(xfa, h_km, "-o", color="#1B6CA8", ms=4, lw=1.3,
                    label="free-atm (MASS)")
            if xgnd > 0:
                ax.plot([xgnd], [0.1], "s", color=FM_C_STAR, ms=7,
                        label="ground layer")
            ax.set_xscale("log")
            # a narrow range (within one decade) has no decade tick to label
            # and matplotlib then labels every minor tick, which collide on
            # a 3-inch panel: label 1/2/5 per decade, nothing else
            ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
            ax.xaxis.set_major_formatter(
                LogFormatterSciNotation(base=10.0, labelOnlyBase=False))
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs="auto"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel(xlab, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3, which="both")
            ax.set_ylim(-1, 20)
        ax0.set_ylabel("Altitude (km)", fontsize=8)
        ax0.set_title("integrated", fontsize=8)
        ax1.set_title("density", fontsize=8)
        ax0.legend(fontsize=6, loc="upper right")
        fig.suptitle(suptitle, fontsize=8)
        fig.tight_layout()

    def _pred_snapshot(self):
        """The synthetic snapshot for the current prediction controls, at the
        current science wavelength (prepared night's if available, else the
        Data-tab selection so the readout works before the first Run)."""
        lam = (self.prep.lam_nm if self.prep is not None
               else self._current_wavelength()[0])
        layers = self._pred_layer_values() if self._pred_layers_on() else None
        return engine.synthetic_field_snapshot(
            self.pred_dimm.value(), self.pred_mass.value(),
            self.pred_za.value(), lam,
            theta0_k_zenith=self.pred_theta0.value(), cn2_layers=layers)

    def _fm_args(self):
        """args for the field-map/terms prediction path: the cached run args
        when a run exists, else a fresh widget collection (collect_args is
        pure widget-reading, and resolves the TT sensor the way
        prepare_night would). None if the controls cannot be parsed."""
        if self.args_cached is not None:
            return self.args_cached
        try:
            return self.collect_args("")
        except Exception:
            return None

    def _fm_prep(self):
        """prep for the prediction path: the real prepared night when one
        exists, else a minimal no-run surrogate carrying exactly the
        attributes the field-map/terms path reads (science wavelength and
        the LTAO bandwidth factor; window shading is a night concept)."""
        if self.prep is not None:
            return self.prep
        from types import SimpleNamespace
        lam_nm, lam_label = self._current_wavelength()
        bw_fac = (2.0 ** (5.0 / 6.0) if self.legacy_cb.isChecked()
                  else engine.ltao_bw_factor(float(self.ltao_floor.value())))
        return SimpleNamespace(lam_nm=lam_nm, lam_label=lam_label,
                               _ltao_bw_fac=bw_fac, windows=[],
                               in_any_window=lambda t: False)

    def _gui_telescope(self):
        """Active telescope: from the cached run when one exists, else the
        live radio (the field map can now render before any run)."""
        return (self.args_cached.telescope if self.args_cached is not None
                else ("K1" if self.tel_k1.isChecked() else "K2"))

    def _apply_pred_preset(self, dimm, mass):
        """Set the seeing pair for a preset and restore theta0 auto-tracking
        (presets are self-consistent scenarios; theta0 derives from them).
        The zenith angle is left untouched -- it is sticky, a viewing-geometry
        choice independent of the seeing regime."""
        self.pred_theta0_auto.setChecked(True)
        self.pred_dimm.setValue(dimm)
        self.pred_mass.setValue(mass)

    def _pred_autoset_theta0(self):
        """theta0 auto-track: while the 'auto' box is checked the theta0 row
        is read-only and follows the prior-shape profile at the current
        free-atm seeing (aniso_scale stays 1); unchecking frees it."""
        auto = self.pred_theta0_auto.isChecked()
        r = self._pred_rows["theta0"]
        r["spin"].setEnabled(not auto)
        r["slider"].setEnabled(not auto)
        if not auto:
            return
        layers = self._pred_layer_values() if self._pred_layers_on() else None
        th0 = engine.synthetic_field_snapshot(
            self.pred_dimm.value(),
            self.pred_mass.value(), cn2_layers=layers)["theta0_k_zenith"]
        th0 = min(max(th0, r["spin"].minimum()), r["spin"].maximum())
        if abs(th0 - r["spin"].value()) > 1e-9:
            for w in (r["spin"], r["slider"]):
                w.blockSignals(True)
            r["spin"].setValue(th0)
            r["slider"].setValue(int(round(th0 * r["scale"])))
            for w in (r["spin"], r["slider"]):
                w.blockSignals(False)

    def _update_pred_readout(self):
        s = self._pred_snapshot()
        clamp = ("  ⚠ free-atm clamped to total"
                 if self.pred_mass.value() > self.pred_dimm.value() else "")
        if s.get("cn2_layers") is not None:
            prof = ("Cn² profile from the layer strengths (reconstructor "
                    "altitudes): ")
        else:
            prof = (f"Synthesized Cn² profile: altitude tilt "
                    f"α={s['alpha']:+.2f}, ")
        self.pred_readout.setText(
            f"Line of sight at ZA {s['zenith_angle_deg']:g}° "
            f"(X={s['airmass']:.2f}): seeing {s['eps_tot_los']:.2f}″, "
            f"free-atm {s['eps_fa_los']:.2f}″, θ₀ {s['theta0_los']:.1f}″ at "
            f"the science wavelength.  {prof}"
            f"LTAO layer mismatch m={s['m']:.2f} (vs the "
            f"reconstructor prior), laser/TT anisoplanatism "
            f"×{s['aniso_scale']:.2f} vs the median MK profile.{clamp}")

    def _on_pred_changed(self, *_):
        if self._pred_layers_on():
            self._sync_pred_mass_from_layers()   # MASS is derived in layer mode
        self._pred_autoset_theta0()      # before the readout/render see theta0
        self._update_pred_readout()
        self._update_pred_profile_plot()
        if self.pred_enable.isChecked():
            self._fieldmap_dirty = True
            self._render_field_map_if_visible()
            self._terms_dirty = True
            self._render_terms_if_visible()

    def _sync_pred_ui(self):
        """Reflect the prediction enable state: the field-map Conditions
        selector is meaningless under a synthetic scenario."""
        on = self.pred_enable.isChecked()
        self.fm_cond.setEnabled(not on)
        self.fm_time.setEnabled(
            not on and self.fm_cond.currentText() == "specific time")
        self._fieldmap_dirty = True
        self._terms_dirty = True
        self._pred_autoset_theta0()
        self._update_pred_readout()

    def _on_pred_toggle(self, on):
        """Enable/disable the prediction; enabling jumps to the Field map tab
        so the effect is immediately visible. Works with no run loaded --
        the field map and Error-terms tab fall back to widget-collected args
        and a no-run prep surrogate (_fm_args/_fm_prep)."""
        self._sync_pred_ui()
        if on and self.prep is None:
            # no run has stamped last_offsets yet -- pick up any WFE
            # sliders already off-default so the first render honors them
            self.last_offsets = self.current_offsets()
        if on:
            self.plot_tabs.setCurrentIndex(1)      # renders via tab-change hook
        self._render_field_map_if_visible()
        self._render_terms_if_visible()

    # ---- small UI utilities -------------------------------------------------
    def _update_m_readout(self, args, res):
        """Refresh the layer-mismatch m summary on the LGS tab from the run."""
        if args.legacy_budget:
            self.m_label.setText("n/a — legacy budget (no layer-mismatch penalty)")
            return
        mm = np.asarray(res.col_mm, float)
        mm = mm[np.isfinite(mm)]
        if mm.size == 0:
            self.m_label.setText("n/a — no MASS profiles")
            return
        applied = "applied to LTAO" if self.prep.tomography_on else \
                  "computed (tomography off — not applied)"
        self.m_label.setText(
            f"mean {mm.mean():.2f}   median {np.median(mm):.2f}   "
            f"range {mm.min():.2f}–{mm.max():.2f}   ({mm.size} profiles; {applied})")

    def _fm_when_time(self):
        """(when, t_hst) from the field-map Conditions selector. Thin
        delegate to _when_time_from -- see that method for the semantics;
        kept so existing field-map call sites don't need to change."""
        return self._when_time_from(self.fm_cond, self.fm_time)

    def _when_time_from(self, cond_combo, time_edit):
        """(when, t_hst) from ANY Conditions-selector combo + time-edit pair
        built to the same 4-option contract as the field map's (fm_cond,
        fm_time): 'observing window' / 'whole night' / 'specific time' /
        NIGHTTIME_FM_COND. `when` is 'window' / 'night' / 'time' with the
        HST datetime for the time case. NIGHTTIME_FM_COND (Nighttime mode's
        "time of last pull") delegates to NighttimeModeMixin, which falls
        back to a whole-night median if no pull has happened yet. Shared by
        the field map (self.fm_cond/fm_time) and the Data-tab summary-stats
        panel (self.stats_cond/stats_time), so both agree on what each
        option means without duplicating the midnight-rollover logic."""
        text = cond_combo.currentText()
        if text == NIGHTTIME_FM_COND:
            return self._nighttime_fm_time()
        when = {"observing window": "window", "whole night": "night",
                "specific time": "time"}[text]
        t_hst = None
        if when == "time" and self.prep is not None:
            from datetime import timedelta
            qt = time_edit.time()
            h = qt.hour()
            if self._utc():
                # UTC mode: the typed clock time is UT -- convert to the
                # HST wall time everything internal runs on (UTC-10)
                h = (h - 10) % 24
            t_hst = self.prep.night_date.replace(hour=h, minute=qt.minute())
            if h < 12:                           # after-midnight clock time
                t_hst = t_hst + timedelta(days=1)
        return when, t_hst

    def _update_lgs_profile_plot(self, *_):
        """Redraw the night's real Cn² profile on the LGS tab for the field-map
        conditions (whole-night / observing-window mean, or the exact profile
        at a set time)."""
        fig = self.lgs_prof_fig
        if self.prep is None or self.res is None:
            return                                # keep the build placeholder
        when, t_hst = self._fm_when_time()
        prof = engine.field_cn2_profile(self.args_cached, self.prep, self.res,
                                        when, t_hst)
        if prof is None:
            fig.clear(); ax = fig.add_subplot(111); ax.axis("off")
            ax.text(0.5, 0.5, "No MASS profiles this night", ha="center",
                    va="center", fontsize=9, color="#777")
        else:
            j_ground = self._ground_layer_j(prof["eps_tot_zenith"],
                                            prof["eps_fa_zenith"])
            self._draw_cn2_profiles(
                fig, prof["cn2_mean"], j_ground,
                f"Night Cn² profile — {self._tz_text(prof['when_desc'])} "
                f"(n={prof['n']}, seeing {prof['eps_tot_zenith']:.2f}″)")
        self.lgs_prof_canvas.draw_idle()

