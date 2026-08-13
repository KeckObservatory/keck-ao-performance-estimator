"""Prediction tab: a what-if seeing/Cn2-profile snapshot (independent of a
Run), its ground-layer split and Cn2-profile preview, and the free-atm/
wind-weighted-bandwidth readouts it drives.
"""
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
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
                ("dimm",   "DIMM total seeing (zenith)",
                 0.20, 3.00, 0.01, engine.REF_TOTAL,   2, " ″", 100.0),
                ("mass",   "MASS free-atm seeing (zenith)",
                 0.05, 3.00, 0.01, engine.REF_FREEATM, 2, " ″", 100.0),
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

        self.pred_enable.toggled.connect(self._on_pred_toggle)
        self._pred_autoset_theta0()      # initial: auto on -> row read-only
        self._update_pred_readout()
        self._update_pred_profile_plot()
        return self._scroll(w)

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
        self._draw_cn2_profiles(
            self.pred_prof_fig, s["cn2_bins"], j_ground,
            f"Synthesized MK profile — θ₀ᴷ {s['theta0_k_zenith']:.1f}″, "
            f"α={s['alpha']:+.2f}, m={s['m']:.2f}")
        self.pred_prof_canvas.draw_idle()

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
        return engine.synthetic_field_snapshot(
            self.pred_dimm.value(), self.pred_mass.value(),
            self.pred_za.value(), lam,
            theta0_k_zenith=self.pred_theta0.value())

    def _fm_args(self):
        """args for the field-map/terms prediction path: the cached run args
        when a run exists, else a fresh widget collection (collect_args is
        pure widget-reading) with the TT sensor resolved the way
        prepare_night would. None if the controls cannot be parsed."""
        if self.args_cached is not None:
            return self.args_cached
        try:
            a = self.collect_args("")
            engine.resolve_tt_sensor(a)
            return a
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
        th0 = engine.synthetic_field_snapshot(
            self.pred_dimm.value(),
            self.pred_mass.value())["theta0_k_zenith"]
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
        self.pred_readout.setText(
            f"Line of sight at ZA {s['zenith_angle_deg']:g}° "
            f"(X={s['airmass']:.2f}): seeing {s['eps_tot_los']:.2f}″, "
            f"free-atm {s['eps_fa_los']:.2f}″, θ₀ {s['theta0_los']:.1f}″ at "
            f"the science wavelength.  Synthesized Cn² profile: altitude tilt "
            f"α={s['alpha']:+.2f}, LTAO layer mismatch m={s['m']:.2f} (vs the "
            f"reconstructor prior), laser/TT anisoplanatism "
            f"×{s['aniso_scale']:.2f} vs the median MK profile.{clamp}")

    def _on_pred_changed(self, *_):
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

