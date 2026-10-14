"""Sky coverage tab: Strehl versus the fraction of sky that reaches it, for a
chosen telescope, AO mode, guide sensor and tilt-Strehl law, under the
conditions the estimator is using -- the Prediction tab's scenario when it is
enabled, else the loaded night at the Field map tab's Conditions selection.
Every other model setting (LGS offset, winds, NGS fit, WFE sliders, science
band) is read from the left dock. The physics lives in
keck_ao_estimator.sky_coverage.
"""
import copy
from types import SimpleNamespace

import numpy as np
from matplotlib.figure import Figure
from qtcompat import QtWidgets

import keck_ao_estimator as engine

_MODES = {"NGS": "ngs", "single-LGS": "single", "LTAO": "ltao"}
_LAWS = {"Maréchal": "marechal", "Hardy": "hardy"}
_CATALOGUE = {"R": "Gaia G as R", "H": "2MASS H", "K": "2MASS K"}


class SkyCoverageTabMixin:
    def _build_sky_coverage_tab(self):
        holder = self._make_canvas_tab(
            "Run first — or enable the Prediction tab's scenario.")
        self._sc_holder = holder
        self._sc_held = []            # [(label, frac, sr)]
        self._sc_last = None          # (label, result) on screen

        self.sc_tel = QtWidgets.QComboBox()
        self.sc_tel.addItems(["K1", "K2"])
        self.sc_tel.setCurrentText(self.defaults.telescope)
        self.sc_tel.setToolTip(
            "Telescope the curve is for. Follows the Data tab's telescope "
            "when that changes, and can be set to the other one here.")
        self.sc_mode = QtWidgets.QComboBox()
        self.sc_mode.addItems(list(_MODES))
        self.sc_mode.setCurrentText("LTAO" if self.defaults.telescope == "K1"
                                    else "single-LGS")
        self.sc_mode.setToolTip(
            "NGS: each star is tried as the wavefront guide star (R "
            "magnitude, isoplanatic fall-off with its offset). single-LGS / "
            "LTAO: each star is tried as the tip-tilt star.")
        self.sc_sensor = QtWidgets.QComboBox()
        self.sc_sensor.addItems(list(self._TT_SENSOR_MAP))
        self.sc_sensor.setToolTip(
            "Tip-tilt sensor for single-LGS / LTAO. TRICK is K1-only and "
            "its dichroic puts the science band in the other of H / K.")
        self.sc_law = QtWidgets.QComboBox()
        self.sc_law.addItems(list(_LAWS))
        self.sc_law.setToolTip(
            "Strehl factor of the tip-tilt residual. Maréchal exp(−x²) is "
            "what the rest of the estimator uses and what the tip-tilt "
            "sensor rows were calibrated through. Hardy 1/(1+x²) is the "
            "form KAON 1303 / 1318 use: pick it to read a curve against "
            "theirs. It is a convention what-if, not a second calibrated "
            "prediction, and it reads higher at large sky fractions. NGS "
            "has no separate tilt term, so it does not apply there.")
        self.sc_field = QtWidgets.QComboBox()
        for tag, (_l, _b, label) in engine.SKY_FIELDS.items():
            self.sc_field.addItem(label, tag)
        self.sc_field.setCurrentIndex(
            self.sc_field.findData(engine.DEF_SKY_FIELD))
        self.sc_field.setToolTip(
            "Star field the pointings are drawn on. Gaia + 2MASS: the real "
            "stars (Gaia DR2 G < 20.5, 2MASS) in a 0.5° cone on that sight "
            "line. Synthetic: a random field matched to the star counts "
            "KAON 1303 publishes for its Besançon-model fields at b = 60° "
            "(0.40 usable stars / arcmin², power-law magnitudes). It is the "
            "kind of supply the KAON 1303 / 1318 curves were computed on, "
            "with about 2–3× fewer stars brighter than magnitude 15 than "
            "the real field; it is not a Besançon model run.")
        row = QtWidgets.QHBoxLayout()
        for lbl, wdg in (("Telescope:", self.sc_tel), ("Mode:", self.sc_mode),
                         ("TT sensor:", self.sc_sensor),
                         ("Tilt Strehl:", self.sc_law),
                         ("Sky field:", self.sc_field)):
            row.addWidget(QtWidgets.QLabel(lbl)); row.addWidget(wdg)
        row.addStretch(1)

        self.sc_ngs_wfs = QtWidgets.QComboBox()
        self.sc_ngs_wfs.addItems(list(engine.NGS_WFS_MODES))
        self.sc_ngs_wfs.setCurrentText(self.ngs_wfs.currentText())
        self.sc_ngs_wfs.setToolTip(
            "K2 NGS wavefront-sensor mode. 29x29 puts 4× the light in each "
            "subaperture, so it holds to fainter stars at a lower "
            "bright-star Strehl; its fit is preliminary (two nights). "
            "Follows the NGS tab's WFS mode when that changes. K1 has no "
            "mode choice.")
        self.sc_patrol = QtWidgets.QComboBox()
        for r in engine.PATROL_RADII_ARCSEC:
            self.sc_patrol.addItem(f"{r:g}″", r)
        self.sc_patrol.setToolTip(
            "Radius within which a guide star can be picked. 60″ is the "
            "unvignetted, rotator-limited radius (KAON 913) and the field "
            "map's field of regard. 51″ is the radius KAON 1318 used for "
            "its sky-coverage curves.")
        self.sc_hold_btn = QtWidgets.QPushButton("Hold curve")
        self.sc_hold_btn.setToolTip(
            "Keep the curve on screen as a thin line, to compare the next "
            "selection (or the next conditions) against it.")
        self.sc_clear_btn = QtWidgets.QPushButton("Clear held")
        self.sc_kaon = QtWidgets.QCheckBox("KAON 1318 Fig. 12")
        self.sc_kaon.setToolTip(
            "Overlay the KAON 1318 Figure 12 curve for the selected sensor "
            "(STRAP or TRICK-H): the K-band target for KAPA LTAO with one "
            "tip-tilt star at b = 60°, ZA 30°. It was computed on SYNTHETIC "
            "Besançon-model star fields, with the Hardy law and a 51″ "
            "patrol radius. For a like-for-like read, pick the synthetic "
            "sky field, the Hardy law and the 51″ patrol. A target for "
            "the four-beacon system at its own assumed conditions, not a "
            "prediction for these.")
        self.sc_logx = QtWidgets.QCheckBox("log sky axis")
        row2 = QtWidgets.QHBoxLayout()
        self.sc_ngs_wfs_label = QtWidgets.QLabel("K2 NGS WFS:")
        for wdg in (self.sc_ngs_wfs_label, self.sc_ngs_wfs,
                    QtWidgets.QLabel("Patrol:"), self.sc_patrol,
                    self.sc_hold_btn, self.sc_clear_btn, self.sc_kaon,
                    self.sc_logx):
            row2.addWidget(wdg)
        row2.addStretch(1)
        holder["layout"].insertLayout(0, row)
        holder["layout"].insertLayout(1, row2)

        for combo in (self.sc_tel, self.sc_mode, self.sc_sensor, self.sc_law,
                      self.sc_field, self.sc_ngs_wfs, self.sc_patrol):
            combo.currentIndexChanged.connect(self._on_skycov_input)
        self.ngs_wfs.currentTextChanged.connect(self.sc_ngs_wfs.setCurrentText)
        self.sc_kaon.toggled.connect(self._on_skycov_input)
        self.sc_logx.toggled.connect(self._on_skycov_input)
        self.sc_hold_btn.clicked.connect(self._skycov_hold)
        self.sc_clear_btn.clicked.connect(self._skycov_clear)
        self.tel_k1.toggled.connect(
            lambda k1: self.sc_tel.setCurrentText("K1" if k1 else "K2"))
        self._sync_skycov_controls()
        return holder["widget"]

    def _sky_coverage_visible(self):
        return (hasattr(self, "_sc_holder")
                and self.plot_tabs.currentWidget() is self._sc_holder["widget"])

    def _sync_skycov_controls(self):
        """TRICK exists on K1 only; the tip-tilt sensor and the tilt law
        mean nothing for NGS; the NGS WFS mode is a K2 NGS choice."""
        k1 = self.sc_tel.currentText() == "K1"
        lgs = self.sc_mode.currentText() != "NGS"
        if not k1 and self.sc_sensor.currentText().startswith("TRICK"):
            self.sc_sensor.blockSignals(True)
            self.sc_sensor.setCurrentText("STRAP (R)")
            self.sc_sensor.blockSignals(False)
        model = self.sc_sensor.model()
        for i in range(self.sc_sensor.count()):
            if model.item(i).text().startswith("TRICK"):
                model.item(i).setEnabled(k1)
        self.sc_sensor.setEnabled(lgs)
        self.sc_law.setEnabled(lgs)
        for wdg in (self.sc_ngs_wfs, self.sc_ngs_wfs_label):
            wdg.setEnabled(not k1 and not lgs)

    def _on_skycov_input(self, *_):
        self._sync_skycov_controls()
        if not self._loading:
            self._render_sky_coverage_if_visible()

    def _render_sky_coverage_if_visible(self):
        if self._sky_coverage_visible():
            self._render_sky_coverage()

    def _skycov_inputs(self):
        """(args, prep, snap) for the tab's selection under the conditions in
        use, or a string saying why there is nothing to show."""
        if self.pred_enable.isChecked():
            base, prep = self._fm_args(), self._fm_prep()
            if base is None:
                return ("Prediction scenario: fix the control inputs "
                        "(a field could not be parsed).")
            snap = self._pred_snapshot()
        elif self.prep is None or self.res is None:
            return ("Run first — or enable the Prediction tab's scenario to "
                    "use predicted conditions with no night loaded.")
        else:
            base, prep = self.args_cached, self.prep
            when, t_hst = self._fm_when_time()
            snap = engine.field_snapshot(base, prep, self.res, when, t_hst)
            if snap is None:
                return "Sky coverage needs MASS profiles (none this night)."
        a = copy.copy(base)
        a.telescope = self.sc_tel.currentText()
        lgs = self.sc_mode.currentText() != "NGS"
        a.tt_sensor = (self._TT_SENSOR_MAP[self.sc_sensor.currentText()][0]
                       if lgs else "strap")
        engine.resolve_tt_sensor(a)       # TRICK: science band -> complement
        a.ngs_wfs = (self.sc_ngs_wfs.currentText() if a.telescope == "K2"
                     else engine.DEF_NGS_WFS)
        if (a.telescope != base.telescope
                or a.ngs_wfs != getattr(base, "ngs_wfs", None)):
            # the NGS-fit fields hold the fit of the dock's telescope and
            # WFS mode; any other gets its own reference fit (as the Summary
            # panel does for the other telescope)
            par = engine.ngs_fit_params(a.telescope, a.ngs_wfs)
            a.ngs_s0, a.ngs_a = par["S0"], par["A"]
            a.ngs_m0, a.ngs_w = par["m0"], par["w"]
        lam_nm, lam_label = prep.lam_nm, prep.lam_label
        if a.band != base.band:
            lam_nm, lam_label = engine.resolve_wavelength(a)
        sc_prep = SimpleNamespace(
            lam_nm=lam_nm, lam_label=lam_label,
            _ltao_bw_fac=prep._ltao_bw_fac,
            show_target=getattr(prep, "show_target", False))
        return a, sc_prep, snap

    def _render_sky_coverage(self):
        holder = self._sc_holder
        got = self._skycov_inputs()
        if isinstance(got, str):
            self._sc_last = None
            self._show_placeholder(holder, got)
            return
        a, prep, snap = got
        mode = _MODES[self.sc_mode.currentText()]
        sensor = (self._TT_SENSOR_MAP[self.sc_sensor.currentText()][1]
                  if mode != "ngs" else "R")
        law = _LAWS[self.sc_law.currentText()] if mode != "ngs" else "marechal"
        dvar = (self._ngs_delta_var(self.last_offsets, a)
                if (mode == "ngs" and self.last_offsets) else 0.0)
        try:
            with engine.budget_overrides(**self.last_offsets):
                res = engine.sky_coverage_curve(
                    a, prep, snap, mode, sensor, law,
                    self.sc_field.currentData(),
                    patrol=self.sc_patrol.currentData(), ngs_delta_var=dvar)
        except Exception as e:
            self._on_failed(f"sky coverage failed: {e}", "")
            return

        config = f"{a.telescope} {self.sc_mode.currentText()}"
        if mode == "ngs":
            if a.telescope == "K2":
                config += f" {a.ngs_wfs}"
                if a.ngs_wfs != engine.DEF_NGS_WFS:
                    config += " · preliminary fit"
        else:
            config += f" · {self.sc_sensor.currentText()}"
            if law == "hardy":
                config += " · Hardy tilt law"
        field = self.sc_field.currentText()
        short_field = field.split(" (")[0].split(",")[0]
        star = "guide star" if mode == "ngs" else "tip-tilt star"
        title = (f"{config} — Strehl vs sky coverage at {field}\n"
                 f"{self._fm_cond_text(snap, prep.lam_label)}\n"
                 f"best {star} within {res['patrol']:g}″, "
                 + (f"synthetic {sensor}" if res["synthetic"]
                    else _CATALOGUE[sensor])
                 + f" ≤ {res['mag_limit']:g}")
        cond = (f"predicted DIMM {snap['eps_tot_zenith']:.2f}″"
                if snap.get("synthetic")
                else f"seeing {snap['eps_tot_los']:.2f}″")
        if res["patrol"] != engine.PATROL_RADIUS_ARCSEC:
            short_field += f", {res['patrol']:g}″ patrol"
        label = f"{config}, {short_field}, {cond}"
        req = None
        if (self.sc_kaon.isChecked() and mode != "ngs"
                and sensor in engine.KAON1318_FIG12):
            f, s = engine.KAON1318_FIG12[sensor]
            req = (engine.KAON1318_FIG12_LABEL, f, s)
        fig = Figure(figsize=(8, 6), layout="constrained")
        engine.draw_sky_coverage(
            fig, res, title, label, self._sc_held, req,
            log_x=self.sc_logx.isChecked(),
            box_title="\n".join(config.split(" · ") + [short_field, cond]))
        note = self._fm_budget_note(mode, dvar)
        if note:
            fig.text(0.012, 0.012, note, color="#C0392B", fontweight="bold",
                     fontsize=9, ha="left", va="bottom")
        self._sc_last = (label, res)
        self._show_figure(holder, fig)

    def _skycov_hold(self):
        if self._sc_last is None:
            return
        label, res = self._sc_last
        if not any(label == h[0] and np.array_equal(res["sr"], h[2])
                   for h in self._sc_held):
            self._sc_held.append((label, res["frac"], res["sr"]))
        self._render_sky_coverage_if_visible()

    def _skycov_clear(self):
        self._sc_held = []
        self._render_sky_coverage_if_visible()
