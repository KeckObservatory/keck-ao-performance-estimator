"""NGS tab: guide-star magnitudes/offset, the assumed-theta0 fallback,
the seeing law, and the editable NGS Gompertz-fit coefficients with their
live fit-preview plot.
"""
from types import SimpleNamespace

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtcompat import QtWidgets

import keck_ao_estimator as engine

from ..widgets import OffsetEntry, _dspin


class NgsTabMixin:
    def _tab_ngs(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        self.ngs_bright = _dspin(0, 25, 0.1, self.defaults.ngs_bright, 1, " mag")
        self.ngs_faint = _dspin(0, 25, 0.1, self.defaults.ngs_faint, 1, " mag")
        self.ngs_offset = OffsetEntry(self.defaults.ngs_offset,
                                      self._effective_science_coords,
                                      fixable=True)
        self.assumed_theta0 = _dspin(0, 60, 0.5, self.defaults.assumed_theta0, 1, '"')
        self.seeing_law = QtWidgets.QComboBox()
        self.seeing_law.addItems(["kolmogorov", "gaussian"])
        self.seeing_law.setCurrentText(self.defaults.ngs_seeing_law)
        f.addRow("NGS bright:", self.ngs_bright)
        f.addRow("NGS faint:", self.ngs_faint)
        f.addRow("NGS offset:", self.ngs_offset)
        # own label widget so it can carry a width floor -- this is the
        # widest label on the tab and the form's minimum is
        # max(label col) + max(field col) (631045c)
        _th0_lbl = QtWidgets.QLabel("Assumed θ₀ (K, zenith):")
        _th0_lbl.setMinimumWidth(140)
        f.addRow(_th0_lbl, self.assumed_theta0)
        f.addRow("Seeing law:", self.seeing_law)

        # --- NGS Gompertz fit editor (recalibration, telescope-aware) --------
        # The empirical on-sky fit itself, S = S0·exp(-A·sK²)·exp(-exp((R-m0)/w)),
        # with every term editable. Values track the ACTIVE telescope's fit and
        # repopulate when the telescope changes (see _sync_ngs_fit_fields).
        gnote = QtWidgets.QLabel(
            "<b>NGS Gompertz fit</b> (empirical on-sky fit for the active "
            "telescope — editing recalibrates the model, it is not a budget "
            "what-if):<br>"
            "&nbsp;&nbsp;S = S₀ · exp(−A·s<sub>K</sub>²) · "
            "exp(−exp((R−m₀)/w))<br>"
            "<small>These are the <b>K-band</b> fit values; other science "
            "wavelengths are extrapolated from them (Maréchal), as in the "
            "estimate.</small>")
        gnote.setWordWrap(True)
        gnote.setStyleSheet("QLabel { color:#333; background:#f7f0ee; "
                            "padding:6px; border:1px solid #dcc; }")
        f.addRow(gnote)
        _k1 = engine.NGS_PARAMS["K1"]      # seed with the startup telescope's
        _k2 = engine.NGS_PARAMS["K2"]      # fit; _sync repopulates per telescope
        _seed = _k1 if self.defaults.telescope == "K1" else _k2
        self.ngs_s0 = _dspin(0.05, 1.0, 0.01, _seed["S0"], 3)
        self.ngs_a  = _dspin(0.05, 3.0, 0.05, _seed["A"], 3)
        self.ngs_m0 = _dspin(5.0, 22.0, 0.1, _seed["m0"], 2)
        self.ngs_w  = _dspin(0.2, 5.0, 0.05, _seed["w"], 2)
        self.ngs_s0.setToolTip(
            f"Bright-star ceiling S₀ (K2 {_k2['S0']:g} / K1 {_k1['S0']:g}). "
            "The fit is calibrated to optimized on-sky data (typically after "
            "a Fast & Furious run, worth ~10 K-band Strehl points): drop S₀ "
            "by 0.1 to estimate performance without F&F.")
        self.ngs_a.setToolTip(f"Seeing exponent A in exp(−A·s_K²) (K2 "
                              f"{_k2['A']:g} / K1 {_k1['A']:g}; K1 steeper for "
                              f"DM-stroke saturation)")
        self.ngs_m0.setToolTip(f"Faint-end midpoint m₀, R mag (K2 {_k2['m0']:g} "
                               f"/ K1 {_k1['m0']:g})")
        self.ngs_w.setToolTip(f"Roll-off width w (K2 {_k2['w']:g} / "
                              f"K1 {_k1['w']:g})")
        f.addRow("  ceiling S₀:", self.ngs_s0)
        f.addRow("  seeing exponent A:", self.ngs_a)
        f.addRow("  faint midpoint m₀:", self.ngs_m0)
        f.addRow("  roll-off width w:", self.ngs_w)

        # K1-only post-fit quadcell penalty (separate from the Gompertz fit)
        self.k1_quadcell = _dspin(0.0, 0.5, 0.01,
                                  self.defaults.k1_quadcell_penalty, 3)
        self.k1_quadcell.setToolTip(
            f"K1 only: flat Strehl subtracted for the KAPA PRO quadcell-"
            f"saturation effect. Default {engine.NGS_K1_QUADCELL_PENALTY:g}; "
            f"set 0 to remove it.")
        f.addRow("K1 quadcell penalty:", self.k1_quadcell)

        # reset-to-fit button (restore the active telescope's fitted values)
        reset_fit = QtWidgets.QPushButton("Reset fit to telescope default")
        reset_fit.clicked.connect(lambda: self._sync_ngs_fit_fields(force=True))
        reset_fit.setMinimumWidth(100)   # floor, not text width (631045c)
        f.addRow("", reset_fit)

        # live preview of the fit itself: K-band Strehl vs guide-star magnitude
        # at a few seeing values, redrawn as the terms are edited. Needs no data
        # (it is just the fit function), so it updates even before a Run.
        self.fit_fig = Figure(figsize=(3.4, 2.7))
        self.fit_canvas = FigureCanvas(self.fit_fig)
        self.fit_canvas.setMinimumHeight(230)
        self.fit_canvas.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                      QtWidgets.QSizePolicy.Policy.Expanding)
        f.addRow(self.fit_canvas)

        # all NGS controls are compute-only -> live recompute
        self._ngs_fit_spins = [self.ngs_s0, self.ngs_a, self.ngs_m0, self.ngs_w]
        for sp in (self.ngs_bright, self.ngs_faint,
                   self.assumed_theta0, self.k1_quadcell, *self._ngs_fit_spins):
            sp.valueChanged.connect(self._on_compute_changed)
        self.seeing_law.currentTextChanged.connect(self._on_compute_changed)
        # the offset entry (multi-mode) reports via its own signal; a mode that
        # needs coordinates also affects Run-enable, so re-validate too
        self.ngs_offset.changed.connect(self._on_compute_changed)
        self.ngs_offset.changed.connect(self._validate)
        self.ngs_offset.changed.connect(self._on_fieldmap_input_changed)
        self.ngs_offset.pos_changed.connect(self._on_fieldmap_input_changed)
        # redraw the fit preview on any change that alters the fit, the marks,
        # or the science wavelength (band / exact-nm live on the Data tab)
        for sp in (self.ngs_bright, self.ngs_faint, self.k1_quadcell,
                   self.wl_nm, *self._ngs_fit_spins):
            sp.valueChanged.connect(self._update_ngs_fit_plot)
        self.seeing_law.currentTextChanged.connect(self._update_ngs_fit_plot)
        self.band_combo.currentTextChanged.connect(self._update_ngs_fit_plot)
        self.wl_enable.toggled.connect(self._update_ngs_fit_plot)
        # repopulate the fit for the active telescope on telescope change
        self.tel_k1.toggled.connect(self._sync_ngs_fit_fields)
        self._sync_ngs_fit_fields()
        self._update_ngs_fit_plot()
        return self._scroll(w)

    def _current_wavelength(self):
        """Resolve the science wavelength (nm, label) from the Data-tab band /
        exact-nm controls, reusing the engine's own resolver so the preview
        can never disagree with a run."""
        a = SimpleNamespace(
            wavelength=(float(self.wl_nm.value())
                        if self.wl_enable.isChecked() else None),
            band=self.band_combo.currentText())
        return engine.resolve_wavelength(a)

    def _update_ngs_fit_plot(self, *_):
        """Draw the current NGS Gompertz fit: Strehl vs R magnitude at a few
        K-band seeing values, AT THE SELECTED SCIENCE WAVELENGTH. The fit is a
        K-band fit; other wavelengths are the same Maréchal extrapolation the
        estimate uses (ngs_strehl backs out the implied WFE and re-evaluates).
        Uses the live field values through the engine."""
        if not getattr(self, "fit_canvas", None):
            return
        tel = "K1" if self.tel_k1.isChecked() else "K2"
        lam_nm, lam_label = self._current_wavelength()
        is_k = abs(lam_nm - engine.LAMBDA_K_NM) < 1.0
        kw = dict(ngs_s0=self.ngs_s0.value(), ngs_a=self.ngs_a.value(),
                  ngs_m0=self.ngs_m0.value(), ngs_w=self.ngs_w.value(),
                  k1_quadcell=self.k1_quadcell.value(),
                  seeing_law=self.seeing_law.currentText())
        R = np.linspace(6.0, 18.0, 90)
        fig = self.fit_fig
        fig.clear()
        ax = fig.add_subplot(111)
        try:
            for epsK, col in ((0.3, "#2E8B57"), (0.5, "#B26A00"), (0.7, "#C0392B")):
                eps500 = epsK / engine.V2K
                S = [engine.ngs_strehl(eps500, r, tel, lam_nm, **kw) for r in R]
                ax.plot(R, S, "-", lw=1.3, color=col, label=f'{epsK:g}" K seeing')
            for mag, lbl, c in ((self.ngs_bright.value(), "bright", "#6A3D9A"),
                                (self.ngs_faint.value(), "faint", "#888")):
                if R[0] <= mag <= R[-1]:
                    ax.axvline(mag, color=c, ls=":", lw=1.1)
                    ax.text(mag, 0.02, lbl, rotation=90, fontsize=6.5,
                            color=c, va="bottom", ha="right")
        except Exception:
            pass
        ax.set_xlim(R[0], R[-1]); ax.set_ylim(0, 1)
        ax.set_xlabel("NGS guide-star R (mag)", fontsize=8)
        ax.set_ylabel(f"Strehl @ {lam_label}", fontsize=8)
        title = f"{tel} NGS fit preview — {lam_label}"
        if not is_k:
            title += "\n(extrapolated from the K-band fit)"
        ax.set_title(title, fontsize=8.5, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
        fig.tight_layout(pad=0.4)
        self.fit_canvas.draw_idle()

    def _sync_ngs_fit_fields(self, *_, force=False):
        """Load the active telescope's Gompertz fit into the editor fields
        (the fit is telescope-specific, so switching telescope replaces them),
        and grey the K1-only quadcell penalty off on K2. Signals are blocked
        so this repopulation does not itself fire a recompute -- the telescope
        toggle already triggers a re-prepare. `force` re-seeds even without a
        telescope change (the Reset button)."""
        tel = "K1" if self.tel_k1.isChecked() else "K2"
        if not force and getattr(self, "_ngs_fit_tel", None) == tel:
            self.k1_quadcell.setEnabled(tel == "K1")
            return
        self._ngs_fit_tel = tel
        par = engine.NGS_PARAMS[tel]
        for sp, key in ((self.ngs_s0, "S0"), (self.ngs_a, "A"),
                        (self.ngs_m0, "m0"), (self.ngs_w, "w")):
            sp.blockSignals(True)
            sp.setValue(par[key])
            sp.blockSignals(False)
        self.k1_quadcell.setEnabled(tel == "K1")
        self._update_ngs_fit_plot()          # fields changed under blockSignals
        if force and self.prep is not None:
            self._schedule("recompute")

