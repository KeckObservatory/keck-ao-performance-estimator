"""WFE sliders tab: the adjustable error-budget what-if sliders (grouped by
how each term scales with seeing), per-term/all reset, and the modified-
budget summary line.
"""
from types import SimpleNamespace

import numpy as np
from qtcompat import Qt, QtWidgets

import keck_ao_estimator as engine

from ..constants import WFE_SCALING
from ..widgets import _dspin, _shrinkable_label


class WfeTabMixin:
    def _tab_wfe(self):
        """One row per ADJUSTABLE_BUDGET_PARAMS entry, arranged into groups
        (fixed / common / LGS-only; see WFE_GROUPS). Built now (from the
        registry, so no invented numbers); wired live in Phase 3."""
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        # note: these are reference-profile values; per-night they scale with
        # seeing (numbers pulled from the engine, not invented).
        note = QtWidgets.QLabel(
            f"Values are <b>nm RMS at the reference profile</b> "
            f"(total seeing {engine.REF_TOTAL:g}″, free-atm "
            f"{engine.REF_FREEATM:g}″). Each night's per-sample term <b>scales "
            f"with that night's seeing</b> as tagged per row:<br>"
            f"• <b>total seeing</b> → ×(total/{engine.REF_TOTAL:g})<sup>5/6</sup>"
            f" &nbsp; • <b>free-atm</b> → ×(free-atm/{engine.REF_FREEATM:g})"
            f"<sup>5/6</sup> &nbsp; • <b>weak</b> → <sup>1/3</sup> &nbsp; "
            f"• <b>fixed</b> → seeing-independent.")
        note.setWordWrap(True)
        note.setStyleSheet("QLabel { color:#333; background:#eef3f7; "
                           "padding:6px; border:1px solid #cdd; }")
        v.addWidget(note)

        # reference-value set picker: v3_1_3 (the defaults) vs the previous
        # v3_1_1 baseline. Selecting a version just SETS the sliders -- the
        # what-if override machinery does the rest, so provenance and the
        # off-default summary stay truthful.
        ver_row = QtWidgets.QHBoxLayout()
        ver_lbl = QtWidgets.QLabel("Budget values:")
        self.budget_version_combo = QtWidgets.QComboBox()
        self.budget_version_combo.addItems(
            ["3_1_3 (default)", "3_1_1 (legacy values)"])
        self.budget_version_combo.setToolTip(
            "Reference-value set for every slider below.\n"
            "3_1_3 — AO performance error budget v3_1_3, re-referenced to "
            "zenith from its ZA=50° column (the defaults).\n"
            "3_1_1 — the previous baseline values, which carried the ZA-50 "
            "scaling inside and double-counted zenith degradation once the "
            "per-night line-of-sight projection was applied.\n"
            "Selecting 3_1_1 sets the sliders to those values (a MODIFIED "
            "budget vs the v3_1_3 defaults); CLI equivalent: "
            "--budget-version.")
        self.budget_version_combo.currentTextChanged.connect(
            self._apply_budget_version_choice)
        ver_row.addWidget(ver_lbl)
        ver_row.addWidget(self.budget_version_combo)
        ver_row.addStretch(1)
        v.addLayout(ver_row)

        self.wfe_rows = {}       # name -> dict(slider, spin, default, reset)
        self._wfe_tel_rows = {"K1": [], "K2": []}  # tel -> [telescope-gated rows]
        SCALE = 10.0             # slider is int; 0.1 nm resolution
        params = dict(engine.ADJUSTABLE_BUDGET_PARAMS)
        for gname, gtip, names in self.WFE_GROUPS:
            box = QtWidgets.QGroupBox(gname)
            if gtip:
                box.setToolTip(gtip)
            gv = QtWidgets.QVBoxLayout(box)
            for name in names:
                desc, lo, hi = params[name]
                default = engine.get_budget_param(name)
                row = QtWidgets.QWidget()
                hl = QtWidgets.QHBoxLayout(row)
                hl.setContentsMargins(0, 0, 0, 0)
                tag = WFE_SCALING.get(name, "")
                fixed = (tag == "fixed")
                lbl = QtWidgets.QLabel(
                    f"{desc}<br><small>[{name}] · scales: "
                    f"<b style='color:{'#888' if fixed else '#1B6B3A'}'>{tag}"
                    f"</b></small>")
                # 200 not 230, and an explicit floor on the spinbox below: a
                # row's minimum width is the SUM of these floors (label +
                # slider + spin + reset), and at 230 + the spinbox's natural
                # ~113 the whole tab's minimum exceeded the control panel ->
                # horizontal scrollbar. Both are floors, not fixed widths --
                # the row still lays out at its natural size when there's room
                lbl.setMinimumWidth(200)
                tip = (f"Reference value {default:g} nm — AO performance "
                       f"error budget v3_1_3 (zenith-referenced). The "
                       f"version picker above swaps in the v3_1_1 legacy "
                       f"values.")
                lbl.setToolTip(tip)
                slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
                slider.setRange(int(lo * SCALE), int(hi * SCALE))
                slider.setValue(int(default * SCALE))
                slider.setToolTip(tip)
                spin = _dspin(lo, hi, 0.5, default, 1, " nm")
                spin.setMinimumWidth(70)
                spin.setToolTip(tip)
                reset = QtWidgets.QPushButton("↺")
                reset.setFixedWidth(28)
                reset.setToolTip(f"reset to the v3_1_3 default {default:g} nm")
                hl.addWidget(lbl)
                hl.addWidget(slider, 1)
                hl.addWidget(spin)
                hl.addWidget(reset)
                gv.addWidget(row)
                self.wfe_rows[name] = dict(slider=slider, spin=spin,
                                           default=float(default), scale=SCALE)
                # keep slider<->spin in sync, then trigger debounced recompute
                slider.valueChanged.connect(
                    lambda val, s=spin, sc=SCALE: s.setValue(val / sc))
                spin.valueChanged.connect(
                    lambda val, sl=slider, sc=SCALE: sl.setValue(int(round(val * sc))))
                spin.valueChanged.connect(self._on_wfe_changed)
                reset.clicked.connect(lambda _=False, n=name: self._reset_wfe(n))
                if name in ("FITTING_ERR_K1", "FITTING_ERR_K2",
                            "STATIC_TEL_K1", "STATIC_TEL_K2"):
                    self._wfe_tel_rows[name[-2:]].append(row)
                elif name == "TOMO_ERR":
                    self._wfe_tomo_row = row
            v.addWidget(box)

        self.wfe_summary = QtWidgets.QLabel()
        _shrinkable_label(self.wfe_summary)
        reset_all = QtWidgets.QPushButton("Reset all")
        reset_all.clicked.connect(self._reset_all_wfe)
        foot = QtWidgets.QHBoxLayout()
        foot.addWidget(self.wfe_summary, 1)
        foot.addWidget(reset_all)
        v.addLayout(foot)
        v.addStretch(1)
        self._update_wfe_summary()
        self.tel_k1.toggled.connect(self._sync_wfe_tel_rows)
        self.tel_k1.toggled.connect(self._sync_wfe_tomo_row)
        self.tomo_combo.currentTextChanged.connect(self._sync_wfe_tomo_row)
        self._sync_wfe_tel_rows()
        self._sync_wfe_tomo_row()
        return self._scroll(w)

    def _sync_wfe_tel_rows(self, *_):
        """Only the fitting-error row for the SELECTED telescope is relevant
        (the other telescope's DM actuator count doesn't apply); grey it out
        rather than hiding it so its value/history stays visible."""
        active = "K1" if self.tel_k1.isChecked() else "K2"
        for tel, rows in self._wfe_tel_rows.items():
            for row in rows:
                row.setEnabled(tel == active)

    def _sync_assumed_theta0(self):
        """The assumed-theta0 spin is a FALLBACK for nights without a MASS
        profile: with masspro loaded, theta0 is measured per sample and the
        spin is ignored everywhere (timeline aniso + field-map snapshot). Grey
        it out then, so a dead knob looks dead."""
        has_mass = (self.res is not None and len(self.res.col_theta0) > 0
                    and bool(np.isfinite(self.res.col_theta0).any()))
        self.assumed_theta0.setEnabled(not has_mass)
        self.assumed_theta0.setToolTip(
            "This night has MASS profiles: θ₀ is MEASURED per sample and this "
            "fallback is ignored (see the field-map title for the θ₀ in use)."
            if has_mass else
            "Fallback θ₀ (K band, zenith) used for the off-axis NGS "
            "anisoplanatism when the night has no MASS profile.")

    def _effective_tomography_on(self):
        """Mirror the engine's resolve_tomography() from the live telescope +
        tomography-combo selection, without needing a prepared night."""
        tel = "K1" if self.tel_k1.isChecked() else "K2"
        tomo = self.tomo_combo.currentText()
        a = SimpleNamespace(
            telescope=tel, tomography=(None if tomo.startswith("auto")
                                        else (tomo == "on")))
        return engine.resolve_tomography(a)

    def _sync_wfe_tomo_row(self, *_):
        """The LTAO tomography residual only applies when tomography is on
        (K2's default is off); grey out the row rather than hiding it."""
        self._wfe_tomo_row.setEnabled(self._effective_tomography_on())

    # ---- prediction tab (§field map: hypothetical-conditions scenario) ------
    # Preset seeing pairs (total DIMM, free-atm MASS) at zenith, arcsec. All
    # three fix the SAME total seeing (0.5") and vary only the free-atm split,
    # so the presets isolate how the turbulence LOCATION (ground vs aloft) --
    # not its strength -- drives theta0 and the off-axis performance. Each
    # preset's theta0 is DERIVED from the synthesized profile (median MK shape,
    # alpha=0) rather than invented -- see synthetic_field_snapshot.
    PRED_PRESETS = [
        ("Reference",              engine.REF_TOTAL, engine.REF_FREEATM),
        ("Free-atm dominated",     engine.REF_TOTAL, 0.45),
        ("Ground-layer dominated", engine.REF_TOTAL, 0.15),
    ]

    def _apply_budget_version_choice(self, *_):
        """Version picker changed: set every slider to that version's value
        (fires the normal what-if machinery; 3_1_3 == reset to defaults)."""
        ver = ("3_1_1" if self.budget_version_combo.currentText()
               .startswith("3_1_1") else "3_1_3")
        vals = engine.BUDGET_VERSIONS[ver]
        for name, r in self.wfe_rows.items():
            r["spin"].setValue(float(vals[name]))
        self._update_wfe_summary()

    def _budget_version_offsets(self, ver="3_1_1"):
        """The override dict selecting budget version `ver` (params whose
        value differs from the engine defaults)."""
        return {k: v for k, v in engine.BUDGET_VERSIONS[ver].items()
                if abs(v - engine.BUDGET_DEFAULTS[k]) > 1e-9}

    def _reset_wfe(self, name):
        r = self.wfe_rows[name]
        r["spin"].setValue(r["default"])       # fires _on_wfe_changed
        self._update_wfe_summary()

    def _reset_all_wfe(self):
        for name in self.wfe_rows:
            self._reset_wfe(name)

    def _on_wfe_changed(self, *_):
        """A WFE slider/spin moved: refresh the off-default summary and, if a
        night is already prepared, schedule a debounced live recompute."""
        self._update_wfe_summary()
        self._schedule("recompute")

    def current_offsets(self):
        """WFE parameters whose value differs from the engine default -> dict
        suitable for budget_overrides(**offsets)."""
        out = {}
        for name, r in self.wfe_rows.items():
            v = float(r["spin"].value())
            if abs(v - r["default"]) > 1e-9:
                out[name] = v
        return out

    def _update_wfe_summary(self):
        offs = self.current_offsets()
        if not offs:
            self.wfe_summary.setText("All parameters at the v3_1_3 defaults.")
        elif offs == self._budget_version_offsets("3_1_1"):
            self.wfe_summary.setText(
                f"Budget values v3_1_1 (legacy) selected — {len(offs)} "
                "parameter(s) off the v3_1_3 defaults (MODIFIED BUDGET).")
        else:
            self.wfe_summary.setText(f"{len(offs)} parameter(s) off-default "
                                     "(MODIFIED BUDGET).")

    # ---- validation (§3) ----------------------------------------------------
    # ---- night's target list ------------------------------------------------
