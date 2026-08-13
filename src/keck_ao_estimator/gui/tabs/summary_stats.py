"""Data-tab summary-stats panel: mean NGS(bright)/single-LGS/LTAO Strehl,
mean DIMM/MASS seeing, mean r0, and mean theta0, over a selectable period of
the night -- the same "observing window / whole night / specific time / time
of last pull" choice the field map's Conditions selector already offers
(PredictionTabMixin._when_time_from, shared rather than reimplemented).

NGS/single-LGS/LTAO are computed for ONE telescope at a time by the engine
(args.telescope), so K1 and K2 numbers can't both come from self.res -- the
NOT-currently-selected telescope's numbers come from a second, cheap
compute_timeline() call reusing the SAME prepared night (DIMM/MASS/masspro
parsing, and the resolved science wavelength prep.lam_nm, don't depend on
telescope -- see _other_telescope_res). DIMM/MASS/r0/theta0 ARE
telescope-independent, so those four always come straight from self.res.
"""
import copy

from qtcompat import Qt, QtWidgets

import keck_ao_estimator as engine

from ..constants import NIGHTTIME_FM_COND
from ..theme import set_cue
from ..widgets import TimeEdit


class SummaryStatsMixin:
    def _init_summary_stats(self, f):
        header = QtWidgets.QLabel("<b>Summary stats</b>")
        f.addRow(header)

        self.stats_cond = QtWidgets.QComboBox()
        self.stats_cond.addItems(["observing window", "whole night",
                                 "specific time", NIGHTTIME_FM_COND])
        # floors, not natural widths: this row otherwise sets the Data tab's
        # minimum and brings back the panel scrollbar (631045c). The combo
        # elides its current item; the button clips only under squeeze.
        self.stats_cond.setMinimumWidth(100)
        self.stats_cond.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.stats_time = TimeEdit()          # free-typing (see widgets.py)
        self.stats_time.setEnabled(False)
        self.stats_match_btn = QtWidgets.QPushButton("Match SR tool")
        self.stats_match_btn.setMinimumWidth(80)
        self.stats_match_btn.setToolTip(
            "Set the period to 'specific time' at the last frame measured "
            "in the Measured SR tab AND the science wavelength to that "
            "frame's EFFWAVE, so these stats describe the atmosphere of "
            "that exact image at the same wavelength")
        self.stats_match_btn.clicked.connect(
            lambda: self._nirc2_match_tool("stats"))
        period_row = QtWidgets.QHBoxLayout()
        period_row.addWidget(self.stats_cond, 1)
        period_row.addWidget(self.stats_time)
        period_row.addWidget(self.stats_match_btn)
        f.addRow("Period:", self._wrap(period_row))

        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        # all leftover width goes to an empty trailing column: otherwise the
        # LAST data column (K2) absorbs it, which drags the centre of the
        # common rows' col-1..2 span far right of the actual K1/K2 divide
        # (Eduardo's centreline screenshot, 2026-07-22). Equal minimum widths
        # keep that centre on the divide even when the value strings differ.
        grid.setColumnStretch(3, 1)
        grid.setColumnMinimumWidth(1, 95)
        grid.setColumnMinimumWidth(2, 95)
        # EVERYTHING in the two telescope columns is centred WITHIN its
        # column -- headers and values alike. That is what makes the common
        # rows' span-centring below land exactly between the two visible
        # columns: the centre of a col-1..2 span equals the midpoint of the
        # two column centres by construction, for ANY widths/fonts. (Two
        # earlier attempts left the values left-aligned in their columns,
        # which parks the span centre ~half a column right of where the eye
        # puts the K1/K2 midline -- Eduardo's red-pen screenshots, twice.)
        _CC = Qt.AlignmentFlag.AlignHCenter
        for col, tel in ((1, "K1"), (2, "K2")):
            lbl = QtWidgets.QLabel(f"<b>{tel}</b>")
            grid.addWidget(lbl, 0, col, _CC)
        self._stats_val = {}

        # caption labels are kept by key: the SR and theta0 captions carry
        # the LIVE science band (which the band combo / a TRICK dichroic
        # swap can change), refreshed in _refresh_summary_stats
        self._stats_caption = {}

        def _row(r, caption, keys, cap_key=None):
            cap = QtWidgets.QLabel(caption)
            # floor, not text width: the caption column otherwise sets the
            # whole Data tab's minimum and brings back the panel's
            # horizontal scrollbar (631045c -- never). Text clips only when
            # the panel is squeezed below ~350 px; centring is unaffected
            # (it lives in columns 1-2).
            cap.setMinimumWidth(110)
            grid.addWidget(cap, r, 0)
            if cap_key:
                self._stats_caption[cap_key] = cap
            for col, key in zip((1, 2), keys):
                lbl = QtWidgets.QLabel("—")
                set_cue(lbl, "secondary")
                grid.addWidget(lbl, r, col, _CC)
                if key is not None:
                    self._stats_val[key] = lbl

        # every row states its band/wavelength -- "quick glance" rule
        # (Eduardo 2026-07-23): SR and theta0 are at the SCIENCE band,
        # DIMM/MASS/r0/tau0 at the seeing monitors' 500 nm
        _row(1, "SR NGS (bright, K):", ("ngs_k1", "ngs_k2"), "ngs")
        _row(2, "SR LGS (single, K):", ("lgs_k1", "lgs_k2"), "lgs")
        # K2 cell intentionally left an unregistered, permanent "—": LTAO is
        # K1 hardware, there is no K2 value to ever fill in
        _row(3, "SR LTAO (K):", ("ltao_k1", None), "ltao")

        # blank spacer row: the rows above are PER-TELESCOPE (the K1/K2
        # columns), the rows below are common to both -- without a visual
        # break everything reads as one K1-headed table
        grid.setRowMinimumHeight(4, 10)

        for r, caption, key, cap_key in (
                (5, "DIMM seeing (500 nm):", "dimm", None),
                (6, "MASS seeing (500 nm):", "mass", None),
                (7, "r0 (500 nm):", "r0", None),
                (8, "theta0 (K):", "theta0", "theta0"),
                # "(forecast)": unlike every row above it -- which are
                # MEASUREMENTS (DIMM/MASS) -- tau0's wind leg comes from the
                # GFS model (or the hand-entered wind fields), so the value
                # is a forecast-derived estimate, not a measured quantity
                (9, "tau0 (forecast, 500 nm):", "tau0", None)):
            cap = QtWidgets.QLabel(caption)
            cap.setMinimumWidth(110)   # same floor as _row above
            grid.addWidget(cap, r, 0)
            if cap_key:
                self._stats_caption[cap_key] = cap
            lbl = QtWidgets.QLabel("—")
            set_cue(lbl, "secondary")
            # centred ACROSS the K1+K2 columns: these values belong to the
            # night, not to either telescope, so they sit on the divide
            # rather than lining up under K1
            grid.addWidget(lbl, r, 1, 1, 2,
                           Qt.AlignmentFlag.AlignHCenter)
            self._stats_val[key] = lbl

        # span BOTH form columns: parked in the field column the grid's
        # width stacked on top of the label column's ("Wind free-atm
        # (jet):") and their sum set the whole tab's minimum -- the panel
        # scrollbar again (631045c)
        f.addRow(self._wrap(grid))
        note = QtWidgets.QLabel(
            "Strehl means are per-telescope (each is its own engine run); "
            "LTAO is K1 hardware, so no K2 value. n/a while the TT sensor is "
            "TRICK (K1-only) -- comparing K2 at K1's forced dichroic band "
            "would be physically meaningless. tau0 is forecast-derived (GFS/"
            "entered winds), not measured like the seeing/r0/theta0 rows.")
        set_cue(note, "secondary")
        note.setWordWrap(True)
        note.setStyleSheet("QLabel { font-size:11px; }")
        f.addRow("", note)

        self.stats_cond.currentTextChanged.connect(
            lambda t: self.stats_time.setEnabled(t == "specific time"))
        self.stats_cond.currentTextChanged.connect(self._refresh_summary_stats)
        self.stats_time.timeChanged.connect(self._refresh_summary_stats)

    def _other_telescope_res(self, args, offsets):
        """compute_timeline() for the telescope NOT currently selected,
        reusing self.prep (cheap: compute_timeline is pure math on the
        already-parsed night, ~30ms). Returns None if there is no prepared
        night, or if the TT sensor is TRICK -- TRICK is K1-only hardware and
        forces a dichroic science-band swap baked into THIS prep's lam_nm,
        so a K2 number computed against it would be silently evaluated at
        the wrong band (K1's forced band, not K2's real one)."""
        if self.prep is None:
            return None
        if self.tt_sensor.currentText().startswith("TRICK"):
            return None
        other_args = copy.copy(args)
        other_args.telescope = "K1" if args.telescope == "K2" else "K2"
        # the NGS Gompertz-fit fields in args came from the NGS tab's
        # spinboxes, which hold the fit FOR THE CURRENTLY SELECTED telescope
        # (_ngs_fit_tel) -- carrying them over would evaluate the other
        # telescope's NGS Strehl with the wrong telescope's fit, making BOTH
        # columns shift whenever the Telescope radio flips (Eduardo's
        # 2026-07-22 report: NGS moved with the selection, LGS/LTAO -- which
        # don't use the fit -- didn't). The other telescope gets its OWN
        # engine-default fit; any user fit edit applies only to the
        # telescope it was edited under, same as everywhere else.
        par = engine.NGS_PARAMS[other_args.telescope]
        other_args.ngs_s0 = par["S0"]
        other_args.ngs_a = par["A"]
        other_args.ngs_m0 = par["m0"]
        other_args.ngs_w = par["w"]
        try:
            with engine.budget_overrides(**offsets):
                return engine.compute_timeline(other_args, self.prep)
        except Exception:
            return None

    def _science_band_label(self):
        """The live science band/wavelength for the caption annotations:
        the explicit nm override when active, else the band combo (which a
        TRICK dichroic swap updates too)."""
        if self.wl_enable.isChecked():
            return f"{self.wl_nm.value():g} nm"
        return self.band_combo.currentText()

    def _refresh_summary_stats(self, *_):
        if not hasattr(self, "_stats_val"):
            return
        vals = self._stats_val
        # SR/theta0 are at the SCIENCE band -- keep the captions telling
        # the truth when the band combo / wavelength override / TRICK swap
        # changes it
        band = self._science_band_label()
        self._stats_caption["ngs"].setText(f"SR NGS (bright, {band}):")
        self._stats_caption["lgs"].setText(f"SR LGS (single, {band}):")
        self._stats_caption["ltao"].setText(f"SR LTAO ({band}):")
        self._stats_caption["theta0"].setText(f"theta0 ({band}):")
        if self.res is None or self.prep is None or self.args_cached is None:
            for lbl in vals.values():
                lbl.setText("—")
            return

        when, t_hst = self._when_time_from(self.stats_cond, self.stats_time)
        cur_tel = self.args_cached.telescope
        res_other = self._other_telescope_res(self.args_cached,
                                              self.last_offsets or {})
        res_by_tel = {cur_tel: self.res}
        if res_other is not None:
            other_tel = "K1" if cur_tel == "K2" else "K2"
            res_by_tel[other_tel] = res_other

        def fmt(v, suffix, prec=2):
            return "—" if v is None else f"{v:.{prec}f}{suffix}"

        trick_active = self.tt_sensor.currentText().startswith("TRICK")
        for tel, res in (("K1", res_by_tel.get("K1")),
                         ("K2", res_by_tel.get("K2"))):
            tl = tel.lower()
            if res is None:
                # blank for the CURRENT telescope only ever means "not run
                # yet" (handled by the early-return above), so this branch
                # is always the OTHER telescope -- explain why on the cell
                # itself, not just in the footer note, since that's easy to
                # miss at a glance
                if tel != cur_tel and trick_active:
                    why = ("n/a — TRICK (K1-only) forces this run's science "
                           "band to its own dichroic swap; a same-footing "
                           f"{tel} number can't be computed against it")
                else:
                    why = f"n/a — {tel} estimate could not be computed"
                for key in (f"ngs_{tl}", f"lgs_{tl}"):
                    vals[key].setText("—")
                    vals[key].setToolTip(why)
                if tel == "K1":
                    vals["ltao_k1"].setText("—")
                    vals["ltao_k1"].setToolTip(why)
                continue
            sel_t = engine.time_selection_mask(res.times, when, t_hst, self.prep)
            sel_p = engine.time_selection_mask(res.p_times, when, t_hst, self.prep)
            vals[f"ngs_{tl}"].setText(
                fmt(engine.masked_mean(res.ngs_bright, sel_t), "", 3))
            vals[f"ngs_{tl}"].setToolTip("")
            vals[f"lgs_{tl}"].setText(
                fmt(engine.masked_mean(res.sr_single, sel_p), "", 3))
            vals[f"lgs_{tl}"].setToolTip("")
            if tel == "K1":
                vals["ltao_k1"].setText(
                    fmt(engine.masked_mean(res.sr_ltao, sel_p), "", 3))
                vals["ltao_k1"].setToolTip("")

        # telescope-independent: read straight off the CURRENT res regardless
        # of which telescope is selected
        sel_t = engine.time_selection_mask(self.res.times, when, t_hst, self.prep)
        sel_p = engine.time_selection_mask(self.res.p_times, when, t_hst, self.prep)
        mean_dimm = engine.masked_mean(self.res.col_dimm, sel_t)
        mean_mass = engine.masked_mean(self.res.col_mass, sel_p)
        # these three are the ZENITH monitor value, not corrected to the
        # target's line of sight (only the internal eps_tot_los used inside
        # the Strehl/FWHM math is airmass-corrected) -- easy to misread as
        # "what the target actually sees", so say so explicitly
        zenith_note = ("Zenith value from the seeing monitor, NOT corrected "
                       "for the target's airmass -- the Strehl/FWHM "
                       "estimates above already apply that correction "
                       "internally.")
        vals["dimm"].setText(fmt(mean_dimm, '"'))
        vals["dimm"].setToolTip(zenith_note)
        vals["mass"].setText(fmt(mean_mass, '"'))
        vals["mass"].setToolTip(zenith_note)
        vals["r0"].setText(fmt(engine.masked_mean(self.res.col_r0_cm, sel_t),
                               " cm", 1))
        vals["r0"].setToolTip(zenith_note)
        vals["theta0"].setText(
            fmt(engine.masked_mean(self.res.col_theta0, sel_p), '"'))
        # tau0 = 0.314 r0/V_eff (500 nm, zenith, like the r0 row): V_eff is
        # the same 5/3-moment ground/FA wind mix the bandwidth term uses,
        # from the CURRENT wind fields -- so "Fetch winds (GFS)" upgrades
        # this row along with the estimate itself
        t0 = engine.tau0_seconds(mean_dimm, mean_mass,
                                 float(self.wind_ground.value()),
                                 float(self.wind_free.value()))
        vals["tau0"].setText("—" if t0 is None else f"{t0 * 1e3:.1f} ms")
        vals["tau0"].setToolTip(
            "0.314·r0/V_eff at 500 nm, zenith; V_eff = Cn²-weighted "
            "ground/free-atm wind mix from the wind fields above "
            f"({self.wind_ground.value():g} / {self.wind_free.value():g} "
            "m/s — Fetch winds (GFS) fills those with the night's real "
            "values)")
