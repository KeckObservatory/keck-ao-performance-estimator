"""Target tab: the science-target airmass overlay toggle, the night's
saved-target list, RA/Dec entry + SIMBAD name resolution, proper motion,
observing windows, and the (target-independent) zenith-angle override.
"""
from datetime import datetime

from qtcompat import Qt, QtWidgets

import keck_ao_estimator as engine

from ..widgets import OffsetEntry, _dspin, _ON_AXIS_OFFSET_CFG, _shrinkable_label
from ..workers import ResolveWorker
from ..theme import set_cue


class TargetTabMixin:
    def _tab_target(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        self.target_enable = QtWidgets.QCheckBox("show target")
        self.target_enable.setToolTip(
            "Adds the target's airmass curve and observing windows to the "
            "timeline plot.")
        self.target_enable.setMinimumWidth(90)
        self.target_enable.setChecked(bool(self.defaults.show_target))
        self.target_enable.toggled.connect(self._on_target_toggle)
        f.addRow(self.target_enable)

        # night's target list: pick a saved target (fills the fields below);
        # Save adds/updates one from the fields; frames auto-add their target.
        self.target_select = QtWidgets.QComboBox()
        self.target_select.setToolTip(
            "Targets for tonight. Selecting one loads its coordinates; a loaded "
            "image adds its own target automatically.")
        self.target_select.currentIndexChanged.connect(self._on_target_selected)
        tbtns = QtWidgets.QHBoxLayout()
        tsave = QtWidgets.QPushButton("Save")
        tsave.setToolTip("save the name/RA/Dec below as a target for tonight")
        tdel = QtWidgets.QPushButton("Delete")
        tsave.clicked.connect(self._save_current_target)
        tdel.clicked.connect(self._delete_target)
        # Keck starlist import (StarlistPickerMixin): parse a standard
        # observer starlist file and pick the target from a pop-up table
        tlist = QtWidgets.QPushButton("Load starlist…")
        tlist.setToolTip(
            "Load a Keck-format starlist file (the list observers hand the "
            "OAs) and pick tonight's science target from it.")
        tlist.clicked.connect(self._load_starlist_clicked)
        tshow = QtWidgets.QPushButton("Show starlist")
        tshow.setToolTip(
            "Reopen the picker for the already-loaded starlist without "
            "re-reading the file (closing the dialog otherwise means "
            "browsing to the file again just to see the list).")
        tshow.clicked.connect(self._show_starlist_clicked)
        # explicit floors: a QPushButton's minimumSizeHint is its full text
        # width, and this four-button row would otherwise set the whole
        # tab's minimum (see 631045c -- no horizontal scrollbar, ever)
        for _b in (tsave, tdel, tlist, tshow):
            _b.setMinimumWidth(56)
        self._starlist_dialog = None
        self._starlist_table = None
        self._starlist_detail = None
        self._starlist_path = None
        self._starlist_fname = None
        self._starlist_raw_entries = []
        self._starlist_entries = []       # merged view (raw - excl + added)
        self._session_additions = []      # Save-button targets, this list
        self._session_exclusions = []     # Delete-button removals, this list
        # a starlist target with 2+ TT-star candidates but no prepared night
        # yet to rank them against: {"target","ra_deg","dec_deg","candidates"}
        # while waiting, else None. Resolved automatically after the next
        # successful run (see StarlistPickerMixin._resolve_pending_guide_star).
        self._pending_gs = None
        self._last_gs_mag = None
        self._pending_gs_mag = None     # async SIMBAD magnitude lookup state
        self._gs_mag_worker = None      # (see StarlistPickerMixin)
        tbtns.addWidget(tsave); tbtns.addWidget(tdel); tbtns.addWidget(tlist)
        tbtns.addWidget(tshow)
        tbtns.addStretch(1)
        f.addRow("Targets:", self.target_select)
        f.addRow(self._wrap(tbtns))          # span full width (no label offset)

        self.tname_edit = QtWidgets.QLineEdit(self.defaults.target_name or "")
        self.tname_edit.setToolTip(
            "Target name. Click Resolve to query SIMBAD for its RA/Dec and "
            "proper motion.")
        self.resolve_btn = QtWidgets.QPushButton("Resolve")
        self.resolve_btn.setToolTip(
            "Query SIMBAD for this name's RA/Dec and proper motion, and fill "
            "them in below (proper motion only if SIMBAD has a measured "
            "value for it).")
        self.resolve_btn.clicked.connect(self._resolve_target_name)
        name_row = QtWidgets.QHBoxLayout()
        name_row.addWidget(self.tname_edit, 1)
        name_row.addWidget(self.resolve_btn)
        f.addRow("Name:", self._wrap(name_row))

        self.ra_edit = QtWidgets.QLineEdit(self.defaults.ra or "")
        self.ra_edit.setPlaceholderText("e.g. 15h49m57.7s")
        self.ra_edit.setToolTip(
            "RA in any of: hms (15h49m57.7s), colon-separated hours "
            "(15:49:57.7), or decimal degrees (237.49).")
        self.dec_edit = QtWidgets.QLineEdit(self.defaults.dec or "")
        self.dec_edit.setPlaceholderText("e.g. -03d55m16s")
        self.dec_edit.setToolTip(
            "Dec in any of: dms (-03d55m16s), colon-separated degrees "
            "(-03:55:16), or decimal degrees (-3.92).")
        f.addRow("RA:", self.ra_edit)
        f.addRow("Dec:", self.dec_edit)

        # proper motion (mas/yr, SIMBAD/Gaia PMRA*cosDec convention): applied
        # in _effective_target_coords() to propagate the typed/resolved J2000
        # position forward to the observing date -- see that method and
        # engine.apply_proper_motion. 0/0 (the default) is a complete no-op,
        # so a target with no known PM is unaffected.
        # +-20000 mas/yr comfortably covers every known real proper motion
        # (Barnard's Star, the fastest known, is ~10328 mas/yr) with 2x
        # margin. Layout: a QDoubleSpinBox's default horizontal size policy is
        # Minimum, so its minimumSizeHint() (driven by the widest possible
        # displayed text, i.e. the range/decimals) is a hard floor the layout
        # cannot shrink below -- two such floors side by side in this one row
        # forced a horizontal scrollbar across the whole tab (Eduardo,
        # 2026-07-20). setMaximumWidth() does NOT lower that floor; an
        # explicit setMinimumWidth() DOES (it overrides minimumSizeHint in
        # Qt's layout math), letting the pair compress when the panel is
        # narrow while still growing to their comfortable width otherwise.
        self.pmra_spin = _dspin(-20000, 20000, 1.0,
                                getattr(self.defaults, "pm_ra", None) or 0.0,
                                1, " mas/yr")
        self.pmra_spin.setMinimumWidth(70)
        self.pmra_spin.setToolTip(
            "Proper motion in RA*cos(Dec) (SIMBAD/Gaia convention -- the "
            "sky-angle rate, not the RA-coordinate rate), mas/yr. Propagates "
            "the RA/Dec above from J2000 to the observing date. 0 = no "
            "correction.")
        self.pmdec_spin = _dspin(-20000, 20000, 1.0,
                                 getattr(self.defaults, "pm_dec", None) or 0.0,
                                 1, " mas/yr")
        self.pmdec_spin.setMinimumWidth(70)
        self.pmdec_spin.setToolTip(
            "Proper motion in Dec, mas/yr. 0 = no correction.")
        pm_row = QtWidgets.QHBoxLayout()
        _pm_ra_lbl = QtWidgets.QLabel("RA·cosDec:")
        _pm_ra_lbl.setMinimumWidth(50)   # floor, not text width (631045c)
        pm_row.addWidget(_pm_ra_lbl)
        pm_row.addWidget(self.pmra_spin)
        pm_row.addWidget(QtWidgets.QLabel("Dec:"))
        pm_row.addWidget(self.pmdec_spin)
        f.addRow("Proper motion:", self._wrap(pm_row))

        # persistent offset applied on TOP of the RA/Dec above (e.g. an offset
        # star, or a small dither) -- the base fields stay independently
        # editable/saveable; 0 offset (the default) uses them unchanged. This
        # is what actually reaches the engine (see _effective_target_coords),
        # not just a display convenience.
        self.target_offset = OffsetEntry(0.0, self._science_coords)
        self.target_offset.setToolTip(
            "Shift the RA/Dec above by this much before using them (an offset "
            "star, a small dither, ...). 0 = use RA/Dec exactly as entered.")
        f.addRow("Target offset:", self.target_offset)
        self.target_offset_readout = QtWidgets.QLabel()
        set_cue(self.target_offset_readout, "secondary")
        _shrinkable_label(self.target_offset_readout)
        f.addRow("", self.target_offset_readout)

        self.ra_edit.textChanged.connect(self._validate)
        self.dec_edit.textChanged.connect(self._validate)
        self.ra_edit.textChanged.connect(self._update_target_offset_readout)
        self.dec_edit.textChanged.connect(self._update_target_offset_readout)
        self.target_offset.changed.connect(self._validate)
        self.target_offset.changed.connect(self._update_target_offset_readout)
        self.target_offset.pos_changed.connect(self._update_target_offset_readout)
        # proper motion feeds the SAME effective-coords pipeline as RA/Dec
        # (see _effective_target_coords) -- wire it identically
        self.pmra_spin.valueChanged.connect(self._validate)
        self.pmdec_spin.valueChanged.connect(self._validate)
        self.pmra_spin.valueChanged.connect(self._update_target_offset_readout)
        self.pmdec_spin.valueChanged.connect(self._update_target_offset_readout)
        # whenever the EFFECTIVE target might have moved (base RA/Dec edited,
        # the proper motion changed, or the target offset itself changed), let
        # any "fix to base" TT/NGS/laser offset re-derive its displayed value
        # to keep its own absolute position fixed
        self.ra_edit.textChanged.connect(self._refresh_fixed_offsets)
        self.dec_edit.textChanged.connect(self._refresh_fixed_offsets)
        self.pmra_spin.valueChanged.connect(self._refresh_fixed_offsets)
        self.pmdec_spin.valueChanged.connect(self._refresh_fixed_offsets)
        self.target_offset.changed.connect(self._refresh_fixed_offsets)
        self.target_offset.pos_changed.connect(self._refresh_fixed_offsets)
        self._update_target_offset_readout()

        # windows: editable list of HH:MM-HH:MM
        self.windows_list = QtWidgets.QListWidget()
        self.windows_list.setMaximumHeight(110)
        for wtxt in (self.defaults.window or engine.DEF_WINDOWS):
            self._add_window_item(wtxt)
        win_btns = QtWidgets.QHBoxLayout()
        add_b = QtWidgets.QPushButton("Add")
        del_b = QtWidgets.QPushButton("Remove")
        add_b.clicked.connect(lambda: self._add_window_item("00:00-01:00", edit=True))
        del_b.clicked.connect(self._remove_window_item)
        win_btns.addWidget(add_b)
        win_btns.addWidget(del_b)
        self.windows_list.itemChanged.connect(self._validate)
        # label object kept: UTC mode (data.py _on_utc_toggled) swaps it to
        # "Windows (UT):" and converts the list entries in place
        self.windows_row_label = QtWidgets.QLabel("Windows (HST):")
        f.addRow(self.windows_row_label, self.windows_list)
        f.addRow("", self._wrap(win_btns))

        # Zenith angle is INDEPENDENT of the target overlay: with no target it
        # projects the whole night onto a fixed line of sight; with a target it
        # applies outside the observing window(s) (the target's own airmass
        # drives ZA inside them). So it is NOT gated by the target checkbox.
        self.za_enable = QtWidgets.QCheckBox("override zenith angle")
        self.za_enable.setMinimumWidth(110)   # floor, not text width (631045c)
        self.za_enable.setToolTip(
            "Independent of the target overlay: with no target it projects "
            "the whole night onto this fixed line of sight; with a target it "
            "applies outside the observing window(s).")
        self.za_spin = _dspin(0, 85, 1, self.defaults.zenith_angle, 1, "°")
        self.za_spin.setEnabled(False)
        self.za_enable.toggled.connect(self.za_spin.setEnabled)
        za_row = QtWidgets.QHBoxLayout()
        za_row.addWidget(self.za_enable)
        za_row.addWidget(self.za_spin)
        # single-line label: the two-line "(independent of target)" variant
        # was the widest LABEL in this form, and a QFormLayout's minimum width
        # is max(label column) + max(field column) -- it cost ~120px of
        # horizontal budget for a parenthetical (now the checkbox tooltip)
        f.addRow("Zenith angle:", self._wrap(za_row))

        # zenith + target overlay both feed prepare_night -> live re-prepare
        self.za_enable.toggled.connect(self._on_prep_changed)
        self.za_spin.valueChanged.connect(self._on_prep_changed)
        self.target_enable.toggled.connect(self._on_prep_changed)

        # target-gated widgets: name/RA/Dec/offset/windows (NOT the zenith row)
        self._target_widgets = [self.tname_edit, self.resolve_btn, self.ra_edit,
                                 self.dec_edit, self.pmra_spin, self.pmdec_spin,
                                 self.target_offset, self.windows_list, add_b,
                                 del_b, self.target_select, tsave, tdel]
        self._on_target_toggle()
        # seed the list with the default target if it has coordinates
        if (self.defaults.ra or "").strip() and (self.defaults.dec or "").strip():
            self._add_target(self.defaults.target_name or "",
                             self.defaults.ra, self.defaults.dec,
                             self.pmra_spin.value(), self.pmdec_spin.value(),
                             select=True)
        else:
            self._refresh_target_combo()
        return self._scroll(w)

    def _add_window_item(self, text, edit=False):
        it = QtWidgets.QListWidgetItem(text)
        it.setFlags(it.flags() | Qt.ItemFlag.ItemIsEditable)
        self.windows_list.addItem(it)
        if edit:
            self.windows_list.editItem(it)

    def _remove_window_item(self):
        for it in self.windows_list.selectedItems():
            self.windows_list.takeItem(self.windows_list.row(it))
        self._validate()

    # ---- reactive handlers --------------------------------------------------
    def _on_target_toggle(self):
        on = self.target_enable.isChecked()
        for wdg in getattr(self, "_target_widgets", []):
            wdg.setEnabled(on)
        self._validate()

    def _refresh_target_combo(self, select=None):
        """Rebuild the target dropdown from self._targets. select: index to
        show (default: keep current)."""
        cur = self.target_select.currentIndex() if select is None else select
        self.target_select.blockSignals(True)
        self.target_select.clear()
        self.target_select.addItems(
            [t["name"] or f"{t['ra']} {t['dec']}" for t in self._targets]
            or ["(no targets)"])
        if self._targets:
            self.target_select.setCurrentIndex(
                min(max(cur, 0), len(self._targets) - 1))
        self.target_select.blockSignals(False)

    def _on_target_selected(self, idx):
        """Load the chosen target's coordinates, proper motion, AND guide
        star into the fields. PM and the guide star (TT-star position/mag)
        must be restored here too (not just RA/Dec) -- otherwise switching
        targets would silently leak the PREVIOUS target's proper motion or
        guide star onto the newly-selected one. .get(...) defaults cover
        targets saved before each feature existed.

        Guide-star rule (2026-07-22): every target has its OWN TT-star
        position (tt_offset_cfg), defaulting to on-axis ("guide star = the
        target") unless something more specific was set for it -- e.g. a
        starlist-derived real TT star (see starlist_picker.py). tt_mag is
        only touched when the entry actually carries one (a real star's
        known magnitude); otherwise it's left alone -- there's no natural
        "on-axis" magnitude to reset it to. NGS has NO exception (the tool
        always assumes the NGS guide star IS the target), so ngs_offset is
        unconditionally forced on-axis here, not read from the entry."""
        if self._loading or not (0 <= idx < len(self._targets)):
            return
        # capture whatever offset is LIVE on the target we're switching
        # AWAY FROM, onto its own saved entry, before the fields below get
        # overwritten with the new target's -- otherwise a small offset
        # set by hand (nobody clicked Save; this fires on every switch,
        # not just a manual one -- an unattended batch run auto-switches
        # targets as different frames come up) is silently discarded the
        # moment ANY other target is selected, and is gone for good once
        # this target comes up again later (Eduardo 2026-07-28: "there
        # are times when small offsets were done and it reverted back to
        # the default values"). Deliberately does NOT touch tt_mag: that
        # already has its own evidence-ranked resolver
        # (nirc2_strehl.py's _nirc2_resolve_guide_star) and baking a
        # possibly-stale spinbox reading into the saved entry would
        # re-introduce the exact contamination that resolver avoids.
        tt_offset = getattr(self, "tt_offset", None)
        if tt_offset is not None:
            prev_name = self.tname_edit.text().strip()
            if prev_name:
                for t in self._targets:
                    if engine.same_star_name(t.get("name", ""), prev_name):
                        t["tt_offset_cfg"] = tt_offset.get_config()
                        break
        t = self._targets[idx]
        self.tname_edit.setText(t["name"])
        self.ra_edit.setText(t["ra"])
        self.dec_edit.setText(t["dec"])
        self.pmra_spin.setValue(t.get("pm_ra", 0.0))
        self.pmdec_spin.setValue(t.get("pm_dec", 0.0))
        # defensive (getattr): the Target tab builds before NGS/LGS, so
        # these may not exist yet the first time this fires (the startup
        # default-target seed, during THIS tab's own construction)
        tt_offset = getattr(self, "tt_offset", None)
        if tt_offset is not None:
            tt_offset.set_config(t.get("tt_offset_cfg") or
                                 dict(_ON_AXIS_OFFSET_CFG))
            if t.get("tt_mag") is not None:
                self.tt_mag.setValue(t["tt_mag"])
        ngs_offset = getattr(self, "ngs_offset", None)
        if ngs_offset is not None:
            ngs_offset.set_config(dict(_ON_AXIS_OFFSET_CFG))
        self._validate()

    def _add_target(self, name, ra, dec, pm_ra=0.0, pm_dec=0.0,
                    tt_offset_cfg=None, tt_mag=None, select=True):
        """Add or update a target (matched by name if named, else by
        coords). tt_offset_cfg/tt_mag: this target's OWN guide star (see
        _on_target_selected) -- None means "don't change it": a genuinely
        NEW entry gets the on-axis default there, an EXISTING one keeps
        whatever it already had. Only _save_current_target (captures the
        LIVE tt_offset/tt_mag, like it already does for pm_ra/pm_dec) and
        the starlist picker (real guide-star candidate data) pass one
        explicitly. Returns its index."""
        key = (name.strip().lower() if name.strip()
               else f"{ra}|{dec}".lower())
        idx = None
        for i, t in enumerate(self._targets):
            tkey = (t["name"].strip().lower() if t["name"].strip()
                    else f"{t['ra']}|{t['dec']}".lower())
            if tkey == key:
                idx = i
                break
        if idx is not None:
            prev = self._targets[idx]
            cfg = tt_offset_cfg if tt_offset_cfg is not None \
                else prev.get("tt_offset_cfg")
            mag = tt_mag if tt_mag is not None else prev.get("tt_mag")
        else:
            cfg, mag = tt_offset_cfg, tt_mag
        entry = {"name": name, "ra": ra, "dec": dec,
                "pm_ra": pm_ra, "pm_dec": pm_dec,
                "tt_offset_cfg": cfg, "tt_mag": mag}
        if idx is None:
            self._targets.append(entry)
            idx = len(self._targets) - 1
        else:
            self._targets[idx] = entry
        self._refresh_target_combo(idx if select else None)
        if select:
            self._on_target_selected(idx)       # load it into the fields
        return idx

    def _save_current_target(self):
        ra, dec = self.ra_edit.text().strip(), self.dec_edit.text().strip()
        if not ra or not dec:
            self.status.setText("Enter RA/Dec before saving a target.")
            return None
        name = self.tname_edit.text().strip()
        idx = self._add_target(name, ra, dec,
                               self.pmra_spin.value(), self.pmdec_spin.value(),
                               tt_offset_cfg=self.tt_offset.get_config(),
                               tt_mag=self.tt_mag.value(), select=True)
        # if a starlist is loaded, mirror the save into it (a session
        # addition, sidecar-persisted -- the real .lst file is never
        # touched); a no-op with no starlist loaded
        self._starlist_add_session_target(name, ra, dec)
        return idx

    def _delete_target(self):
        i = self.target_select.currentIndex()
        if 0 <= i < len(self._targets):
            name = self._targets[i]["name"]
            del self._targets[i]
            self._refresh_target_combo(min(i, len(self._targets) - 1))
            # mirror into the loaded starlist's session layer, if any (see
            # _save_current_target) -- a no-op for a target that was never
            # part of any loaded starlist
            self._starlist_remove_session_target(name)

    # ---- SIMBAD name resolution ---------------------------------------------
    def _resolve_target_name(self):
        """Resolve button: query SIMBAD for the typed name off the GUI thread
        (ResolveWorker). Does NOT overwrite the typed name with SIMBAD's
        canonical identifier -- only RA/Dec/PM are filled in -- so a
        deliberately-chosen nickname the user typed isn't silently renamed."""
        name = self.tname_edit.text().strip()
        if not name:
            self.status.setText("Type a target name to resolve.")
            return
        self.resolve_btn.setEnabled(False)
        self.status.setText(f"Resolving “{name}”…")
        self._resolve_worker = ResolveWorker(name, self)
        self._resolve_worker.done.connect(self._on_resolved)
        self._resolve_worker.start()

    def _on_resolved(self, result, err):
        self.resolve_btn.setEnabled(True)
        if result is None:
            self.status.setText(f"Resolve failed: {err}")
            return
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        c = SkyCoord(result["ra_deg"] * u.deg, result["dec_deg"] * u.deg)
        self.ra_edit.setText(
            c.ra.to_string(unit="hourangle", sep="hms", precision=3))
        self.dec_edit.setText(
            c.dec.to_string(unit="deg", sep="dms", precision=2, alwayssign=True))
        pmra, pmdec = result["pmra"], result["pmdec"]
        if pmra is not None:
            self.pmra_spin.setValue(pmra)
        if pmdec is not None:
            self.pmdec_spin.setValue(pmdec)
        pm_note = ("no proper motion in SIMBAD" if pmra is None and pmdec is None
                  else f"PM {pmra or 0:+.1f}, {pmdec or 0:+.1f} mas/yr")
        # a newly-resolved star has no known guide star of its own: default
        # to on-axis (this tool's rule -- NGS/LGS guide star is the target
        # unless a loaded starlist says otherwise) rather than silently
        # carrying over whatever guide star was dialed in for the PREVIOUS
        # star. Defensive getattr: see _on_target_selected's own note.
        tt_offset = getattr(self, "tt_offset", None)
        if tt_offset is not None:
            tt_offset.set_config(dict(_ON_AXIS_OFFSET_CFG))
        ngs_offset = getattr(self, "ngs_offset", None)
        if ngs_offset is not None:
            ngs_offset.set_config(dict(_ON_AXIS_OFFSET_CFG))
        # ...and as its own guide star it gets its OWN magnitudes: SIMBAD's
        # fluxes fill tt_mag + ngs_bright, or an explicit warning says they
        # were NOT updated (a stale magnitude left silently in place looks
        # deliberate). No async fallback here -- this IS the SIMBAD result.
        mag_note = ""
        if tt_offset is not None:
            mags = result.get("mags") or {}
            band = self._tt_sensor_band()
            mag, kind, _lb = engine.estimate_sensing_mag(mags, band)
            nmag, nkind, _nl = engine.estimate_sensing_mag(mags, "R")
            bits, failed = [], []
            if mag is not None:
                self._tt_mag_auto = True
                self.tt_mag.setValue(mag)
                self._tt_mag_auto = False
                bits.append(f"TT {band}{'=' if kind == 'exact' else '≈'}{mag:.1f}")
            else:
                failed.append("TT-star magnitude NOT updated")
            if nmag is not None:
                self.ngs_bright.setValue(nmag)
                bits.append(f"NGS R{'=' if nkind == 'exact' else '≈'}{nmag:.1f}")
            else:
                failed.append("NGS magnitude NOT adjusted")
            if bits:
                mag_note = f"  ({', '.join(bits)})"
            if failed:
                mag_note += f"  ⚠ {'; '.join(failed)} — set manually"
        # ra_edit/dec_edit/pmra_spin/pmdec_spin's own signals already ran
        # _validate() as a side effect of the setText/setValue calls above --
        # an explicit call here would win the race and clobber this message
        # with _validate()'s own "ready to run" / "Cannot run" status text.
        self.status.setText(
            f"Resolved “{self.tname_edit.text().strip()}” → {result['name']}: "
            f"{self.ra_edit.text()}  {self.dec_edit.text()}  ({pm_note})"
            f"{mag_note}")

    def _science_coords(self):
        """(RA, Dec) strings of the BASE science target (the typed RA/Dec
        fields, before the target-offset control). Used as the reference
        point for the target-offset control ITSELF (which must not depend on
        its own resolved output) -- everything else that means "the science
        target" wants _effective_science_coords instead."""
        return (self.ra_edit.text(), self.dec_edit.text())

    def _effective_target_coords(self):
        """(ra, dec) strings for the ACTUAL target position used by the engine
        and the field map: the RA/Dec fields above, propagated by proper
        motion (see engine.apply_proper_motion; 0/0 is a no-op) from J2000 to
        the observing date, then shifted by target_offset (0 offset ->
        unchanged, the overwhelmingly common case). None if the base RA/Dec,
        or (star-coordinate mode) the offset itself, is missing or
        unparseable -- callers already have a "can this run?" check in
        target_offset.ok(); this mirrors it rather than raising."""
        ra, dec = self._science_coords()
        if not (ra.strip() and dec.strip()):
            return None
        try:
            base = engine.parse_radec(ra, dec)
            base = engine.apply_proper_motion(
                base, self.pmra_spin.value(), self.pmdec_spin.value(),
                self._pm_obs_date())
            resolved = self.target_offset.resolved_skycoord(base)
        except Exception:
            return None
        return (str(resolved.ra.to_string(unit="hourangle", sep="hms", precision=3)),
                str(resolved.dec.to_string(unit="deg", sep="dms", precision=2,
                                           alwayssign=True)))

    @staticmethod
    def _pm_obs_date():
        """Best-effort date to propagate proper motion TO (see
        engine.apply_proper_motion; reference epoch is J2000). Proper motion
        accumulates over YEARS (arcsec/decade even for a fast-moving target),
        so being off by days-to-months here -- which is all any reasonable
        choice could differ by -- has no practical effect; today's real date
        is used uniformly rather than trying to key off the loaded night
        (which may not even be known yet at validation time, before a run).
        Known limitation: retrospectively analyzing an OLD archived night for
        a target with a large proper motion will propagate slightly past that
        night to today, not to the night itself -- negligible unless both the
        archive is old AND the proper motion is large."""
        return datetime.now().date()

    def _effective_science_coords(self):
        """(ra, dec) strings of the EFFECTIVE science target (base RA/Dec
        shifted by the target-offset control), for the NGS/TT-star offset
        entries' star-coordinate mode -- so "how far is my guide star from
        the science target" measures from where the target actually is, not
        the pre-offset base. ("", "") on failure, matching _science_coords'
        always-a-string-pair contract (OffsetEntry calls .strip() on both
        unconditionally)."""
        eff = self._effective_target_coords()
        return eff if eff is not None else ("", "")

    def _update_target_offset_readout(self, *_):
        if not hasattr(self, "target_offset_readout"):
            return
        c = self._effective_target_coords()
        if c is None or not self.target_offset.ok():
            self.target_offset_readout.setText("")
            return
        ra_s, dec_s = c
        self.target_offset_readout.setText(f"→ effective target: {ra_s}  {dec_s}")

    def _refresh_fixed_offsets(self, *_):
        """Tell every "fix to base" TT/NGS/laser offset control that the
        effective target may have moved, so it can re-derive its displayed
        value and keep its own anchored absolute position. Defensive
        (getattr): the Target tab builds before NGS/LGS, so these may not
        exist yet the first time this fires during construction."""
        ngs_offset = getattr(self, "ngs_offset", None)
        if ngs_offset is not None:
            ngs_offset.refresh_from_base()
        tt_offset = getattr(self, "tt_offset", None)
        if tt_offset is not None:
            tt_offset.refresh_from_base()
        laser_refresh = getattr(self, "_laser_refresh_from_base", None)
        if laser_refresh is not None:
            laser_refresh()

    @staticmethod
    def _radec_ok(ra, dec):
        if not ra.strip() or not dec.strip():
            return False
        try:
            engine.parse_radec(ra, dec)
            return True
        except Exception:
            return False

    @staticmethod
    def _window_ok(text):
        try:
            a, b = text.split("-", 1)
            for hhmm in (a, b):
                h, m = (int(x) for x in hhmm.split(":"))
                if not (0 <= h < 24 and 0 <= m < 60):
                    return False
            return True
        except Exception:
            return False

    # ---- the single control-to-flag mapping (§3) ----------------------------
