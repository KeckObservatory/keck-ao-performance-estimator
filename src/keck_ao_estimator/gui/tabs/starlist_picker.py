"""Keck starlist picker: load a Keck-format starlist file (engine
starlist.parse_starlist) and pick the science target from a pop-up table,
following the guide-star ranking dialog's form (fieldmap_overlays.py):
non-modal so the rest of the GUI stays usable, double-click selects, and the
finished signal drops every Qt reference so no close path can leave a
dangling table.

Guide-star rule (2026-07-22): NGS/LGS normally assume the guide star IS the
science target (on-axis) -- the exception is a starlist target=<name>-linked
TT-star candidate for LGS, which is what a real starlist provides this data
for in the first place. A single candidate is used directly; 2+ are ranked
by delivered Strehl at the target using the SAME engine.rank_guide_stars the
field map's Rank button uses, and the top-ranked one wins (see
_rank_gs_candidates). Picking with 2+ candidates before any Run has happened
(no prepared night to evaluate delivered Strehl against) defers the ranking
-- _resolve_pending_guide_star runs it automatically once a result exists.
"""
import datetime
import math
import os

from qtcompat import Qt, QtCore, QtWidgets

import keck_ao_estimator as engine

from ..theme import set_cue
from ..widgets import SortableItem, TimeEdit, _ON_AXIS_OFFSET_CFG
from ..workers import ResolveWorker


def _entry_from_target(name, ra, dec):
    """Build a starlist-entry dict (same shape parse_starlist_text produces)
    for a target typed/saved in the Target tab -- so it can be appended to
    the session starlist-additions list and round-tripped through
    engine.write_starlist/parse_starlist exactly like a hand-written entry.
    Raises on unparseable ra/dec (callers already validate before saving)."""
    c = engine.parse_radec(ra, dec)
    ra_s = c.ra.to_string(unit="hourangle", sep=":", precision=2, pad=True)
    dec_s = c.dec.to_string(unit="deg", sep=":", precision=1, pad=True,
                            alwayssign=True)
    return dict(name=name, ra=ra_s, dec=dec_s,
               ra_deg=float(c.ra.deg), dec_deg=float(c.dec.deg),
               equinox="2000.0", keys={}, notes="", lgs=False, target=None,
               lineno=0)


def _starlist_entry_mags(e):
    """A starlist entry's magnitudes in the {band: value_or_None} shape
    engine.estimate_sensing_mag/pick_mag expect. Only the bands a Keck
    starlist actually carries (rmag=/vmag=/kmag=or K=/bare J=/H=) -- no
    Gaia/PanSTARRS/GSC fields exist in this format, so estimate_sensing_mag's
    colour-transform paths simply won't fire for these, falling through to
    its nearest-available-band fallback when R/H/K isn't given directly."""
    return {"V": engine.entry_float(e, "vmag"), "R": engine.entry_float(e, "rmag"),
           "K": engine.entry_float(e, "kmag", "K"),
           "J": engine.entry_float(e, "J"), "H": engine.entry_float(e, "H")}


class StarlistPickerMixin:
    def _load_starlist_clicked(self):
        start = getattr(self, "_starlist_dir", "") or ""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Keck starlist", start,
            "Keck starlist (*.lst *.txt *.list);;All files (*)")
        if path:
            self._starlist_dir = os.path.dirname(path)
            self._open_starlist(path)

    def _open_starlist(self, path):
        """Parse `path` and show the picker. Split from the file dialog so
        the regression suite (and a scripted user) can drive it directly."""
        try:
            entries, skipped = engine.parse_starlist(path)
        except OSError as e:
            self.status.setText(f"Starlist load failed: {e}")
            return
        if not entries:
            self.status.setText(
                f"No parseable starlist entries in {os.path.basename(path)}"
                + (f" ({len(skipped)} malformed lines)" if skipped else ""))
            return
        self._starlist_path = path
        self._starlist_fname = os.path.basename(path)
        self._starlist_raw_entries = entries
        self._session_additions, self._session_exclusions = \
            self._load_session_sidecars(path)
        self._refresh_starlist_merged()
        msg = (f"Starlist: {len(self._starlist_entries)} entries from "
              f"{self._starlist_fname}")
        if self._session_additions or self._session_exclusions:
            msg += (f"  ({len(self._session_additions)} session-added, "
                    f"{len(self._session_exclusions)} session-removed -- "
                    f"the loaded file itself is never modified)")
        if skipped:
            msg += (f" ({len(skipped)} malformed lines skipped, first at line "
                    f"{skipped[0][0]}: {skipped[0][2]})")
        self.status.setText(msg)
        self._show_starlist_dialog(self._starlist_fname)

    def _show_starlist_clicked(self):
        """Reopen the picker for the already-loaded starlist WITHOUT
        re-parsing the file from disk (Eduardo: closing the dialog otherwise
        means re-browsing the file just to see the list again)."""
        if not getattr(self, "_starlist_path", None):
            self.status.setText(
                "No starlist loaded yet -- use Load starlist... first.")
            return
        self._show_starlist_dialog(self._starlist_fname)

    def _session_sidecar_paths(self, path):
        """(additions_path, exclusions_path): sidecar files alongside the
        loaded starlist `path` that carry session Save/Delete actions,
        without ever writing to the real file. additions is Keck-starlist-
        format text (round-trips through engine.parse_starlist); exclusions
        is one target name per line."""
        stem, _ext = os.path.splitext(path)
        return stem + ".session.lst", stem + ".session_excluded.txt"

    def _load_session_sidecars(self, path):
        add_path, excl_path = self._session_sidecar_paths(path)
        additions = []
        if os.path.exists(add_path):
            try:
                additions, _skipped = engine.parse_starlist(add_path)
            except OSError:
                additions = []
        exclusions = []
        if os.path.exists(excl_path):
            try:
                with open(excl_path, encoding="utf-8") as fh:
                    exclusions = [ln.strip() for ln in fh if ln.strip()]
            except OSError:
                exclusions = []
        return additions, exclusions

    def _save_session_additions(self):
        add_path, _excl_path = self._session_sidecar_paths(self._starlist_path)
        try:
            engine.write_starlist(add_path, self._session_additions)
        except OSError as e:
            self.status.setText(
                f"Could not save session starlist additions: {e}")

    def _save_session_exclusions(self):
        _add_path, excl_path = self._session_sidecar_paths(self._starlist_path)
        try:
            with open(excl_path, "w", encoding="utf-8") as fh:
                for name in self._session_exclusions:
                    fh.write(name + "\n")
        except OSError as e:
            self.status.setText(
                f"Could not save session starlist exclusions: {e}")

    def _refresh_starlist_merged(self):
        """Recompute self._starlist_entries -- the MERGED view the picker
        dialog, guide-star ranking, etc. all read -- from the raw loaded
        file, minus session exclusions, plus session additions. Call after
        any change to _session_additions/_session_exclusions."""
        excl = self._session_exclusions
        base = [e for e in self._starlist_raw_entries
               if not any(engine.same_star_name(e["name"], x) for x in excl)]
        self._starlist_entries = base + self._session_additions

    def _starlist_add_session_target(self, name, ra, dec):
        """Save-button hook (target.py): if a starlist is loaded and `name`
        isn't already in it (raw file or a prior session addition), append
        it as a session addition (persisted to the sidecar) and drop any
        matching session exclusion (re-adding after a delete). No-op
        (silently) with no starlist loaded -- session additions are
        meaningless without a list to merge into."""
        if not getattr(self, "_starlist_path", None) or not name.strip():
            return
        already = any(engine.same_star_name(e["name"], name)
                     for e in self._starlist_raw_entries + self._session_additions)
        if already:
            return
        try:
            entry = _entry_from_target(name, ra, dec)
        except Exception:
            return
        self._session_additions.append(entry)
        self._session_exclusions = [x for x in self._session_exclusions
                                    if not engine.same_star_name(x, name)]
        self._save_session_additions()
        self._save_session_exclusions()
        self._refresh_starlist_merged()
        self._starlist_maybe_refresh_dialog()

    def _starlist_remove_session_target(self, name):
        """Delete-button hook (target.py): if `name` came from the loaded
        starlist file, mark it session-excluded; if it came from a prior
        session addition, drop it from there instead. No-op if `name` isn't
        part of any loaded starlist (an ordinary session-only target)."""
        if not getattr(self, "_starlist_path", None) or not name.strip():
            return
        was_addition = False
        kept = []
        for e in self._session_additions:
            if engine.same_star_name(e["name"], name):
                was_addition = True
            else:
                kept.append(e)
        if was_addition:
            self._session_additions = kept
            self._save_session_additions()
        elif any(engine.same_star_name(e["name"], name)
                for e in self._starlist_raw_entries):
            if not any(engine.same_star_name(name, x)
                      for x in self._session_exclusions):
                self._session_exclusions.append(name)
                self._save_session_exclusions()
        else:
            return
        self._refresh_starlist_merged()
        self._starlist_maybe_refresh_dialog()

    def _starlist_maybe_refresh_dialog(self):
        if getattr(self, "_starlist_dialog", None) is not None:
            self._show_starlist_dialog(self._starlist_fname)

    def _show_starlist_dialog(self, fname):
        entries = self._starlist_entries
        if self._starlist_dialog is not None:
            self._starlist_dialog.close()
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Keck starlist — {fname}")
        dlg.resize(820, 620)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(QtWidgets.QLabel(
            "Click a row for details (az/el, Moon separation, evaluated at "
            "the time below). Double-click to make it the science target "
            "(added to tonight's Targets list). Click a column header to "
            "sort."))

        # Internally self._starlist_eval_dt is always HST (matching the
        # app-wide convention -- see data.py's _fmt_hm: "All internal
        # datetimes stay HST; only labels convert"). Snapshot the UTC-mode
        # toggle ONCE per dialog build: the label and the two widgets below
        # display/accept whichever mode was active when the dialog opened,
        # so what "10:00" means never silently changes underneath a value
        # already typed in.
        is_utc = self._utc()
        if getattr(self, "_starlist_eval_dt", None) is None:
            self._starlist_eval_dt = self._starlist_now_hst()

        time_row = QtWidgets.QHBoxLayout()
        time_row.addWidget(QtWidgets.QLabel(
            f"Evaluate at ({'UT' if is_utc else 'HST'}):"))
        date_edit = QtWidgets.QDateEdit()
        date_edit.setCalendarPopup(True)
        date_edit.setDisplayFormat("yyyy-MM-dd")
        date_edit.setToolTip(
            f"Date ({'UT' if is_utc else 'HST'}) az/el and Moon separation "
            "below are computed at. Defaults to today; change it to check "
            "a target at a different date/time.")
        time_edit = TimeEdit()
        time_edit.setToolTip(
            f"Time of day ({'UT' if is_utc else 'HST'}) -- type e.g. "
            '"21:35", "2135", or a bare hour "9"; commits on Enter/'
            "focus-out.")
        now_btn = QtWidgets.QPushButton("Now")
        now_btn.setToolTip("Reset to the current time.")
        # otherwise Enter/Return pressed in EITHER the date or the time
        # field (neither consumes it) activates whichever button Qt picks
        # as the dialog's implicit default -- here, the first one added --
        # silently resetting the just-typed time back to "now" (Eduardo
        # 2026-07-28)
        now_btn.setAutoDefault(False)
        now_btn.setDefault(False)

        def _set_display(dt):
            date_edit.setDate(QtCore.QDate(dt.year, dt.month, dt.day))
            time_edit.setTime(QtCore.QTime(dt.hour, dt.minute))

        def _on_display_changed(*_):
            qd = date_edit.date()
            qt = time_edit.time()
            disp = datetime.datetime(qd.year(), qd.month(), qd.day(),
                                     qt.hour(), qt.minute())
            self._starlist_eval_dt = (
                disp - datetime.timedelta(hours=engine.HST_TO_UTC_HOURS)
                if is_utc else disp)
            self._starlist_refresh_detail()
            self._starlist_refresh_ha_column()

        def _reset_now():
            _set_display(self._starlist_dt_for_display(
                self._starlist_now_hst(), is_utc))
        _set_display(self._starlist_dt_for_display(
            self._starlist_eval_dt, is_utc))
        date_edit.dateChanged.connect(_on_display_changed)
        time_edit.timeChanged.connect(_on_display_changed)
        now_btn.clicked.connect(_reset_now)
        time_row.addWidget(date_edit)
        time_row.addWidget(time_edit)
        time_row.addWidget(now_btn)
        time_row.addStretch(1)
        lay.addLayout(time_row)
        self._starlist_time_edit = time_edit
        self._starlist_date_edit = date_edit
        # remembered so _starlist_show_detail's result line reports in the
        # SAME units this dialog's input widgets are using, rather than
        # silently converting to HST regardless of what was typed
        self._starlist_dialog_is_utc = is_utc
        cols = ["#", "name", "RA", "Dec", "HA", "V", "R", "K", "sep (″)",
               "role"]
        self._starlist_ha_col = cols.index("HA")
        table = QtWidgets.QTableWidget(len(entries), len(cols))
        table.setHorizontalHeaderLabels(cols)
        table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        session_added = {id(e) for e in self._session_additions}
        # HA (hour angle) depends only on RA + time (not Dec/latitude), so
        # it's computed once per entry at the dialog's "Evaluate at" time --
        # refreshed in place (_starlist_refresh_ha_column) when that time
        # changes, without rebuilding the whole table/dialog
        ha_when_utc = self._starlist_hst_to_utc(self._starlist_eval_dt)
        for i, e in enumerate(entries):
            vmag = engine.entry_float(e, "vmag")
            rmag = engine.entry_float(e, "rmag")
            kmag = engine.entry_float(e, "kmag", "K")
            sep = engine.entry_float(e, "sep")
            role = (f"TT star → {e['target']}" if e["target"]
                    else ("target" + (" (lgs)" if e["lgs"] else "")))
            if id(e) in session_added:
                role += "  [session-added]"
            if e["equinox"] not in ("2000", "2000.0"):
                role += f" ⚠ equinox {e['equinox']}"

            def _mag(v):
                return SortableItem("—" if v is None else f"{v:.1f}", v)
            try:
                ha = engine.hour_angle_hours(e["ra"], ha_when_utc)
                ha_item = SortableItem(f"{ha:+.2f}h", ha)
            except Exception:
                ha_item = SortableItem("—", None)
            # the '#' item carries the ENTRY index (UserRole): once the user
            # sorts by a column, the visual row no longer equals the entry
            # index, so _starlist_pick must never index entries by row
            num = SortableItem(str(e["lineno"]), e["lineno"])
            num.setData(Qt.ItemDataRole.UserRole, i)
            for j, item in enumerate((
                    num, QtWidgets.QTableWidgetItem(e["name"]),
                    SortableItem(e["ra"], e["ra_deg"]),
                    SortableItem(e["dec"], e["dec_deg"]),
                    ha_item,
                    _mag(vmag), _mag(rmag), _mag(kmag), _mag(sep),
                    QtWidgets.QTableWidgetItem(role))):
                table.setItem(i, j, item)
        table.resizeColumnsToContents()
        # only AFTER populating: with sorting live, every setItem could
        # re-order the rows out from under the population loop. Enabling
        # applies Qt's default indicator (column 0 DESCENDING, reversing the
        # file) -- sort ascending explicitly to open in file order
        table.setSortingEnabled(True)
        table.sortItems(0, Qt.SortOrder.AscendingOrder)
        table.cellClicked.connect(
            lambda row, _col: self._starlist_show_detail(row))
        table.cellDoubleClicked.connect(
            lambda row, _col: self._starlist_pick(row))
        lay.addWidget(table, 1)

        detail_box = QtWidgets.QVBoxLayout()

        def _detail_label(text):
            lbl = QtWidgets.QLabel(text)
            lbl.setWordWrap(True)
            lbl.setStyleSheet("QLabel { font-size:11px; }")
            set_cue(lbl, "secondary")
            detail_box.addWidget(lbl)
            return lbl

        self._starlist_detail_id = _detail_label(
            "Click a row to see its details.")
        self._starlist_detail_azel = _detail_label("")
        self._starlist_detail_moon = _detail_label("")
        detail_frame = QtWidgets.QFrame()
        detail_frame.setLayout(detail_box)
        detail_frame.setStyleSheet(
            "QFrame { padding: 4px; }")
        lay.addWidget(detail_frame)
        self._starlist_selected_row = None
        note = QtWidgets.QLabel(
            "\"TT star → name\" rows are tip-tilt-star candidates the list "
            "explicitly links (target=) to that science target; they can "
            "still be picked directly. Starlists carry no proper motion, so "
            "picking a row zeroes the PM fields.")
        note.setStyleSheet("QLabel { font-size:11px; }")
        set_cue(note, "secondary")
        note.setWordWrap(True)
        lay.addWidget(note)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(dlg.close)
        # same reasoning as now_btn above: don't let Enter in the date/time
        # fields (or anywhere else in the dialog) close it unexpectedly
        close.setAutoDefault(False)
        close.setDefault(False)
        lay.addWidget(close)
        # however it closes (button, window X) drop the Qt references so
        # nothing can touch a deleted table/dialog afterwards
        dlg.finished.connect(lambda *_: self._on_starlist_dialog_closed())
        dlg.show()
        self._starlist_dialog = dlg
        self._starlist_table = table

    # Moon-proximity warning thresholds, degrees -- adjust if these don't
    # match observing practice; there's no standing convention elsewhere in
    # this codebase to match, so these are reasonable astronomer defaults
    # (a nominal ~30 deg avoidance radius, ~15 deg as a harder warning).
    _MOON_ERR_DEG = 15.0
    _MOON_WARN_DEG = 30.0

    @staticmethod
    def _starlist_now_hst():
        """Current time as a naive HST datetime -- matching the app-wide
        internal convention (data.py's _fmt_hm: "All internal datetimes
        stay HST; only labels convert")."""
        from datetime import timezone
        utc_now = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
        return utc_now - datetime.timedelta(hours=engine.HST_TO_UTC_HOURS)

    @staticmethod
    def _starlist_dt_for_display(hst_dt, is_utc):
        """`hst_dt` (internal, HST) converted to whichever wall-clock the
        dialog's date/time widgets are currently showing (UT if is_utc,
        else HST unchanged)."""
        if is_utc:
            return hst_dt + datetime.timedelta(hours=engine.HST_TO_UTC_HOURS)
        return hst_dt

    def _starlist_hst_to_utc(self, hst_dt):
        return hst_dt + datetime.timedelta(hours=engine.HST_TO_UTC_HOURS)

    def _starlist_show_detail(self, row):
        """Single-click: fill the detail panel with this entry's magnitudes,
        az/el, and Moon separation/illumination, evaluated at the dialog's
        "Evaluate at" time (defaults to now, editable -- see
        _show_starlist_dialog). self._starlist_eval_dt is always HST
        internally; compute_airmass_curve wants HST (it converts to UTC
        itself), while moon_separation_deg/moon_illumination_fraction want
        UTC directly -- each call below converts as needed rather than
        assuming one convention for both (an earlier version of this method
        silently fed the same ambiguous value to both, which happened to be
        ~10 hours off for the Moon calculations specifically).

        Az/el is the same for K1/K2 (they share one location) for the
        NUMBERS, but the pointing-limit classification (open/vignetted/
        blocked -- engine.pointing_state, which already models the Nasmyth-
        deck wedge) is telescope-specific, so it uses the live Telescope
        selector (self.tel_k1/tel_k2). Colors follow the app's existing
        semantic cues (theme.py: 'err' red / 'warn' amber / 'ok' green /
        'secondary' neutral) so they stay legible in dark mode too. Never
        raises: an astropy/network hiccup (e.g. a stale offline IERS table)
        degrades to an "unavailable" line rather than losing the whole
        dialog."""
        table = self._starlist_table
        if table is None or not (0 <= row < table.rowCount()):
            return
        idx = table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        if idx is None or not (0 <= idx < len(self._starlist_entries)):
            return
        self._starlist_selected_row = row
        e = self._starlist_entries[idx]
        mags = _starlist_entry_mags(e)
        mag_bits = ", ".join(f"{k}={v:.1f}" for k, v in mags.items()
                             if v is not None)
        self._starlist_detail_id.setText(
            f"<b>{e['name']}</b> &nbsp; RA {e['ra']} &nbsp; Dec {e['dec']}"
            + (f" &nbsp; ({mag_bits})" if mag_bits else ""))

        when_hst = getattr(self, "_starlist_eval_dt", None) \
            or self._starlist_now_hst()
        when_utc = self._starlist_hst_to_utc(when_hst)
        is_utc = getattr(self, "_starlist_dialog_is_utc", self._utc())
        when_disp = self._starlist_dt_for_display(when_hst, is_utc)
        when_txt = f"{when_disp:%Y-%m-%d %H:%M} {'UT' if is_utc else 'HST'}"
        tel = "K1" if self.tel_k1.isChecked() else "K2"
        try:
            am, el, az = engine.compute_airmass_curve(e["ra"], e["dec"],
                                                       [when_hst])
            el0, az0 = float(el[0]), float(az[0])
            state = engine.pointing_state(el0, az0, tel)
            cue, note = {
                "open": ("ok", ""),
                "vignetted": ("warn", f"  — VIGNETTED by {tel}'s Nasmyth "
                              "deck"),
                "blocked": ("err", ("  — BELOW HORIZON" if el0 < 0
                            else f"  — BLOCKED ({tel} pointing limit)")),
            }[state]
            self._starlist_detail_azel.setText(
                f"Az/El at {when_txt}: {az0:.1f}° / {el0:.1f}°{note}")
            set_cue(self._starlist_detail_azel, cue)
        except Exception as ex:
            self._starlist_detail_azel.setText(f"Az/El: unavailable ({ex})")
            set_cue(self._starlist_detail_azel, "secondary")
        try:
            sep = engine.moon_separation_deg(e["ra"], e["dec"], when_utc)
            illum = engine.moon_illumination_fraction(when_utc) * 100.0
            if sep < self._MOON_ERR_DEG:
                cue, note = "err", "  — VERY CLOSE TO THE MOON"
            elif sep < self._MOON_WARN_DEG:
                cue, note = "warn", "  — close to the Moon"
            else:
                cue, note = "ok", ""
            self._starlist_detail_moon.setText(
                f"Moon separation at {when_txt}: {sep:.1f}°  "
                f"(illumination {illum:.0f}%, 100% = full moon){note}")
            set_cue(self._starlist_detail_moon, cue)
        except Exception as ex:
            self._starlist_detail_moon.setText(
                f"Moon separation: unavailable ({ex})")
            set_cue(self._starlist_detail_moon, "secondary")

    def _starlist_refresh_detail(self):
        """Re-evaluate the currently-shown row (if any) -- called when the
        "Evaluate at" time changes, so the detail panel updates live rather
        than requiring a re-click."""
        row = getattr(self, "_starlist_selected_row", None)
        if row is not None:
            self._starlist_show_detail(row)

    def _starlist_refresh_ha_column(self):
        """Recompute the HA column for every row -- called when the
        "Evaluate at" time changes (see _on_display_changed in
        _show_starlist_dialog). Updates cell VALUES in place rather than
        rebuilding the table/dialog: rebuilding from within the very
        date/time widget's own change signal would destroy a widget still
        inside its own event handler."""
        table = self._starlist_table
        ha_col = getattr(self, "_starlist_ha_col", None)
        if table is None or ha_col is None:
            return
        when_utc = self._starlist_hst_to_utc(
            getattr(self, "_starlist_eval_dt", None)
            or self._starlist_now_hst())
        # sorting OFF for the duration: if the table happens to be sorted
        # BY the HA column, each setItem() below would re-sort mid-loop,
        # shuffling rows out from under the remaining iterations (the same
        # reason _nirc2_fill_csv_table brackets its own bulk update)
        table.setSortingEnabled(False)
        for row in range(table.rowCount()):
            idx = table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            if idx is None or not (0 <= idx < len(self._starlist_entries)):
                continue
            e = self._starlist_entries[idx]
            try:
                ha = engine.hour_angle_hours(e["ra"], when_utc)
                item = SortableItem(f"{ha:+.2f}h", ha)
            except Exception:
                item = SortableItem("—", None)
            table.setItem(row, ha_col, item)
        table.setSortingEnabled(True)

    def _starlist_pick(self, row):
        """Double-click: make this entry the science target AND, if the
        starlist links TT-star candidate(s) to it (target=<name>), set the
        guide star too. Goes through _add_target + _on_target_selected (the
        same path the Targets dropdown uses) so the entry lands in tonight's
        target list AND the fields -- including the PM spinboxes, zeroed: a
        starlist carries no proper motion, and leaking the previously-loaded
        target's PM onto these coordinates would silently shift the
        pointing. `row` is the VISUAL row -- the table is user-sortable, so
        the entry index comes from the '#' item's UserRole, never from the
        row number itself."""
        table = self._starlist_table
        if table is None or not (0 <= row < table.rowCount()):
            return
        idx = table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        if idx is None or not (0 <= idx < len(self._starlist_entries)):
            return
        e = self._starlist_entries[idx]
        gs_note = self._resolve_starlist_guide_star(e)
        idx = self._add_target(e["name"], e["ra"], e["dec"], 0.0, 0.0,
                               tt_offset_cfg=self.tt_offset.get_config(),
                               tt_mag=self._last_gs_mag, select=True)
        msg = f"Target from starlist: {e['name']}  {e['ra']}  {e['dec']}{gs_note}"
        if e["equinox"] not in ("2000", "2000.0"):
            msg += (f" — ⚠ equinox {e['equinox']} entered as-is (the tool "
                    f"assumes J2000/ICRS coordinates)")
        self.status.setText(msg)

    def _resolve_starlist_guide_star(self, target_entry):
        """Set the guide star for `target_entry` (a picked starlist target
        row) and return a short status-note string.

        TT/LGS: entries in the SAME loaded starlist whose target= field
        names this one are TT-star candidates: none -> the target is ITS
        OWN TT star (on-axis, ITS OWN magnitude -- not just a zero offset
        with a stale magnitude); one -> that star directly; 2+ -> ranked by
        delivered Strehl (_rank_gs_candidates), the top-ranked one wins --
        unless there is no prepared night yet to rank against, in which
        case the pick is DEFERRED (self._pending_gs) and resolved
        automatically after the next successful Run.

        NGS: the guide star is ALWAYS the target itself (no starlist
        exception exists for NGS), so ngs_bright is set to the target's own
        R magnitude on every pick.

        Either magnitude missing from the starlist's own fields -> one
        async SIMBAD lookup for the target name (ResolveWorker); a failed
        lookup, or one with no usable magnitude, WARNS that the TT/NGS
        magnitudes were not updated rather than leaving a stale value
        looking deliberate (see _on_gs_mag_resolved)."""
        # same_star_name, not ==: a target= value can't contain spaces (it's
        # a whitespace-delimited token), so real lists write underscores for
        # a spaced name ('target=Syn_Cluster_A' -> 'Syn Cluster A')
        candidates = [c for c in self._starlist_entries
                     if engine.same_star_name(c["target"], target_entry["name"])]
        self._pending_gs = None
        self._pending_gs_mag = None
        self._last_gs_mag = None
        if not candidates:
            self.tt_offset.set_config(dict(_ON_AXIS_OFFSET_CFG))
            note = " — guide star = target itself (on-axis)"
            set_tt = True
        elif len(candidates) > 1 and (self.prep is None or self.res is None):
            self.tt_offset.set_config(dict(_ON_AXIS_OFFSET_CFG))
            self._pending_gs = {"target": target_entry["name"],
                                "ra_deg": target_entry["ra_deg"],
                                "dec_deg": target_entry["dec_deg"],
                                "candidates": candidates}
            note = (f" — {len(candidates)} guide-star candidates; run the "
                    f"estimator to auto-pick the best one")
            set_tt = False              # the ranked winner brings its own mag
        else:
            if len(candidates) == 1:
                winner, why = candidates[0], "only candidate"
            else:
                winner, why = self._rank_gs_candidates(candidates, target_entry)
            self._apply_gs_winner(winner)
            note = f" — guide star: {winner['name']} ({why})"
            set_tt = False              # TT mag came from the winner
        mag_note = self._apply_target_self_mags(
            _starlist_entry_mags(target_entry), target_entry["name"], set_tt)
        return note + mag_note

    def _apply_target_self_mags(self, mags, target_name, set_tt):
        """Apply the target's OWN magnitudes wherever the target is its own
        guide star: ngs_bright always (NGS has no starlist exception), and
        tt_mag when set_tt (no linked TT candidate supplied one). Missing
        magnitude(s) -> ONE async SIMBAD lookup by name (ResolveWorker),
        applied or warned about in _on_gs_mag_resolved. Returns a short
        status-note fragment."""
        bits, missing = [], []
        band = self._tt_sensor_band()
        if set_tt:
            mag, kind, label = engine.estimate_sensing_mag(mags, band)
            if mag is not None:
                self._tt_mag_auto = True
                self.tt_mag.setValue(mag)
                self._tt_mag_auto = False
                self._last_gs_mag = mag
                bits.append(f"TT {band}{'=' if kind == 'exact' else '≈'}{mag:.1f}")
            else:
                missing.append("tt")
        nmag, nkind, _nlabel = engine.estimate_sensing_mag(mags, "R")
        if nmag is not None:
            self.ngs_bright.setValue(nmag)
            bits.append(f"NGS R{'=' if nkind == 'exact' else '≈'}{nmag:.1f}")
        else:
            missing.append("ngs")
        note = ("; " + ", ".join(bits)) if bits else ""
        # an R guessed from IR-only photometry gets the standing dusty-field
        # warning (same one the catalogue picker/ranking append)
        warn = self._reddening_warning(mags, "R") if bits else None
        if warn:
            note += f"  ⚠ {warn}"
        if missing:
            self._pending_gs_mag = {"target": target_name,
                                    "need_tt": "tt" in missing,
                                    "need_ngs": "ngs" in missing}
            w = ResolveWorker(target_name, self)
            w.done.connect(lambda res, err, w=w:
                           self._on_gs_mag_resolved(w, res, err))
            self._gs_mag_worker = w
            w.start()
            note += "; no magnitude in the list — asking SIMBAD…"
        return note

    def _on_gs_mag_resolved(self, worker, result, err):
        """The async SIMBAD magnitude lookup finished. Applies the found
        magnitude(s) to whatever _apply_target_self_mags couldn't fill from
        the starlist -- or WARNS that they were NOT updated (a stale
        magnitude silently left in place looks deliberate; a warning is the
        honest failure mode). Dropped outright if a newer lookup has
        superseded this one, or the user has moved to a different target."""
        if worker is not self._gs_mag_worker:
            return                       # superseded by a newer pick
        self._gs_mag_worker = None
        pend, self._pending_gs_mag = self._pending_gs_mag, None
        if pend is None:
            return
        if not engine.same_star_name(self.tname_edit.text(), pend["target"]):
            return                       # user moved on; nothing to apply to
        mags = (result.get("mags") or {}) if result else {}
        bits, failed = [], []
        band = self._tt_sensor_band()
        if pend["need_tt"]:
            mag, kind, _label = engine.estimate_sensing_mag(mags, band)
            if mag is not None:
                self._tt_mag_auto = True
                self.tt_mag.setValue(mag)
                self._tt_mag_auto = False
                self._last_gs_mag = mag
                idx = self.target_select.currentIndex()
                if 0 <= idx < len(self._targets):
                    self._targets[idx]["tt_mag"] = mag
                bits.append(
                    f"TT {band}{'=' if kind == 'exact' else '≈'}{mag:.1f}")
            else:
                failed.append("TT-star magnitude NOT updated")
        if pend["need_ngs"]:
            nmag, nkind, _nl = engine.estimate_sensing_mag(mags, "R")
            if nmag is not None:
                self.ngs_bright.setValue(nmag)
                bits.append(f"NGS R{'=' if nkind == 'exact' else '≈'}{nmag:.1f}")
            else:
                failed.append("NGS magnitude NOT adjusted")
        msg = self.status.text()
        if bits:
            msg += f"  |  SIMBAD: {', '.join(bits)}"
            warn = self._reddening_warning(mags, "R")
            if warn:
                msg += f"  ⚠ {warn}"
        if failed:
            why = err or "no usable magnitude in SIMBAD"
            msg += f"  |  ⚠ {'; '.join(failed)} ({why}) — set manually"
        self.status.setText(msg)

    def _apply_gs_winner(self, winner):
        """Set self.tt_offset/self.tt_mag to `winner` (a starlist entry) via
        the SAME mechanism the field-map catalogue picker uses
        (_fm_select_star), and remember whether a magnitude was actually
        derivable in self._last_gs_mag. Callers use that (not a re-read of
        self.tt_mag.value()) to decide whether to PERSIST tt_mag for this
        target -- a winner with no derivable magnitude must not silently
        persist whatever stale value happened to be showing from a
        PREVIOUS target (_fm_select_star leaves tt_mag untouched when it
        can't derive one)."""
        mags = _starlist_entry_mags(winner)
        band = self._tt_sensor_band()
        mag, _kind, _label = engine.estimate_sensing_mag(mags, band)
        self._last_gs_mag = mag
        star = {"id": winner["name"], "ra": winner["ra_deg"],
               "dec": winner["dec_deg"], "mags": mags}
        self._fm_select_star(star, "tt")

    def _rank_gs_candidates(self, candidates, target_entry):
        """Best LGS TT-star among `candidates` (2+ starlist entries sharing
        a target= link), by delivered Strehl at the target -- the SAME
        engine.rank_guide_stars the field map's Rank button uses. Caller
        ensures self.prep/self.res exist. Falls back to closest-by-field-
        offset if this night has no MASS profiles (LGS/TT physics needs
        them for a delivered-Strehl comparison) -- a DATA limitation, not a
        timing one, so there is nothing to defer for. Returns (winning
        starlist entry, a short reason string)."""
        stars = [{"id": i, "ra": c["ra_deg"], "dec": c["dec_deg"],
                 "mags": _starlist_entry_mags(c)}
                for i, c in enumerate(candidates)]
        stars = engine.stars_field_xy(stars, target_entry["ra_deg"],
                                      target_entry["dec_deg"])
        when, t_hst = self._fm_when_time()
        snap = engine.field_snapshot(self.args_cached, self.prep, self.res,
                                     when, t_hst)
        if snap is None:
            best = min(stars, key=lambda s: math.hypot(s["x"], s["y"]))
            return candidates[best["id"]], "no MASS profiles — picked by proximity"
        mode = "ltao" if self.prep.tomography_on else "single"
        sensor = self._tt_sensor_band()
        with engine.budget_overrides(**(self.last_offsets or {})):
            ranked = engine.rank_guide_stars(
                self.args_cached, self.prep, snap, mode, stars, (0.0, 0.0),
                sensor, metric="strehl")
        usable = [e for e in ranked if e["rank"] is not None]
        if not usable:
            best = min(stars, key=lambda s: math.hypot(s["x"], s["y"]))
            return candidates[best["id"]], "all candidates excluded — picked closest"
        return candidates[usable[0]["id"]], \
            f"ranked #1 of {len(candidates)} by delivered Strehl"

    def _resolve_pending_guide_star(self):
        """Called after every successful run/recompute (mainwindow.py's
        recompute_and_draw): if a starlist pick deferred its guide-star
        ranking for lack of a prepared night, and the CURRENTLY ACTIVE
        target is still the one that was waiting, resolve it now. If the
        user has since moved on to a different target, the deferred pick is
        simply dropped -- there is no live control left for it to apply to."""
        pend = self._pending_gs
        if pend is None or self.prep is None or self.res is None:
            return
        if not engine.same_star_name(self.tname_edit.text(), pend["target"]):
            self._pending_gs = None
            return
        winner, why = self._rank_gs_candidates(pend["candidates"], pend)
        self._apply_gs_winner(winner)
        self._pending_gs = None
        idx = self.target_select.currentIndex()
        if 0 <= idx < len(self._targets):
            self._targets[idx]["tt_offset_cfg"] = self.tt_offset.get_config()
            self._targets[idx]["tt_mag"] = self._last_gs_mag
        self.status.setText(
            self.status.text() + f"  |  guide star resolved: {winner['name']} "
            f"({why})")

    def _on_starlist_dialog_closed(self):
        self._starlist_dialog = None
        self._starlist_table = None
        self._starlist_detail_id = None
        self._starlist_detail_azel = None
        self._starlist_detail_moon = None
        self._starlist_time_edit = None
        self._starlist_date_edit = None
        self._starlist_selected_row = None
        self._starlist_ha_col = None
