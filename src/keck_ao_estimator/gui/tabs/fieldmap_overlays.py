"""Interactive overlays on the field map, kept out of the (already large)
field-map rendering module: the user-dropped science-target markers and the
guide-star catalogue overlay (load a Vizier catalogue -> stars plotted on the
field, right-click one to select it as the TT/NGS guide star, taking its
position and best-available magnitude).

FieldMapMixin's renderers call into here (_draw_fm_targets, _draw_catalog_stars,
_fm_eval_markers); everything the observer clicks/drops on the map lives here.
"""
import numpy as np
from qtcompat import Qt, QtGui, QtWidgets

import keck_ao_estimator as engine

from ..constants import (
    FIELD_OF_REGARD_RADIUS_ARCSEC, FM_C_CATSTAR, FM_C_CATSTAR_RING, FM_C_MARKER,
    FM_C_TSS, FM_C_WARN,
)
from ..widgets import SortableItem
from ..workers import CatalogFetchWorker
from ..theme import set_cue

# Catalogue-star marker sizing (scatter area, pt²): brighter sensing-band
# magnitude -> bigger dot. Anchored at _CAT_MAG_BRIGHT (max size) down to the
# sensor's faint guide limit (min size); too-faint/unknown stars use the min.
_CAT_MAG_BRIGHT = 8.0
_CAT_SIZE_MIN = 14.0
_CAT_SIZE_MAX = 130.0


class FieldMapOverlaysMixin:
    # ---- overlay controls + state (built into the field-map tab) ------------
    def _init_fm_overlays(self, v):
        """Build the guide-star catalogue row and the dropped-targets row, seed
        overlay state, and connect the map's right-click menu. Called from
        FieldMapMixin._build_field_map_tab with its layout `v`."""
        # dropped science targets
        self._fm_markers = []                # {name, x, y, val}
        self._fm_click_xy = None
        # catalogue overlay
        self._catalog_stars = []             # {id, ra, dec, mags}
        self._catalog_name = ""
        self._catalog_worker = None
        self._catalog_inspected = None       # id of the left-clicked star
        # guide-star auto-ranking (Rank button): result of the last
        # rank_guide_stars() call, [] until run or invalidated by a later
        # control change (see _invalidate_gs_ranking)
        self._gs_ranking = []
        self._gs_rank_dialog = None
        self._gs_rank_table = None           # QTableWidget in the open dialog
        # the catalogue star (if any) currently backing the TT-star selection,
        # so a STRAP<->TRICK sensor switch can re-derive the magnitude in the
        # new band instead of leaving a stale one (see _sync_tt_mag_for_band
        # in gui/tabs/lgs.py); cleared on a manual tt_mag edit
        self._tt_star_ref = None
        self._tt_mag_auto = False            # guards a programmatic tt_mag set

        rowc = QtWidgets.QHBoxLayout()
        rowc.addWidget(QtWidgets.QLabel("Guide-star catalog:"))
        self.fm_catalog = QtWidgets.QComboBox()
        self.fm_catalog.addItems(list(engine.CATALOGS))
        self.fm_catalog.setToolTip(
            "Look up guide-star candidates around the field centre from a "
            "Vizier catalogue. Right-click a star on the map to set it as the "
            "TT or NGS guide star (its position and best-available magnitude).")
        rowc.addWidget(self.fm_catalog)
        self.fm_catalog_load = QtWidgets.QPushButton("Load")
        self.fm_catalog_load.clicked.connect(self._fetch_catalog)
        rowc.addWidget(self.fm_catalog_load)
        self.fm_catalog_clear = QtWidgets.QPushButton("Clear")
        self.fm_catalog_clear.clicked.connect(self._clear_catalog)
        rowc.addWidget(self.fm_catalog_clear)
        self.fm_catalog_rank = QtWidgets.QPushButton("Rank")
        self.fm_catalog_rank.setEnabled(False)
        self.fm_catalog_rank.setToolTip(
            "Rank every loaded candidate by the delivered Strehl/FWHM AT THE "
            "SCIENCE TARGET if each, in turn, were used as the guide/TT-tilt "
            "star for the current Mode -- the actual decision of picking a "
            "guide star, not just inspecting candidates one at a time. Top 3 "
            "are badged on the map; the full table opens in a window "
            "(double-click a row to select that star as the TT/NGS star).")
        self.fm_catalog_rank.clicked.connect(self._rank_guide_stars)
        rowc.addWidget(self.fm_catalog_rank)
        self.fm_catalog_status = QtWidgets.QLabel()
        set_cue(self.fm_catalog_status, "secondary")
        rowc.addWidget(self.fm_catalog_status, 1)
        v.addLayout(rowc)

        rowt = QtWidgets.QHBoxLayout()
        rowt.addWidget(QtWidgets.QLabel("Targets:"))
        self.fm_target_list = QtWidgets.QListWidget()
        self.fm_target_list.setFixedHeight(58)
        self.fm_target_list.setToolTip(
            "Science targets dropped on the field map, each with its predicted "
            "performance at that field position. Right-click the map to add "
            "one, or to place the laser / TT / NGS star.")
        rowt.addWidget(self.fm_target_list, 1)
        self.fm_target_clear = QtWidgets.QPushButton("Clear")
        self.fm_target_clear.setToolTip("remove all dropped science targets")
        self.fm_target_clear.clicked.connect(self._fm_clear_targets)
        rowt.addWidget(self.fm_target_clear)
        v.addLayout(rowt)

        self._fm_holder["canvas"].mpl_connect(
            "button_press_event", self._on_fm_canvas_click)
        self._fm_holder["canvas"].mpl_connect(
            "scroll_event", self._on_fm_scroll)

    # ---- mouse-wheel zoom -----------------------------------------------
    _FM_ZOOM_STEP = 0.85              # per wheel notch
    _FM_ZOOM_MIN_HALF = 0.5           # arcsec -- don't zoom in past this
    _FM_ZOOM_MAX_HALF = 300.0         # arcsec -- don't zoom out past this

    def _on_fm_scroll(self, event):
        """Zoom the field map in/out about the cursor on a mouse-wheel
        scroll, like the nav toolbar's magnifier but without needing to
        drag a box. Purely a view change (ax.set_xlim/ylim) -- like the nav
        toolbar's own zoom, it's reset by the next full redraw (a control
        change, or the trailing full-resolution redraw after scrubbing)."""
        if event.inaxes is None or event.xdata is None or event.button not in (
                "up", "down"):
            return
        ax = event.inaxes
        scale = self._FM_ZOOM_STEP if event.button == "up" else 1.0 / self._FM_ZOOM_STEP
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        old_half = max((x1 - x0) / 2.0, (y1 - y0) / 2.0)   # square-ish view
        new_half = min(max(old_half * scale, self._FM_ZOOM_MIN_HALF),
                       self._FM_ZOOM_MAX_HALF)
        if abs(new_half - old_half) < 1e-9:
            return                     # already at a zoom bound, no-op
        xd, yd = event.xdata, event.ydata
        relx = (xd - x0) / (x1 - x0) if x1 != x0 else 0.5
        rely = (yd - y0) / (y1 - y0) if y1 != y0 else 0.5
        new_w = new_h = new_half * 2.0
        ax.set_xlim(xd - new_w * relx, xd + new_w * (1 - relx))
        ax.set_ylim(yd - new_h * rely, yd + new_h * (1 - rely))
        ax.figure.canvas.draw_idle()

    # ---- right-click menu: select a star, drop a target, place laser/star ----
    def _on_fm_canvas_click(self, event):
        """Field-map clicks: LEFT-click a catalogue star to inspect its
        magnitudes (no selection); RIGHT-click for the menu -- select a nearby
        star as the TT/NGS guide star, drop a science target, or place the
        laser / TT / NGS star at the click. event.xdata/ydata are already in
        the plot frame (x = West+, y = North+ arcsec from the field centre)."""
        if event.inaxes is None or event.xdata is None:
            return
        if self.res is None or self.plot_tabs.currentIndex() != 1:
            return
        x, y = float(event.xdata), float(event.ydata)
        if event.button == 1:                          # inspect (no drag/zoom)
            if getattr(self._fm_holder["navbar"], "mode", ""):
                return                                 # pan/zoom active
            self._inspect_catalog_star(x, y)
            return
        if event.button != 3:
            return
        self._fm_click_xy = (x, y)
        menu = QtWidgets.QMenu(self)
        star = self._catalog_star_near(x, y)
        if star is not None:
            sid = star["id"]
            menu.addAction(f"Set “{sid}” as TT star",
                           lambda: self._fm_select_star(star, "tt"))
            menu.addAction(f"Set “{sid}” as NGS star",
                           lambda: self._fm_select_star(star, "ngs"))
            menu.addSeparator()
        menu.addAction(f"Drop science target here  ({x:+.1f}, {y:+.1f})″",
                       lambda: self._fm_add_target(x, y))
        menu.addAction(f"Redefine field centre here  ({x:+.1f}, {y:+.1f})″",
                       lambda: self._fm_set_real_target(x, y))
        menu.addSeparator()
        menu.addAction("Put laser here", lambda: self._fm_put_laser(x, y))
        menu.addAction("Put TT star here",
                       lambda: self._fm_put_star(self.tt_offset, x, y))
        menu.addAction("Put NGS star here",
                       lambda: self._fm_put_star(self.ngs_offset, x, y))
        if self._fm_markers:
            menu.addSeparator()
            menu.addAction("Remove nearest target",
                           lambda: self._fm_remove_near(x, y))
            menu.addAction("Clear all targets", self._fm_clear_targets)
        menu.exec(QtGui.QCursor.pos())

    # ---- dropped science targets -------------------------------------------
    def _fm_add_target(self, x, y):
        """Add a science target at plot-frame (x, y)″ from the field centre."""
        n = len(self._fm_markers) + 1
        self._fm_markers.append({"name": f"T{n}", "x": float(x), "y": float(y),
                                 "val": None})
        self._fieldmap_dirty = True
        self._render_field_map_if_visible()

    def _fm_set_real_target(self, x, y):
        """Redefine the field centre: MOVE the currently-selected target's
        own RA/Dec in place to plot-frame (x, y)″ off the OLD field centre
        (the effective target position, _effective_target_coords -- same
        origin the marker plot itself uses), converted with the SAME sign
        convention _fm_put_star already established (ΔRA-East = -x,
        ΔDec-North = y). Unlike _fm_add_target (a lightweight, unsaved field
        marker), this really does move the active target -- but unlike an
        earlier version of this action, it does NOT create a new T1/T2/...
        target-list entry every click (Eduardo 2026-07-28: that cluttered
        the list every time he just wanted to look at a different part of
        the field). _add_target's own dedup-by-name keys on the CURRENT
        name, so passing it through unchanged updates the existing entry
        in place rather than adding a new one -- the same mechanism
        _save_current_target (target.py) uses to update a re-saved
        target."""
        eff = self._effective_target_coords()
        if eff is None:
            self.status.setText(
                "No target position to re-centre from -- enter a base "
                "RA/Dec first.")
            return
        import astropy.units as u
        center = engine.parse_radec(*eff)
        new = center.spherical_offsets_by(-x * u.arcsec, y * u.arcsec)
        ra_s = new.ra.to_string(unit="hourangle", sep="hms", precision=3)
        dec_s = new.dec.to_string(unit="deg", sep="dms", precision=2,
                                  alwayssign=True)
        self.ra_edit.setText(ra_s)
        self.dec_edit.setText(dec_s)
        name = self.tname_edit.text().strip()
        self._add_target(name, ra_s, dec_s,
                         self.pmra_spin.value(), self.pmdec_spin.value(),
                         tt_offset_cfg=self.tt_offset.get_config(),
                         tt_mag=self.tt_mag.value(), select=True)
        self.status.setText(
            f"Field re-centred on {ra_s}  {dec_s}  "
            f"({x:+.1f}, {y:+.1f})″ from the previous centre")
        # unlike every sibling action in this file (_fm_add_target,
        # _fm_remove_near, _fm_clear_targets, ...), this one moved the
        # actual target rather than a field-map-only marker, so it's easy
        # to miss that the map itself still needs telling to redraw --
        # _add_target's chain (target.py) only refreshes the Target tab's
        # own fields, never _fieldmap_dirty. Without this, NOTHING on the
        # map re-renders (backdrop included) until an unrelated trigger
        # (Run, switching plot tabs, touching an fm_*/offset control)
        # happens to fire one (Eduardo 2026-07-28: the backdrop looked
        # frozen in place because the whole canvas was stale, not because
        # the backdrop-shift math was wrong).
        self._fieldmap_dirty = True
        self._render_field_map_if_visible()

    def _fm_put_laser(self, x, y):
        """Place the LGS/laser at plot-frame (x, y): set the LGS-offset
        magnitude + laser PA (N→E) so _laser_xy() lands there, and enable it."""
        r = float(np.hypot(x, y))
        self.laser_pa.setValue(float(np.degrees(np.arctan2(-x, y)) % 360.0))
        self.lgs_offset_enable.setChecked(True)
        self.lgs_offset.setValue(r)

    def _fm_put_star(self, entry, x, y):
        """Place a TT/NGS offset star at plot-frame (x, y): ΔRA (East) = −x,
        ΔDec (North) = y (see OffsetEntry.offset_xy). Used when only a field
        POSITION is known (a bare right-click on the map) -- no real sky
        coordinate to record."""
        entry.mode.setCurrentIndex(1)          # ΔRA / ΔDec
        entry.dra.setValue(-float(x)); entry.ddec.setValue(float(y))

    def _fm_put_star_radec(self, entry, ra_deg, dec_deg):
        """Place a TT/NGS offset star at an ABSOLUTE sky position (star
        RA/Dec mode), not a delta offset. Used for a catalogue star, whose
        real RA/Dec is known -- a planning user needs those coordinates to
        hand off (e.g. to an observing script), not just an offset from
        tonight's target that stops meaning anything once the target moves."""
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        c = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
        entry.mode.setCurrentIndex(2)          # star RA/Dec
        entry.sra.setText(
            c.ra.to_string(unit="hourangle", sep="hms", precision=3))
        entry.sdec.setText(
            c.dec.to_string(unit="deg", sep="dms", precision=2, alwayssign=True))

    def _fm_remove_near(self, x, y):
        if not self._fm_markers:
            return
        i = min(range(len(self._fm_markers)),
                key=lambda k: (self._fm_markers[k]["x"] - x) ** 2
                + (self._fm_markers[k]["y"] - y) ** 2)
        del self._fm_markers[i]
        self._fieldmap_dirty = True
        self._render_field_map_if_visible()

    def _fm_clear_targets(self):
        if not self._fm_markers:
            return
        self._fm_markers = []
        self._fieldmap_dirty = True
        self._render_field_map_if_visible()

    def _fm_eval_markers(self, snap, mode, metric, ngs_xy, tt_xy, laser_xy,
                         ngs_dvar, args=None, prep=None):
        """Compute each dropped target's metric at its field position, using
        the same per-point model as the map (must be called inside the same
        budget_overrides context as field_map_grid). args/prep default to
        the cached run; the prediction path passes its no-run surrogates."""
        args = self.args_cached if args is None else args
        prep = self.prep if prep is None else prep
        for m in self._fm_markers:
            try:
                m["val"] = engine.field_metric_at(
                    args, prep, snap, mode, metric,
                    ngs_xy[:2], tt_xy[:2], laser_xy, (m["x"], m["y"]),
                    ngs_delta_var=ngs_dvar)
            except Exception:
                m["val"] = None

    def _draw_fm_targets(self, ax, is_fwhm, transform=None):
        """Plot every dropped science target with its metric label.
        `transform` (default ax.transData) is the field-PA rotation, if any
        -- see _fm_rotation_transform. Returns the list of artists created
        (so the interactive renderer can remove them)."""
        trans = transform if transform is not None else ax.transData
        arts = []
        for m in self._fm_markers:
            arts += ax.plot(m["x"], m["y"], "s", ms=9, mfc=FM_C_MARKER,
                            mec="k", mew=0.8, zorder=7, transform=trans)
            if m["val"] is None:
                lbl = m["name"]
            elif is_fwhm:
                lbl = f"{m['name']}: {m['val']:.0f} mas"
            else:
                lbl = f"{m['name']}: {m['val']:.3f}"
            arts.append(ax.annotate(
                lbl, xy=(m["x"], m["y"]), xycoords=trans,
                textcoords="offset points",
                xytext=(7, 6), fontsize=7, fontweight="bold", color=FM_C_MARKER,
                zorder=8, bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                    ec=FM_C_MARKER, lw=0.7, alpha=0.9)))
        return arts

    def _fm_target_radec(self, x, y):
        """Absolute (RA, Dec) SkyCoord for a plot-frame (x, y)″ offset
        (x=West+/−East, y=North+) from the field centre, or None if there is
        currently no field centre to anchor it to."""
        fc = self._field_center_deg()
        if fc is None:
            return None
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        center = SkyCoord(fc[0] * u.deg, fc[1] * u.deg)
        return center.spherical_offsets_by(-float(x) * u.arcsec, float(y) * u.arcsec)

    def _fm_refresh_target_list(self, is_fwhm):
        """Sync the Targets list widget text with the current markers/values,
        including each dropped target's own RA/Dec (planning users need real
        coordinates, not just an offset from the field centre)."""
        if not hasattr(self, "fm_target_list"):
            return
        self.fm_target_list.clear()
        for m in self._fm_markers:
            r = float(np.hypot(m["x"], m["y"]))
            if m["val"] is None:
                val = "—"
            elif is_fwhm:
                val = f"{m['val']:.0f} mas"
            else:
                val = f"SR {m['val']:.3f}"
            c = self._fm_target_radec(m["x"], m["y"])
            radec = (f"  ({c.ra.to_string(unit='hourangle', sep='hms', precision=1)} "
                     f"{c.dec.to_string(unit='deg', sep='dms', precision=0, alwayssign=True)})"
                     if c is not None else "")
            self.fm_target_list.addItem(
                f"{m['name']}:  {r:.1f}″ from centre{radec}  →  {val}")

    # ---- guide-star catalogue overlay --------------------------------------
    def _fetch_catalog(self):
        """Query the selected catalogue around the field centre (radius = the
        field of regard) off the GUI thread."""
        fc = self._field_center_deg()
        if fc is None:
            self.fm_catalog_status.setText("set a target first")
            return
        name = self.fm_catalog.currentText()
        self.fm_catalog_load.setEnabled(False)
        self.fm_catalog_status.setText(f"querying {name}…")
        self._catalog_worker = CatalogFetchWorker(
            name, fc[0], fc[1], FIELD_OF_REGARD_RADIUS_ARCSEC, self)
        self._catalog_worker.done.connect(self._on_catalog_loaded)
        self._catalog_worker.start()

    def _on_catalog_loaded(self, name, stars, err):
        self.fm_catalog_load.setEnabled(True)
        if err:
            self.fm_catalog_status.setText(f"catalog error: {err}")
            return
        self._catalog_stars = list(stars)
        self._catalog_name = name
        self.fm_catalog_rank.setEnabled(bool(self._catalog_stars))
        self._invalidate_gs_ranking()
        self.fm_catalog_status.setText(
            f"{len(stars)} stars from {name} (right-click one to select)")
        self._fieldmap_dirty = True
        self._render_field_map_if_visible()

    def _clear_catalog(self):
        if not self._catalog_stars:
            return
        self._catalog_stars = []
        self._catalog_name = ""
        self._catalog_inspected = None
        self.fm_catalog_rank.setEnabled(False)
        self._invalidate_gs_ranking()
        self.fm_catalog_status.setText("")
        self._fieldmap_dirty = True
        self._render_field_map_if_visible()

    def _catalog_stars_xy(self):
        """The loaded catalogue stars with current plot-frame (x, y) offsets
        from the field centre (recomputed live, so they track a moving target).
        [] if none loaded or no field centre."""
        fc = self._field_center_deg()
        if not self._catalog_stars or fc is None:
            return []
        try:
            return engine.stars_field_xy(self._catalog_stars, fc[0], fc[1])
        except Exception:
            return []

    def _inspect_catalog_star(self, x, y):
        """Left-click on the MAP: inspect the nearest star (or clear, if the
        click missed one). Delegates to _inspect_star."""
        self._inspect_star(self._catalog_star_near(x, y))

    def _inspect_star(self, star):
        """Highlight `star` (a catalogue star dict with id/mags, or None to
        clear): ring it on the map (cyan), select its row in an open ranking
        table, and report its magnitudes + any reddening warning in the status
        line. Selection/highlight only -- NEVER sets it as the guide star
        (that stays a double-click in the table, or the right-click map menu).
        Shared by map left-clicks (_inspect_catalog_star) and ranking-table
        row clicks (_gs_rank_row_clicked), so map<->table highlighting stays
        in sync both ways."""
        self._catalog_inspected = star["id"] if star else None
        self._gs_rank_highlight(self._catalog_inspected)
        if star is not None:
            band = self._tt_sensor_band()
            have = ", ".join(f"{b}={v:.1f}" for b, v in star["mags"].items()
                             if v is not None) or "no magnitudes"
            mag, kind, label = engine.estimate_sensing_mag(star["mags"], band)
            if mag is None:
                tail = f"; no {band} derivable"
            elif kind == "exact":
                tail = f"; {band}={mag:.1f}"
            elif kind == "est":
                tail = f"; {band}≈{mag:.1f} (est. {label})"
            else:
                tail = f"; {band}≈{mag:.1f} (rough, {label})"
            warn = self._reddening_warning(star["mags"], band)
            self.fm_catalog_status.setText(
                f"{self._catalog_name} “{star['id']}”: {have}{tail}"
                + (f"   ⚠ {warn}" if warn else ""))
        self._fieldmap_dirty = True
        self._render_field_map_if_visible()

    def _reddening_warning(self, mags, band):
        """Amber 'verify, don't assume' string when an OPTICAL (R) sensing
        magnitude guessed from this star's near-IR photometry is likely
        unreliable due to interstellar reddening (dust dims the optical far
        more than the IR, so the star may be invisible to a STRAP-class WFS
        even while bright in K -- the dusty-field trap), else None. Only fires
        for R sensing of an IR-only star; an IR sensor (TRICK H/K) reads the
        photometry directly and is unaffected."""
        if band != "R":
            return None
        _mag, kind, _lab = engine.estimate_sensing_mag(mags, band)
        if kind != "near":                 # real optical band / colour transform
            return None
        _a_r, note = engine.optical_extinction_lower_bound(mags)
        if not note:
            return None
        return (f"IR-red ({note}) — optical WFS may not see it; verify vs "
                f"imagery")

    def _catalog_star_near(self, x, y, tol_arcsec=3.0):
        """The loaded catalogue star nearest to plot-frame (x, y), or None if
        none is within tol_arcsec (so a right-click only offers 'select' when
        it actually lands on a star)."""
        best, bd = None, tol_arcsec ** 2
        for s in self._catalog_stars_xy():
            d = (s["x"] - x) ** 2 + (s["y"] - y) ** 2
            if d < bd:
                best, bd = s, d
        return best

    def _tt_sensor_band(self):
        """The TT sensor's working band label ('R' STRAP / 'H' or 'K' TRICK)."""
        s = self.tt_sensor.currentText().lower()
        if "trick" in s and "(h)" in s:
            return "H"
        if "trick" in s and "(k)" in s:
            return "K"
        return "R"

    def _fm_select_star(self, star, role):
        """Select a catalogue star as the TT ('tt') or NGS ('ngs') guide star:
        set that control's POSITION -- as the star's real, ABSOLUTE RA/Dec
        (star RA/Dec mode), not a delta offset, so a planning user has actual
        coordinates to work with -- and its MAGNITUDE in the sensor's working
        band -- R for NGS/STRAP, H/K for TRICK -- preferring the catalogue's
        own band, else a published colour-transform ESTIMATE (flagged),
        reporting how it was derived."""
        if role == "tt":
            entry, magw, band = self.tt_offset, self.tt_mag, self._tt_sensor_band()
            # remember this star so a later STRAP<->TRICK sensor switch can
            # re-derive the magnitude in the new band (see
            # _sync_tt_mag_for_band in gui/tabs/lgs.py) instead of leaving a
            # stale one behind
            self._tt_star_ref = star
        else:
            entry, magw, band = self.ngs_offset, self.ngs_bright, "R"
        self._fm_put_star_radec(entry, star["ra"], star["dec"])
        mag, kind, label = engine.estimate_sensing_mag(star["mags"], band)
        head = f"{role.upper()} star ← {self._catalog_name} “{star['id']}”"
        if mag is None:
            self.fm_catalog_status.setText(
                f"{head}: no {band}-band magnitude derivable — set it manually")
            return
        if role == "tt":
            self._tt_mag_auto = True
        magw.setValue(mag)
        self._tt_mag_auto = False
        if kind == "exact":
            self.fm_catalog_status.setText(f"{head}: {band}={mag:.1f}")
        elif kind == "est":
            self.fm_catalog_status.setText(
                f"{head}: {band}≈{mag:.1f} (estimated from {label})")
        else:                                          # 'near'
            self.fm_catalog_status.setText(
                f"{head}: {band}≈{mag:.1f} (rough — {label}, no {band} in "
                f"catalogue)")
        # a reddened IR-only star can look usable here (bright J -> bright R
        # guess) yet be optically invisible; append the standing warning
        warn = self._reddening_warning(star["mags"], band)
        if warn:
            self.fm_catalog_status.setText(
                self.fm_catalog_status.text() + f"   ⚠ {warn}")

    def _draw_catalog_stars(self, ax, transform=None):
        """Plot the loaded catalogue stars as guide-star candidates, SIZED by
        their brightness in the current TT sensing band (brighter = bigger, the
        better guide stars) so they read over the viridis heat map without a
        clashing colour, and RED-OUTLINED (FM_C_CATSTAR) so they stay visible
        over a loaded grayscale backdrop image too; stars fainter than the
        sensor's practical guide limit (or with no derivable sensing
        magnitude) are drawn dimmer (lower alpha, same red hue), so the
        usable candidates stand out. Every marker is UNFILLED (edge/ring only)
        -- a filled dot, however small, sits on top of and hides the real star
        underneath a sky backdrop, which defeats the point of overlaying a
        backdrop at all. `transform` (default ax.transData) is the field-PA
        rotation, if any -- see _fm_rotation_transform. One scatter for all
        of them (cheap). Returns the artists so the interactive renderer can
        remove them."""
        stars = self._catalog_stars_xy()
        if not stars:
            return []
        import matplotlib.colors as mcolors
        trans = transform if transform is not None else ax.transData
        band = self._tt_sensor_band()
        limit = engine.SENSOR_FAINT_LIMIT[band]
        faint_ec = mcolors.to_rgba(FM_C_CATSTAR, alpha=0.45)   # dimmer, same hue
        # A star the TSS cannot reach is not a guide-star candidate at all,
        # however bright (KAON 913 / engine.vignetting). Only meaningful for
        # the TT sensor, so it keys off the sensing band's mode via the Mode
        # combo, not off the star.
        tt_mode = self.fm_mode.currentText() != "NGS"
        show_tss = tt_mode and getattr(self, "fm_tss", None) is not None \
            and self.fm_tss.isChecked()
        xs, ys, sizes, ecs = [], [], [], []
        unreach_x, unreach_y = [], []
        for s in stars:
            mag, _kind, _lab = engine.estimate_sensing_mag(s["mags"], band)
            xs.append(s["x"]); ys.append(s["y"])
            if show_tss and not engine.tss_reachable(
                    float(np.hypot(s["x"], s["y"])))[0]:
                unreach_x.append(s["x"]); unreach_y.append(s["y"])
            if mag is None or mag > limit:             # too faint / unknown
                sizes.append(_CAT_SIZE_MIN)
                ecs.append(faint_ec)
            else:                                      # usable: bigger = brighter
                f = (mag - _CAT_MAG_BRIGHT) / (limit - _CAT_MAG_BRIGHT)
                f = min(max(f, 0.0), 1.0)
                sizes.append(_CAT_SIZE_MAX - f * (_CAT_SIZE_MAX - _CAT_SIZE_MIN))
                ecs.append(FM_C_CATSTAR)
        arts = [ax.scatter(xs, ys, s=sizes, facecolors=["none"] * len(xs),
                           edgecolors=ecs, linewidths=0.9, zorder=6,
                           transform=trans,
                           label=f"{self._catalog_name} "
                                 f"({len(stars)}, size∝{band}-mag)")]
        if unreach_x:
            # a cross through it: "the stage cannot go here", distinct from
            # the faint-star dimming, which is about photons, not geometry
            arts.append(ax.scatter(
                unreach_x, unreach_y, s=_CAT_SIZE_MAX * 1.3, marker="x",
                c=FM_C_TSS, linewidths=1.3, zorder=6.5, transform=trans,
                label=f"outside TSS travel ({len(unreach_x)})"))
        # ring the left-click-inspected star -- cyan, so it still stands out
        # against the (now red) markers themselves
        if self._catalog_inspected is not None:
            hit = next((s for s in stars
                        if s["id"] == self._catalog_inspected), None)
            if hit is not None:
                arts += ax.plot(hit["x"], hit["y"], "o", ms=14, mfc="none",
                                mec=FM_C_CATSTAR_RING, mew=1.8, ls="", zorder=6,
                                transform=trans)
        return arts

    # ---- guide-star auto-ranking --------------------------------------------
    def _invalidate_gs_ranking(self):
        """Clear a stale ranking (and its badges/dialog) whenever something
        that would change it changes: catalogue reload/clear, sensor switch,
        Mode/Metric, or any other field-map input. Cheap enough to just
        require a fresh Rank click rather than track exactly what changed.
        Guarded with getattr: the LGS tab builds (and can fire a sensor-
        change signal) BEFORE the field-map tab creates _gs_ranking."""
        if not getattr(self, "_gs_ranking", None):
            return
        self._gs_ranking = []
        if self._gs_rank_dialog is not None:
            self._gs_rank_dialog.close()  # triggers _on_gs_rank_dialog_closed

    def _rank_guide_stars(self):
        """Rank button: evaluate every loaded catalogue star as the guide/TT
        reference for the CURRENT Mode/Metric, delivered AT THE SCIENCE
        TARGET (the field centre). Badges the top 3 on the map and opens the
        full ranked table."""
        stars = self._catalog_stars_xy()
        if not stars or self.prep is None or self.res is None:
            return
        if self.pred_enable.isChecked():
            snap = self._pred_snapshot()
        else:
            when, t_hst = self._fm_when_time()
            snap = engine.field_snapshot(self.args_cached, self.prep, self.res,
                                         when, t_hst)
        if snap is None:
            self.fm_catalog_status.setText(
                "field map needs MASS profiles (none this night) — can't rank")
            return
        mode = {"NGS": "ngs", "single-LGS": "single",
               "LTAO": "ltao"}[self.fm_mode.currentText()]
        metric = {"Strehl": "strehl", "FWHM (half-max)": "fwhm",
                 "FWHM (Gaussian fit)": "fwhm_gaussfit",
                 "FWHM (Gaussian fit +background)": "fwhm_gaussfit_sky",
                 "FWHM (as the SR tool reads it)": "fwhm_srtool"}[
                     self.fm_metric.currentText()]
        sensor = self._tt_sensor_band()
        laser_xy = self._laser_xy()
        fm_dvar = (self._ngs_delta_var(self.last_offsets, self.args_cached)
                  if (mode == "ngs" and self.last_offsets) else 0.0)
        with engine.budget_overrides(**self.last_offsets):
            ranked = engine.rank_guide_stars(
                self.args_cached, self.prep, snap, mode, stars, laser_xy,
                sensor, metric=metric, ngs_delta_var=fm_dvar)
        self._gs_ranking = ranked
        n_ok = sum(1 for e in ranked if e["rank"] is not None)
        self.fm_catalog_status.setText(
            f"ranked {n_ok}/{len(ranked)} usable for {mode}/{metric} "
            f"({sensor}-band) — top 3 badged; table opened")
        self._show_gs_rank_dialog(ranked, mode, metric)
        self._fieldmap_dirty = True
        self._render_field_map_if_visible()

    def _show_gs_rank_dialog(self, ranked, mode, metric):
        """Non-modal table of the ranking, so the observer can keep working
        the map (right-click, zoom, left-click-inspect) while comparing
        candidates. Recreated on every Rank click rather than updated in
        place -- simpler, and a stale ranking is closed by
        _invalidate_gs_ranking anyway."""
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        is_fwhm = metric != "strehl"
        role = "ngs" if mode == "ngs" else "tt"
        if self._gs_rank_dialog is not None:
            self._gs_rank_dialog.close()
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Guide-star ranking — {self.fm_mode.currentText()} "
                          f"/ {self.fm_metric.currentText()}")
        dlg.resize(700, 360)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(QtWidgets.QLabel(
            "Delivered performance AT THE SCIENCE TARGET if each star were "
            "the guide/TT-tilt star. Double-click a row to select that star; "
            "left-click a star on the map to highlight its row here. Click "
            "a column header to sort."))
        cols = ["#", "id", "RA", "Dec", "mag", "offset (″)",
               "FWHM (mas)" if is_fwhm else "Strehl", "status"]
        table = QtWidgets.QTableWidget(len(ranked), len(cols))
        table.setHorizontalHeaderLabels(cols)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        for i, e in enumerate(ranked):
            rank_s = str(e["rank"]) if e["rank"] is not None else "—"
            c = SkyCoord(e["ra"] * u.deg, e["dec"] * u.deg)
            ra_s = c.ra.to_string(unit="hourangle", sep="hms", precision=1)
            dec_s = c.dec.to_string(unit="deg", sep="dms", precision=0,
                                    alwayssign=True)
            mag_s = "—" if e["mag"] is None else (
                f"{e['mag']:.1f}" + ("" if e["mag_kind"] == "exact" else "*"))
            val_s = "—" if e["delivered_value"] is None else (
                f"{e['delivered_value']:.0f}" if is_fwhm
                else f"{e['delivered_value']:.3f}")
            # a kept-but-reddened star gets the flag in its status too, not
            # just excluded ones; either way it reads amber (verify, don't
            # assume) rather than the default text colour
            if e["excluded_reason"]:
                status = e["excluded_reason"]
            elif e.get("reddening_note"):
                status = f"⚠ reddened ({e['reddening_note']}) — verify"
            else:
                status = "ok"
            warn = bool(e.get("reddening_note")) or (
                e["excluded_reason"] is not None
                and "IR-red" in e["excluded_reason"])
            # SortableItem keys make every column sortable by VALUE (text
            # sorting puts rank 10 before 9 and scatters the "—" rows);
            # excluded rows (rank None) always sort to the far end. The '#'
            # item carries the ranked-list index (UserRole): once sorted,
            # the visual row no longer equals the ranked index, so the
            # click/highlight handlers must resolve through it, never
            # index `ranked` by row.
            items = [SortableItem(rank_s, e["rank"]),
                     QtWidgets.QTableWidgetItem(e["id"]),
                     SortableItem(ra_s, e["ra"]),
                     SortableItem(dec_s, e["dec"]),
                     SortableItem(mag_s, e["mag"]),
                     SortableItem(f"{e['offset_arcsec']:.1f}",
                                  e["offset_arcsec"]),
                     SortableItem(val_s, e["delivered_value"]),
                     QtWidgets.QTableWidgetItem(status)]
            items[0].setData(Qt.ItemDataRole.UserRole, i)
            for j, item in enumerate(items):
                if j == len(cols) - 1 and warn:        # status column, amber
                    item.setForeground(QtGui.QColor(FM_C_WARN))
                elif e["rank"] is not None and e["rank"] <= 3:
                    item.setForeground(QtGui.QColor(FM_C_CATSTAR_RING))
                table.setItem(i, j, item)
        table.resizeColumnsToContents()
        # only AFTER populating -- live sorting would re-order the rows out
        # from under the population loop's setItem(i, ...). Enabling applies
        # Qt's default indicator (column 0 DESCENDING, i.e. worst-first) --
        # sort ascending explicitly to open in rank order, excluded last
        table.setSortingEnabled(True)
        table.sortItems(0, Qt.SortOrder.AscendingOrder)
        # single-click a row -> highlight that star on the MAP (the reverse of
        # a map left-click highlighting the row); double-click still SELECTS it
        # as the guide star. cellClicked fires only on real user clicks, not on
        # the programmatic selectRow() used by map->table highlighting, so the
        # two directions can't feed back into each other.
        table.cellClicked.connect(
            lambda row, _col: self._gs_rank_row_clicked(ranked, row))
        table.cellDoubleClicked.connect(
            lambda row, _col: self._gs_rank_select(ranked, row, role))
        lay.addWidget(table, 1)
        note = QtWidgets.QLabel(
            "mag* = estimated/nearest-band, not the sensor's own band. "
            "Excluded stars (no magnitude, too faint, or optically reddened "
            "for the sensor) sort last with a status reason. Amber = verify, "
            "don't assume (e.g. an IR-red star's optical mag is unreliable).")
        note.setStyleSheet("QLabel { font-size:11px; }")
        set_cue(note, "secondary")
        note.setWordWrap(True)
        lay.addWidget(note)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(dlg.close)
        lay.addWidget(close)
        # however it closes (this button, the window's X, or
        # _invalidate_gs_ranking) drop the Qt references so a later
        # left-click-highlight can never touch a deleted table/dialog
        dlg.finished.connect(lambda *_: self._on_gs_rank_dialog_closed())
        dlg.show()
        self._gs_rank_dialog = dlg
        self._gs_rank_table = table
        self._gs_rank_highlight(self._catalog_inspected)

    def _on_gs_rank_dialog_closed(self):
        self._gs_rank_dialog = None
        self._gs_rank_table = None

    def _gs_rank_entry_at(self, ranked, row):
        """The ranked entry behind VISUAL table row `row`. The table is
        user-sortable, so the visual row is mapped back to the ranked-list
        index stored on the '#' item (UserRole) -- never `ranked[row]`.
        Falls back to positional indexing only if the table is gone (a
        queued click racing the dialog close)."""
        table = getattr(self, "_gs_rank_table", None)
        if table is not None and 0 <= row < table.rowCount():
            idx = table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            if idx is not None and 0 <= idx < len(ranked):
                return ranked[idx]
            return None
        return ranked[row] if 0 <= row < len(ranked) else None

    def _gs_rank_highlight(self, star_id):
        """Select (highlight only -- not "set as guide star") the ranking
        table row matching star_id, if the table is open and that star is in
        it. star_id=None (or not found, e.g. it was excluded from a
        DIFFERENT ranking than the one currently open) clears the selection.
        Found by scanning the id column live: a build-time id->row dict
        would go stale on every header-click sort."""
        table = getattr(self, "_gs_rank_table", None)
        if table is None:
            return
        if star_id is not None:
            for row in range(table.rowCount()):
                if table.item(row, 1).text() == star_id:
                    table.selectRow(row)
                    table.scrollToItem(table.item(row, 0))
                    return
        table.clearSelection()

    def _gs_rank_row_clicked(self, ranked, row):
        """Single-clicked table row: HIGHLIGHT that star on the map (ring it,
        report its mags) -- the reverse of a map left-click highlighting the
        row. Highlight only; double-click still selects it as the guide star.
        Works for excluded rows too (they are still catalogue stars on the
        map), which is useful for locating a flagged/too-faint candidate."""
        e = self._gs_rank_entry_at(ranked, row)
        if e is not None:
            self._inspect_star({"id": e["id"], "mags": e["mags"]})

    def _gs_rank_select(self, ranked, row, role):
        """Double-clicked table row: select that star as the TT/NGS guide
        star, exactly like right-click "Set as TT/NGS star" on the map."""
        e = self._gs_rank_entry_at(ranked, row)
        if e is None:
            return
        if e["rank"] is None:
            self.fm_catalog_status.setText(
                f"“{e['id']}” is excluded ({e['excluded_reason']}) — not "
                f"selected")
            return
        star = {"id": e["id"], "ra": e["ra"], "dec": e["dec"], "mags": e["mags"]}
        self._fm_select_star(star, role)

    def _draw_gs_ranking_badges(self, ax, transform=None):
        """Badge the top-3 ranked guide stars with their rank number, cyan
        (FM_C_CATSTAR_RING) like the left-click inspect ring so it reads over
        both the heat map and a loaded backdrop. [] once _invalidate_gs_ranking
        has cleared a stale ranking. `transform` (default ax.transData) is the
        field-PA rotation, if any. Returns the artists created."""
        trans = transform if transform is not None else ax.transData
        arts = []
        for e in self._gs_ranking[:3]:
            if e["rank"] is None:
                continue
            arts.append(ax.annotate(
                str(e["rank"]), xy=(e["x"], e["y"]), xycoords=trans,
                textcoords="offset points", xytext=(-10, 10), zorder=9,
                fontsize=9, fontweight="bold", color="black",
                ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.18", fc=FM_C_CATSTAR_RING,
                          ec="black", lw=0.6)))
        return arts

    # ---- TSS reachability + vignetting overlay (KAON 913) -------------------
    def _draw_tss_vignetting(self, ax, mode, transform=None):
        """Draw where the tip-tilt sensor can actually GO, and how much light
        it loses getting there (engine.vignetting; KAON 913).

        Eduardo 2026-08-07 asked for visual feedback to go with the ranking
        change. Three things are drawn, and they say three different things:

          * the GUARANTEED circle (solid) -- inside it the TSS reaches the
            star at every rotator angle AND the rotator's own 60" unvignetted
            radius is not exceeded. Nothing to think about inside here.
          * the TRAVEL LIMIT (dashed) -- the stage box's longest reach. The
            band between the two is the honest "depends on the bench angle"
            region: the travel box is asymmetric and lives in the BENCH
            frame, and this app does not carry the bench-to-sky rotation, so
            drawing a rotated box would be inventing an angle. The band is
            what can be said without one.
          * vignetting contours (dotted, labelled) -- the MODEL, at the same
            levels the ranking charges as lost flux.

        Returns the artists (the caller collects them as dynamic overlay).
        Empty for NGS mode: the NGS star is sensed on the high-order WFS, not
        through the TSS that KAON 913 measured.
        """
        import keck_ao_estimator as engine
        if mode == "ngs" or not getattr(self, "fm_tss", None) \
                or not self.fm_tss.isChecked():
            return []
        from matplotlib.patches import Circle
        kw = {} if transform is None else {"transform": transform}
        out = []
        r_ok = engine.TSS_INSCRIBED_ARCSEC
        r_max = engine.TSS_CIRCUMSCRIBED_ARCSEC
        # the "depends on the rotator" band, as a filled annulus
        out.append(ax.add_patch(Circle((0, 0), r_max, fill=True, lw=0,
                                       fc=FM_C_TSS, alpha=0.10, zorder=1.5,
                                       **kw)))
        out.append(ax.add_patch(Circle((0, 0), r_ok, fill=True, lw=0,
                                       fc="white", alpha=0.0, zorder=1.6,
                                       **kw)))
        out.append(ax.add_patch(Circle((0, 0), r_ok, fill=False, ls="-",
                                       lw=1.4, ec=FM_C_TSS, alpha=0.95,
                                       zorder=4, **kw)))
        out.append(ax.add_patch(Circle((0, 0), r_max, fill=False, ls="--",
                                       lw=1.3, ec=FM_C_TSS, alpha=0.85,
                                       zorder=4, **kw)))
        out.append(ax.annotate(
            f"TSS reach {r_ok:.0f}″", xy=(0, r_ok), xytext=(0, 3),
            textcoords="offset points", ha="center", va="bottom", fontsize=7,
            color=FM_C_TSS, zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7,
                      ec="none")))
        out.append(ax.annotate(
            f"stage travel {r_max:.0f}″ (rotator-dependent)", xy=(0, r_max),
            xytext=(0, 3), textcoords="offset points", ha="center",
            va="bottom", fontsize=7, color=FM_C_TSS, zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7,
                      ec="none")))
        # vignetting contours at the levels the ranking actually charges
        for frac in (0.05, 0.15, 0.30):
            r = engine.UNVIGNETTED_RADIUS_ARCSEC * (
                frac / engine.VIGNETTE_COEFF) ** (1.0 / engine.VIGNETTE_EXP)
            if r > r_max:
                continue
            out.append(ax.add_patch(Circle((0, 0), r, fill=False, ls=":",
                                           lw=1.0, ec=FM_C_TSS, alpha=0.7,
                                           zorder=4, **kw)))
            out.append(ax.annotate(
                f"{100 * frac:.0f}%", xy=(r * 0.707, -r * 0.707),
                xytext=(2, -2), textcoords="offset points", ha="left",
                va="top", fontsize=7, color=FM_C_TSS, zorder=5))
        out.append(ax.annotate(
            "TSS vignetting: MODEL (KAON 913, 3 points)", xy=(0.985, 0.985),
            xycoords="axes fraction", ha="right", va="top", fontsize=7,
            color=FM_C_TSS, zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75,
                      ec=FM_C_TSS)))
        return out
