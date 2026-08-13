"""Small reusable widgets and widget-building helpers whose defaults come
from the parser: OffsetEntry (angular-offset entry, three input modes),
_dspin (a QDoubleSpinBox factory), _shrinkable_label, and SortableItem
(numeric-keyed table cell for user-sortable QTableWidgets).
"""
import numpy as np
import astropy.units as u

from qtcompat import QtCore, QtWidgets, Qt, Signal

import keck_ao_estimator as engine

from .theme import set_cue

# The "guide star = target" default for a TT/NGS OffsetEntry: on-axis, zero
# offset. Shared so every "define a new target" path (Resolve, Save, a
# starlist pick with no linked candidate, per-target restore) resets to the
# SAME thing rather than each re-deriving its own on-axis dict -- always
# copy it (dict(_ON_AXIS_OFFSET_CFG)), never hand out the module-level dict
# itself, since OffsetEntry.set_config()/callers must not mutate a shared
# object.
_ON_AXIS_OFFSET_CFG = {"mode": 0, "total": 0.0, "pa": 0.0, "dra": 0.0,
                       "ddec": 0.0, "sra": "", "sdec": ""}


def _shrinkable_label(label):
    """Let a status/readout QLabel hold long text without forcing the whole
    window wider than the screen: it may shrink to nothing (text clips) and the
    full string is mirrored into its tooltip. Wraps setText to keep the tooltip
    current."""
    label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                        QtWidgets.QSizePolicy.Policy.Preferred)
    label.setMinimumWidth(0)
    _orig = label.setText

    def _set(text):
        _orig(text)
        label.setToolTip(text)
    label.setText = _set
    return label


class TimeEdit(QtWidgets.QLineEdit):
    """Free-typing replacement for QTimeEdit (API-compatible where the app
    uses it: time()/setTime()/timeChanged/setDisplayFormat). QTimeEdit's
    section-based editing means you can only overtype hours OR minutes --
    you cannot select-all, clear, and just type a new time (Eduardo,
    2026-07-23). This is a plain line edit: type "21:35", "2135", "935",
    or "9" (bare hour) and it commits on Enter/focus-out; garbage reverts
    to the last good value instead of erroring."""
    timeChanged = Signal(object)          # emits the new QtCore.QTime

    def __init__(self, parent=None):
        super().__init__(parent)
        self._time = QtCore.QTime(0, 0)
        super().setText("00:00")
        self.setMaximumWidth(70)
        self.setPlaceholderText("HH:MM")
        self.setToolTip('Type a time as "21:35", "2135", "935", or a bare '
                        'hour "9" — commits on Enter/focus-out; an '
                        'unparseable entry reverts.')
        self.editingFinished.connect(self._commit)

    @staticmethod
    def _parse(text):
        s = text.strip().replace(".", ":")
        try:
            if ":" in s:
                hh, mm = s.split(":", 1)
                h, m = int(hh), int(mm or 0)
            elif s.isdigit() and 1 <= len(s) <= 4:
                if len(s) <= 2:
                    h, m = int(s), 0
                else:
                    h, m = int(s[:-2]), int(s[-2:])
            else:
                return None
        except ValueError:
            return None
        if 0 <= h <= 23 and 0 <= m <= 59:
            return QtCore.QTime(h, m)
        return None

    def _commit(self):
        t = self._parse(self.text())
        if t is None:
            super().setText(self._time.toString("HH:mm"))   # revert
            return
        changed = t != self._time
        self._time = t
        super().setText(t.toString("HH:mm"))
        if changed:
            self.timeChanged.emit(t)

    def time(self):
        return QtCore.QTime(self._time)

    def setTime(self, t):
        if not isinstance(t, QtCore.QTime) or not t.isValid():
            return
        changed = t != self._time
        self._time = QtCore.QTime(t)
        super().setText(self._time.toString("HH:mm"))
        if changed:
            self.timeChanged.emit(QtCore.QTime(t))

    def setDisplayFormat(self, _fmt):
        pass                              # QTimeEdit-API compatibility


class SortableItem(QtWidgets.QTableWidgetItem):
    """Table cell that sorts by an explicit numeric key instead of its
    display text -- text sorting puts "10" before "9", scrambles "—"
    placeholders among magnitudes, and orders RA/Dec strings only by
    accident. key=None means "no value" and sorts as +inf: last ascending
    (and consequently first descending -- acceptable, standard behaviour).
    Text-only columns should keep plain QTableWidgetItem. Used by the
    guide-star-ranking and starlist picker dialogs' sortable tables."""

    def __init__(self, text, key=None):
        super().__init__(text)
        self.sort_key = float("inf") if key is None else float(key)

    def __lt__(self, other):
        return self.sort_key < getattr(other, "sort_key", float("inf"))


def _dspin(lo, hi, step, val, decimals=2, suffix=""):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(decimals)
    # Qt's keyboardTracking defaults ON, which emits valueChanged on EVERY
    # keystroke -- deleting one character of "8.0" fires a recompute for "8",
    # then ".", etc. Off means valueChanged fires once, when the edit is
    # committed (Enter / focus-out) or the arrows/scroll are used.
    w.setKeyboardTracking(False)
    if suffix:
        w.setSuffix(suffix)
    if val is not None:
        w.setValue(float(val))
    return w


class OffsetEntry(QtWidgets.QWidget):
    """Enter an angular offset from the science target three ways, all resolving
    to the scalar arcsec the engine takes:

      0. total offset in arcsec (the original single number);
      1. sky-projected components (ΔRA, ΔDec) in arcsec -> hypot;
      2. the offset star's RA/Dec -> angular separation from the science target
         (from the Target tab), so an observer with the star's coordinates but
         not the computed separation can enter them directly.

    Drop-in-ish for a spinbox: value()/setValue() operate on the resolved total,
    so existing call sites keep working; changed fires when the magnitude may
    have changed (drives the science recompute); pos_changed fires when only the
    direction changed (the total-mode PA -- field map only); ok() reports whether
    the current mode can be computed (mode 2 needs valid coordinates).

    fixable=True adds a "fix to base position" checkbox (modes 0/1 only --
    mode 2 is already an absolute position, nothing to fix): when checked, the
    CURRENTLY resolved absolute sky position is frozen as an anchor, and
    refresh_from_base() (call whenever science_getter's target may have moved,
    e.g. the Target tab's offset control changed) re-derives the displayed
    ΔRA/ΔDec needed to keep that anchor fixed -- so a guide star's real,
    unmoving sky position doesn't silently follow a target-offset exploration."""
    changed = Signal()
    pos_changed = Signal()

    MODES = ["total (arcsec)", "ΔRA, ΔDec (arcsec)", "star RA/Dec"]

    def __init__(self, default_total, science_getter, hi=120.0, parent=None,
                 fixable=False):
        super().__init__(parent)
        self._science = science_getter          # () -> (ra_str, dec_str)
        self._val, self._ok = float(default_total), True
        self._fixable = fixable
        self._anchor = None                      # SkyCoord while fixed, else None
        self._refreshing = False                 # reentrancy guard, see _recompute
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0); v.setSpacing(2)
        self.mode = QtWidgets.QComboBox(); self.mode.addItems(self.MODES)
        v.addWidget(self.mode)
        self.stack = QtWidgets.QStackedWidget()

        # total mode: magnitude + a position angle (N->E) for the field map, so
        # a one-value offset can still be placed directionally (default 0 = N).
        self.total = _dspin(0.0, hi, 0.5, default_total, 1, '"')
        self.pa = _dspin(0.0, 360.0, 5.0, 0.0, 0, "°")
        self.pa.setToolTip("Position angle (deg, North→East) for the field-map "
                           "marker when only a total offset is given.")
        self.stack.addWidget(self._page([("total:", self.total),
                                         ("PA (N→E):", self.pa)]))
        self.dra = _dspin(-hi, hi, 0.5, 0.0, 1, '"')
        self.ddec = _dspin(-hi, hi, 0.5, 0.0, 1, '"')
        self.stack.addWidget(self._page([("ΔRA (E):", self.dra),
                                         ("ΔDec (N):", self.ddec)]))
        self.sra = QtWidgets.QLineEdit(); self.sra.setPlaceholderText("15h49m57.7s")
        self.sra.setToolTip("hms (15h49m57.7s), colon hours (15:49:57.7), "
                            "or decimal degrees (237.49).")
        self.sdec = QtWidgets.QLineEdit(); self.sdec.setPlaceholderText("-03d55m16s")
        self.sdec.setToolTip("dms (-03d55m16s), colon degrees (-03:55:16), "
                             "or decimal degrees (-3.92).")
        self.stack.addWidget(self._page([("Star RA:", self.sra),
                                         ("Star Dec:", self.sdec)]))
        # QStackedWidget defaults to an EXPANDING vertical policy: in a form/
        # scroll layout it would grab all the leftover height (a large empty gap
        # under the control). Pin the stack -- and this compound widget -- to
        # their natural height so they take only what the current page needs.
        _fixed_v = QtWidgets.QSizePolicy.Policy.Fixed
        self.stack.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, _fixed_v)
        v.addWidget(self.stack)
        self.result = QtWidgets.QLabel()
        self.result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        _shrinkable_label(self.result)   # readout must never widen the panel
        v.addWidget(self.result)
        if self._fixable:
            self.fix_to_base = QtWidgets.QCheckBox("fix to base position")
            self.fix_to_base.setToolTip(
                "Keep this offset's ABSOLUTE sky position fixed as the Target "
                "tab's offset changes, instead of silently following it (so a "
                "real guide star's separation from the target doesn't need "
                "re-entering by hand every time you nudge the target offset). "
                "Total/ΔRA,ΔDec modes only -- star RA/Dec is already absolute.")
            v.addWidget(self.fix_to_base)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, _fixed_v)

        self.mode.currentIndexChanged.connect(self.stack.setCurrentIndex)
        self.mode.currentIndexChanged.connect(self._recompute)
        for sp in (self.total, self.dra, self.ddec):
            sp.valueChanged.connect(self._recompute)
        for le in (self.sra, self.sdec):
            le.textChanged.connect(self._recompute)
        self.pa.valueChanged.connect(self._pa_changed)   # direction only
        if self._fixable:
            self.fix_to_base.toggled.connect(self._on_fix_to_base_toggled)
        self._recompute()

    @staticmethod
    def _page(rows):
        p = QtWidgets.QWidget(); fl = QtWidgets.QFormLayout(p)
        fl.setContentsMargins(0, 0, 0, 0)
        for lbl, wdg in rows:
            fl.addRow(lbl, wdg)
        return p

    def _compute(self):
        m = self.mode.currentIndex()
        if m == 0:
            return float(self.total.value()), True
        if m == 1:
            return float(np.hypot(self.dra.value(), self.ddec.value())), True
        sra, sdec = self.sra.text().strip(), self.sdec.text().strip()
        tra, tdec = self._science()
        if not (sra and sdec and tra.strip() and tdec.strip()):
            return self._val, False
        try:
            sep = engine.parse_radec(tra, tdec).separation(
                engine.parse_radec(sra, sdec))
            return float(sep.arcsec), True
        except Exception:
            return self._val, False

    def _update_label(self):
        if self._ok:
            extra = (f"  @ PA {self.pa.value():g}°"
                     if self.mode.currentIndex() == 0 else "")
            self.result.setText(f"= {self._val:.2f}″ from target{extra}")
            set_cue(self.result, "ok")
        else:
            self.result.setText("= enter valid star + target coordinates")
            set_cue(self.result, "err")

    def _recompute(self, *_):
        self._val, self._ok = self._compute()
        self._update_label()
        # re-anchor on a genuine user edit while fixed (mode/total/dra/ddec
        # change): the just-edited offset becomes the new fixed position. NOT
        # while refresh_from_base() is itself the one setting dra/ddec, or
        # every refresh would re-derive the anchor from its own output --
        # harmless in principle (self-consistent to ~1e-6") but pointless and
        # a source of slow drift over many refreshes; the anchor only needs
        # to change because of a real edit.
        if self._fixable and self.fix_to_base.isChecked() and not self._refreshing:
            self._anchor = self._resolve_from_science()
        self.changed.emit()

    def _on_fix_to_base_toggled(self, checked):
        self._anchor = self._resolve_from_science() if checked else None
        self._recompute()

    def _resolve_from_science(self):
        """This offset resolved against the CURRENT science()/effective
        target, as an absolute SkyCoord -- or None if science() or this
        offset (mode 2) is currently unparseable."""
        try:
            tra, tdec = self._science()
            if not (tra.strip() and tdec.strip()):
                return None
            return self.resolved_skycoord(engine.parse_radec(tra, tdec))
        except Exception:
            return None

    def refresh_from_base(self):
        """Re-derive the displayed ΔRA/ΔDec from the CURRENT science()/
        effective target so a fixed (anchored) absolute position is
        preserved. Call whenever the target/target-offset may have moved.
        No-op unless fix_to_base is checked, in star-coordinate mode (already
        absolute -- nothing to refresh), or the anchor/current target is
        unavailable."""
        if not (self._fixable and self.fix_to_base.isChecked()):
            return
        if self._anchor is None or self.mode.currentIndex() == 2:
            return
        try:
            tra, tdec = self._science()
            if not (tra.strip() and tdec.strip()):
                return
            current = engine.parse_radec(tra, tdec)
            dlon, dlat = current.spherical_offsets_to(self._anchor)
        except Exception:
            return
        self._refreshing = True
        try:
            for w in (self.mode, self.dra, self.ddec):
                w.blockSignals(True)
            self.mode.setCurrentIndex(1)
            self.stack.setCurrentIndex(1)
            self.dra.setValue(float(dlon.arcsec))
            self.ddec.setValue(float(dlat.arcsec))
            for w in (self.mode, self.dra, self.ddec):
                w.blockSignals(False)
            self._recompute()
        finally:
            self._refreshing = False

    def _pa_changed(self, *_):
        self._update_label()
        self.pos_changed.emit()               # direction only, not the magnitude

    def offset_xy(self, center_radec=None):
        """The offset star's 2-D position for the field map, in the plot frame
        (x = West+, i.e. −East; y = North+). Returns (x, y, dir_known):
        ΔRA/ΔDec and star-coord modes give a real position (known); total mode
        uses the magnitude at the user-set PA (default North), flagged as an
        assumed direction (known=False).

        center_radec: (ra, dec) strings for the field centre in star-coord
        mode; None uses the science target. When a real image defines the field
        (its pointing), the field map passes that so the star position is
        measured from the image centre, not the typed target."""
        m = self.mode.currentIndex()
        if m == 1:                            # ΔRA (East), ΔDec (North) arcsec
            return -float(self.dra.value()), float(self.ddec.value()), True
        if m == 2:
            try:
                tra, tdec = center_radec if center_radec else self._science()
                dlon, dlat = engine.parse_radec(tra, tdec).spherical_offsets_to(
                    engine.parse_radec(self.sra.text().strip(),
                                       self.sdec.text().strip()))
                return -float(dlon.arcsec), float(dlat.arcsec), True
            except Exception:
                pass
        r = float(self._val); pa = np.radians(self.pa.value())   # total + PA
        return -r * np.sin(pa), r * np.cos(pa), False

    def resolved_skycoord(self, base_coord):
        """The actual SkyCoord after applying this offset to base_coord: for
        total/ΔRA,ΔDec mode, base_coord shifted by the resolved (ΔRA, ΔDec);
        for star RA/Dec mode, the typed absolute position directly (base_coord
        is then irrelevant). At the zero-offset default (mode 0, total=0"),
        returns base_coord UNCHANGED -- astropy's spherical_offsets_by(0, 0)
        is not bit-exact, so a never-touched offset control must short-circuit
        rather than round-trip through it, or a report run under the default
        (no offset) would drift off the byte-identical reference outputs."""
        if self.mode.currentIndex() == 2:
            return engine.parse_radec(self.sra.text().strip(),
                                      self.sdec.text().strip())
        x, y, _known = self.offset_xy()          # x = West+, y = North+
        if x == 0.0 and y == 0.0:
            return base_coord
        return base_coord.spherical_offsets_by(-x * u.arcsec, y * u.arcsec)

    # --- spinbox-like API + (de)serialization -------------------------------
    def value(self):
        return self._val

    def ok(self):
        return self._ok

    def setValue(self, v):                       # programmatic: total mode
        self.mode.setCurrentIndex(0)
        self.total.setValue(float(v))

    def get_config(self):
        c = dict(mode=self.mode.currentIndex(), total=self.total.value(),
                 pa=self.pa.value(), dra=self.dra.value(),
                 ddec=self.ddec.value(),
                 sra=self.sra.text(), sdec=self.sdec.text())
        if self._fixable:
            c["fix_to_base"] = self.fix_to_base.isChecked()
        return c

    def set_config(self, c):
        ws = (self.mode, self.total, self.pa, self.dra, self.ddec,
              self.sra, self.sdec)
        if self._fixable:
            ws = ws + (self.fix_to_base,)
        for wd in ws:
            wd.blockSignals(True)
        self.mode.setCurrentIndex(int(c.get("mode", 0)))
        self.stack.setCurrentIndex(self.mode.currentIndex())
        self.total.setValue(c.get("total", 0.0)); self.pa.setValue(c.get("pa", 0.0))
        self.dra.setValue(c.get("dra", 0.0)); self.ddec.setValue(c.get("ddec", 0.0))
        if self._fixable:
            self.fix_to_base.setChecked(bool(c.get("fix_to_base", False)))
        self.sra.setText(c.get("sra", "")); self.sdec.setText(c.get("sdec", ""))
        for wd in ws:
            wd.blockSignals(False)
        self._recompute()
