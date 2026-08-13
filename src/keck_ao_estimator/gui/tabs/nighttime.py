"""Nighttime mode: on an ACTIVE night, auto re-fetch tonight's MKWC DIMM/MASS/
MASSPRO data and re-run on a timer, so the estimate keeps up without the
observer re-clicking Run. Cross-cutting (Data-tab controls, the Run pipeline,
and a Field-map Conditions option), kept in its own mixin per the project's
compartmentalize-don't-grow-monoliths rule rather than folding it into
data.py/mainwindow.py/fieldmap_tab.py.

Deliberately NOT persisted in _collect_config/_apply_config: like the guide-
star catalogue's "Load" button, enabling nighttime mode is an ACTION (it
starts a background network timer), not a static preference -- a loaded
config should never silently start polling. Only the ordinary fetch-date/
mode settings it drives are persisted, via the existing keys.
"""
from datetime import datetime, timedelta, timezone

from qtcompat import QtCore, QtWidgets

import keck_ao_estimator as engine

from ..widgets import _shrinkable_label
from ..theme import set_cue

# MKWC publishes new DIMM/MASS/MASSPRO samples roughly every 5 minutes; this
# matches that cadence rather than hammering the archive faster than it
# updates.
NIGHTTIME_PULL_INTERVAL_MS = 5 * 60 * 1000


class NighttimeModeMixin:
    def _init_nighttime_mode(self, f):
        """Build the Nighttime-mode row and its timer. `f` is the Data tab's
        QFormLayout (called from DataTabMixin._tab_data)."""
        self._nighttime_last_pull = None      # HST datetime of the last
                                               # successful auto-pull, or None
        self._nighttime_next_pull = None      # HST datetime the timer will
                                               # next fire, or None
        self._nighttime_timer = QtCore.QTimer(self)
        self._nighttime_timer.setInterval(NIGHTTIME_PULL_INTERVAL_MS)
        self._nighttime_timer.timeout.connect(self._nighttime_pull)

        self.nighttime_enable = QtWidgets.QCheckBox("Nighttime mode")
        self.nighttime_enable.setMinimumWidth(100)   # floor (631045c)
        self.nighttime_enable.setToolTip(
            "For an ACTIVE night only: automatically re-fetches tonight's "
            "MKWC DIMM/MASS/MASSPRO data every 5 minutes (matching MKWC's "
            "publish cadence) and re-runs, so the estimate stays current "
            "without pressing Run. Forces fetch-by-date mode on tonight's "
            "UT date while enabled.")
        self.nighttime_enable.toggled.connect(self._on_nighttime_toggled)

        self.nighttime_pull_now = QtWidgets.QPushButton("Pull now")
        self.nighttime_pull_now.setEnabled(False)
        self.nighttime_pull_now.setToolTip(
            "Force an immediate re-fetch + re-run, and reset the 5-minute "
            "timer from now.")
        self.nighttime_pull_now.clicked.connect(
            lambda: self._nighttime_pull(forced=True))

        self.nighttime_status = QtWidgets.QLabel("not active")
        set_cue(self.nighttime_status, "secondary")
        # the "last pull … · next pull …" string is wide; a plain QLabel's
        # minimumSizeHint (full text width) is a hard QFormLayout floor, and
        # ONE over-wide row forces a horizontal scrollbar across the whole
        # tab (the 2026-07-20 631045c lesson). Shrinkable = may clip, full
        # text mirrored to the tooltip, same as the field-map status labels.
        _shrinkable_label(self.nighttime_status)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.nighttime_enable)
        row.addWidget(self.nighttime_pull_now)
        row.addStretch(1)
        f.addRow("", self._wrap(row))
        # the status gets its OWN row below the controls: inline it only got
        # the width left over after the checkbox + button and visibly clipped.
        # Alone it spans the full field column, and with word-wrap it can
        # never clip at ANY font size -- it shows one line normally and wraps
        # to two if squeezed (shrinkable above stays on so its text still
        # contributes zero width floor; don't trust font-size arithmetic
        # here, that's what shipped the clipped inline version)
        self.nighttime_status.setWordWrap(True)
        f.addRow("", self.nighttime_status)

    def _nighttime_is_night(self):
        """The day/night gate (engine.is_night_at_keck: sun below the
        standard -0.833 deg rise/set horizon at Keck). A one-line wrapper so
        the offline tests can stub the gate instead of the astronomy."""
        return engine.is_night_at_keck()

    def _nighttime_refuse(self, message):
        """Back out of an enable attempt (or a running session) without
        recursing through the toggle handler, and say why."""
        self.nighttime_enable.blockSignals(True)
        self.nighttime_enable.setChecked(False)
        self.nighttime_enable.blockSignals(False)
        # run the normal disable path for the side effects (timer, source
        # controls, theme handback), then overwrite the status with the reason
        self._on_nighttime_toggled(False)
        self.nighttime_status.setText(message)

    def _on_nighttime_toggled(self, checked):
        """Enabling forces fetch-by-date mode on tonight's UT date and does
        one immediate pull, then the 5-minute timer takes over. Disabling
        just stops the timer -- whatever data is currently loaded stays
        loaded, and manual control of the data source returns.

        Safety: only enables between sunset and sunrise at Keck -- MKWC only
        publishes seeing data at night, so a daytime enable would poll the
        archive every 5 minutes for nothing."""
        if checked and not self._nighttime_is_night():
            self._nighttime_refuse(
                "daytime at Keck — nighttime mode runs between sunset "
                "and sunrise")
            return
        self.nighttime_pull_now.setEnabled(checked)
        if not checked:
            self._nighttime_timer.stop()
            self._nighttime_next_pull = None
            for w in (self.mode_fetch, self.mode_local, self.fetch_date):
                w.setEnabled(True)
            # hand the theme back ONLY if our auto-switch still owns it (the
            # user didn't touch View ▸ Dark theme meanwhile)
            if self._dark_auto and self.dark_action.isChecked():
                self._sync_dark(False)
            self._dark_auto = False
            self._update_nighttime_status()
            return
        self.mode_fetch.setChecked(True)
        self._on_data_mode()
        # nighttime mode owns the data source while it's running -- disable
        # AFTER _on_data_mode(), which itself re-enables fetch_date for fetch
        # mode and would otherwise silently undo this
        for w in (self.mode_fetch, self.mode_local, self.fetch_date):
            w.setEnabled(False)
        # auto dark theme: only if the user hasn't already chosen dark
        # themselves -- then it's theirs and we neither claim it now nor
        # revert it on disable
        if not self.dark_action.isChecked():
            self._sync_dark(True)
            self._dark_auto = True
        self._nighttime_pull(forced=True)

    @staticmethod
    def _nighttime_now_hst():
        return (datetime.now(timezone.utc).replace(tzinfo=None)
                - timedelta(hours=engine.HST_TO_UTC_HOURS))

    def _nighttime_pull(self, forced=False):
        """(Re)point fetch_date at tonight's UT date and re-run. Safe to call
        while a run is already in flight: on_run() is a no-op then (self.worker
        is not None), and the timer will catch up next cycle. Always (re)arms
        the timer for a full interval from now, so a forced pull doesn't leave
        a short leftover gap before the next automatic one.

        Safety: every tick re-checks the day/night gate, so a session left
        running past dawn shuts itself off (within one 5-minute cycle of
        sunrise) instead of polling MKWC all day."""
        if not self.nighttime_enable.isChecked():
            return
        if not self._nighttime_is_night():
            self._nighttime_refuse(
                f"sunrise — nighttime mode stopped "
                f"({self._fmt_hm(self._nighttime_now_hst())})")
            return
        ut_today = QtCore.QDateTime.currentDateTimeUtc().date()
        self.fetch_date.setDate(ut_today)
        self._nighttime_timer.start()
        self._nighttime_next_pull = (self._nighttime_now_hst()
                                     + timedelta(milliseconds=NIGHTTIME_PULL_INTERVAL_MS))
        self._update_nighttime_status()
        self.on_run()

    def _on_nighttime_pull_done(self):
        """Call from _on_prepared, BEFORE recompute_and_draw(), whenever
        nighttime mode is active: records the successful pull's timestamp so
        a field map showing NIGHTTIME_FM_COND picks it up on the redraw that
        recompute_and_draw() is about to trigger."""
        self._nighttime_last_pull = self._nighttime_now_hst()
        self._update_nighttime_status()

    def _on_nighttime_pull_failed(self):
        """Call from _on_failed whenever nighttime mode is active. A failed
        FIRST pull (no successful pull yet this session -- typically MKWC
        simply has no DIMM file for tonight so early in the evening) disarms
        the mode and says to retry later, rather than re-hammering the
        archive every 5 minutes for a file that isn't there. Once a pull HAS
        succeeded, a later failure is treated as transient (a network blip,
        a mid-append fetch): stay armed, the timer retries next cycle."""
        if self._nighttime_last_pull is not None:
            return
        self._nighttime_refuse(
            "no DIMM data for tonight yet — nighttime mode disabled; "
            "try again later")

    def _update_nighttime_status(self):
        # one trailing zone tag (both times share it) keeps the string
        # compact enough for its full-width row at real font sizes;
        # _fmt_hm honours UTC mode
        def fmt(dt):
            if dt is None:
                return "—"
            return self._fmt_hm(dt, seconds=True).rsplit(" ", 1)[0]
        if not self.nighttime_enable.isChecked() and self._nighttime_last_pull is None:
            self.nighttime_status.setText("not active")
            return
        zone = "UTC" if self._utc() else "HST"
        self.nighttime_status.setText(
            f"last pull {fmt(self._nighttime_last_pull)} · "
            f"next pull {fmt(self._nighttime_next_pull)} {zone}")

    def _nighttime_fm_time(self):
        """(when, t_hst) for the field-map Conditions selector's
        NIGHTTIME_FM_COND option: the last pull's HST timestamp if one has
        happened, else a graceful whole-night-median fallback (same fallback
        _fm_when_time already uses when a picked selector has nothing to key
        off -- e.g. 'observing window' with no window set)."""
        if self._nighttime_last_pull is not None:
            return "time", self._nighttime_last_pull
        return "night", None


__all__ = ["NighttimeModeMixin", "NIGHTTIME_PULL_INTERVAL_MS"]
