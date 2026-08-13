#!/usr/bin/env python3
"""Nighttime mode: auto-fetch tonight's MKWC data on a timer (gui/tabs/
nighttime.py NighttimeModeMixin), the Data-tab status/force-pull controls, and
the field map's "time of last pull" Conditions option.

The real MKWC fetch (on_run -> PrepareWorker -> fetch_mkwc_files) is a network
call and stays OUT of this offline suite, same policy as the Vizier catalogue
lookups (gui_phase20/23): on_run is monkeypatched to a counting stub while
exercising the timer/widget wiring (including a genuine QTimer firing, sped up
so the test doesn't wait 5 real minutes), and the _on_prepared "pull completed"
hook is exercised against a REAL prep obtained the normal offline way
(mode_local + the bundled test data), not nighttime mode's own (network) path.
Run headless.
"""
import os, sys, time
from datetime import datetime, timedelta, timezone
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
from qtcompat import QtWidgets, QtCore
import keck_ao_estimator as engine
import keck_ao_estimator.gui as gui
DATA = os.path.join(HERE, "data")


def pump(cond, timeout=90):
    """Nested-QEventLoop wait (not processEvents+msleep) -- see gui_phase12's
    pump() docstring for why: msleep() doesn't service the message queue, so
    it can miss a QTimer callback for well longer than the timer's interval."""
    app = QtWidgets.QApplication.instance()
    loop = QtCore.QEventLoop()
    t0 = time.time()

    done = False

    def check():
        # A stale re-arm MUST be inert. The last singleShot(10, check) is
        # armed BEFORE cond() goes true, so it survives loop.quit() and
        # fires after pump() has returned -- by which time `loop` may be
        # garbage-collected, and calling quit() on a deleted C++
        # QEventLoop segfaults. That is the SIGSEGV (-11) that failed
        # gui_phase24 on CI 2026-07-28: intermittent, load-dependent, and
        # not reproducible on an idle box.
        nonlocal done
        if done:
            return
        if cond() or time.time() - t0 > timeout:
            done = True
            loop.quit()
        else:
            QtCore.QTimer.singleShot(10, check)
    QtCore.QTimer.singleShot(0, check)
    loop.exec()
    done = True          # neutralise any callback still in flight


def main():
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.resize(1500, 950); win.show(); app.processEvents()

    # pin the day/night gate OPEN for the wiring tests below -- they run at
    # any wall-clock time (CI included), and the real gate would refuse a
    # daytime enable. The gate itself is tested explicitly further down.
    win._nighttime_is_night = lambda: True

    # interval matches MKWC's ~5-minute publish cadence
    assert gui.NIGHTTIME_PULL_INTERVAL_MS == 5 * 60 * 1000
    assert win._nighttime_timer.interval() == gui.NIGHTTIME_PULL_INTERVAL_MS
    assert not win._nighttime_timer.isActive()
    assert win.nighttime_status.text() == "not active"
    assert not win.nighttime_pull_now.isEnabled()
    print("  [ok] nighttime timer configured (5 min), inactive, status "
          "'not active' before enabling")

    # layout (2026-07-21 post-release fixes, in order): the status label must
    # (a) contribute NO width floor (shipped as a plain QLabel it re-widened
    # the Data tab into a horizontal scrollbar -- the gui_phase18 floor guard
    # covers that half), and (b) live on its OWN form row with word-wrap --
    # inline after the checkbox+button it only got the leftover width and
    # visibly clipped the "last pull ... next pull ..." string; wrapping
    # (not clipping) is what makes it robust at any font size.
    lbl = win.nighttime_status
    assert lbl.wordWrap(), "status label must word-wrap, not clip"
    assert lbl.minimumWidth() == 0, "status label must stay width-shrinkable"
    row_box = win.nighttime_enable.parentWidget()
    assert lbl.parentWidget() is not row_box, \
        "status label must sit on its own form row (inline it clips after " \
        "the checkbox + button)"
    print("  [ok] status label: own wrapped row, zero width floor")

    # _nighttime_now_hst honours the HST = UTC-10 convention (engine.HST_TO_UTC_HOURS)
    want = (datetime.now(timezone.utc).replace(tzinfo=None)
            - timedelta(hours=engine.HST_TO_UTC_HOURS))
    got = win._nighttime_now_hst()
    assert abs((got - want).total_seconds()) < 2, (got, want)
    print("  [ok] _nighttime_now_hst uses HST = UTC-10")

    # --- enabling: no real network fetch -- stub on_run and count calls -----
    calls = []
    orig_on_run = win.on_run
    win.on_run = lambda: calls.append(1)

    win.mode_local.setChecked(True)
    win.nighttime_enable.setChecked(True); app.processEvents()

    ut_today = QtCore.QDateTime.currentDateTimeUtc().date().toString("yyyyMMdd")
    assert win.mode_fetch.isChecked(), "nighttime mode must force fetch-by-date"
    assert win.fetch_date.date().toString("yyyyMMdd") == ut_today, \
        "must point at tonight's UT date"
    assert not win.mode_fetch.isEnabled() and not win.mode_local.isEnabled() \
        and not win.fetch_date.isEnabled(), \
        "nighttime mode must own (disable) the data-source controls while active"
    assert win.nighttime_pull_now.isEnabled()
    assert win._nighttime_timer.isActive()
    assert len(calls) == 1, "enabling must do one immediate pull"
    print("  [ok] enabling forces fetch-by-date on tonight's UT date, disables "
          "the source controls, arms the timer, does one immediate pull")

    # --- "Pull now": forces another pull and keeps the timer running --------
    win.nighttime_pull_now.click(); app.processEvents()
    assert len(calls) == 2, "Pull now must trigger another pull"
    assert win._nighttime_timer.isActive(), \
        "Pull now must (re)arm the timer, not leave it stopped"
    assert not win.fetch_date.isEnabled(), \
        "the source controls must stay disabled through a forced pull " \
        "(regression: _on_data_mode() re-enables fetch_date for fetch mode " \
        "if the disable happens before it runs)"
    print("  [ok] 'Pull now' forces another pull and keeps the timer armed")

    # --- the QTimer really fires _nighttime_pull on its own -----------------
    # (not just via enable/"Pull now"): shrink the interval so the test
    # doesn't wait 5 real minutes, and let the ALREADY-CONNECTED
    # timeout -> _nighttime_pull -> on_run() chain fire for real. on_run is
    # still the stub from above, so this stays fully offline.
    win._nighttime_timer.setInterval(30)
    win._nighttime_timer.start()
    pump(lambda: len(calls) >= 3, timeout=5)
    assert len(calls) >= 3, "the QTimer must fire _nighttime_pull on its own"
    print("  [ok] the QTimer actually fires _nighttime_pull on its own")
    win._nighttime_timer.stop()
    win._nighttime_timer.setInterval(gui.NIGHTTIME_PULL_INTERVAL_MS)

    win.on_run = orig_on_run   # restore before any real run below

    # --- disabling stops the timer and returns manual control --------------
    win.nighttime_enable.setChecked(False); app.processEvents()
    assert not win._nighttime_timer.isActive()
    assert win.mode_fetch.isEnabled() and win.mode_local.isEnabled() \
        and win.fetch_date.isEnabled()
    assert not win.nighttime_pull_now.isEnabled()
    print("  [ok] disabling stops the timer and restores manual data-source control")

    # --- _on_nighttime_pull_done: timestamp + status, using a REAL prep -----
    # obtained the ordinary offline way (mode_local + bundled test data), not
    # nighttime mode's own (network) fetch path -- see module docstring.
    win.mode_local.setChecked(True)
    win.dimm_edit.setText(f"{DATA}/20260525_dimm.dat")
    win.mass_edit.setText(f"{DATA}/20260525_mass.dat")
    win.masspro_edit.setText(f"{DATA}/20260525_masspro.dat")
    win.tel_k1.setChecked(True); win._validate(); win.on_run()
    pump(lambda: win.res is not None)
    assert win._nighttime_last_pull is None, \
        "a run with nighttime mode OFF must not record a pull"

    # flip the checkbox's state WITHOUT firing _on_nighttime_toggled (which
    # would force fetch mode / disable widgets / kick off a real fetch) --
    # this test only wants _on_prepared's hook to see "nighttime mode is on"
    win.nighttime_enable.blockSignals(True)
    win.nighttime_enable.setChecked(True)
    win.nighttime_enable.blockSignals(False)
    win._on_prepared(win.prep, win.prep_log)   # re-invoke the hook directly
    pump(lambda: win.res is not None)
    assert win._nighttime_last_pull is not None
    now_hst = win._nighttime_now_hst()
    assert abs((now_hst - win._nighttime_last_pull).total_seconds()) < 5
    assert "last pull" in win.nighttime_status.text()
    assert "next pull" in win.nighttime_status.text()
    print(f"  [ok] _on_prepared records the pull timestamp when nighttime mode "
          f"is active ({win.nighttime_status.text()})")

    # --- field map: "time of last pull" Conditions option -------------------
    win.fm_cond.setCurrentText(gui.NIGHTTIME_FM_COND)
    when, t_hst = win._fm_when_time()
    assert when == "time" and t_hst == win._nighttime_last_pull, (when, t_hst)
    print("  [ok] field map's 'time of last pull' resolves to the recorded "
          "pull timestamp")

    # graceful fallback when no pull has happened yet -> whole-night median,
    # not a crash or a stale/undefined time
    win._nighttime_last_pull = None
    when, t_hst = win._fm_when_time()
    assert when == "night" and t_hst is None, (when, t_hst)
    print("  [ok] 'time of last pull' falls back to whole-night median before "
          "any pull has happened")

    # the field map actually renders under this option (exercises the full
    # engine.field_snapshot("time", ...) path via a real MASS/DIMM night).
    # The timestamp itself is arbitrary here -- field_snapshot just picks the
    # nearest available sample by |Δt|, so realism vs. the bundled night's
    # actual date doesn't matter for this render-pipeline check.
    win._nighttime_last_pull = win._nighttime_now_hst() - timedelta(hours=2)
    win.plot_tabs.setCurrentIndex(1); app.processEvents()
    pump(lambda: not win._fm_debounce.isActive() and not win._fm_settle.isActive())
    fig = win._fm_holder["canvas"].figure
    assert any(ax.images for ax in fig.axes), \
        "field map must render under the nighttime Conditions option"
    print("  [ok] field map renders using the 'time of last pull' snapshot")

    # nighttime_enable is deliberately NOT in config persistence (see
    # nighttime.py's module docstring): a loaded config must never silently
    # start background network polling.
    cfg = win._collect_config()
    assert "nighttime_enable" not in cfg, \
        "nighttime mode must not be a persisted/auto-resumed config setting"
    print("  [ok] nighttime mode is deliberately excluded from saved configs")

    # --- dark theme (gui/theme.py): View menu toggle + nighttime auto -------
    # reset to a clean baseline first (nighttime was left checked via a
    # blockSignals flip above), then drive the real toggle paths with on_run
    # stubbed so nothing touches the network.
    win.nighttime_enable.blockSignals(True)
    win.nighttime_enable.setChecked(False)
    win.nighttime_enable.blockSignals(False)
    win.dark_action.setChecked(False); app.processEvents()
    orig_on_run2 = win.on_run
    win.on_run = lambda: None

    assert not gui.is_dark(app)
    assert win.nighttime_status.property("cue") == "secondary", \
        "status labels must use semantic cues (theme-adaptive), not " \
        "hard-coded light-theme colors"
    assert 'cue="secondary"' in app.styleSheet()

    win.dark_action.setChecked(True); app.processEvents()
    assert gui.is_dark(app), "View menu dark toggle must darken the palette"
    assert win._dark_auto is False, "a manual toggle is user-owned, not auto"
    win.dark_action.setChecked(False); app.processEvents()
    assert not gui.is_dark(app), "unchecking must restore the light palette"
    print("  [ok] View ▸ Dark theme toggles the palette (widgets-only)")

    win.nighttime_enable.setChecked(True); app.processEvents()
    assert gui.is_dark(app) and win._dark_auto is True, \
        "enabling nighttime mode must auto-switch to dark"
    win.nighttime_enable.setChecked(False); app.processEvents()
    assert not gui.is_dark(app) and win._dark_auto is False, \
        "disabling nighttime mode must hand the auto-dark back"
    print("  [ok] nighttime mode auto-darkens and auto-restores")

    # ownership: a user's own dark choice is neither claimed nor reverted
    win.dark_action.setChecked(True); app.processEvents()
    win.nighttime_enable.setChecked(True); app.processEvents()
    assert win._dark_auto is False
    win.nighttime_enable.setChecked(False); app.processEvents()
    assert gui.is_dark(app), "a pre-chosen dark theme must survive nighttime off"
    # ...and a mid-night override wins over the auto-restore
    win.dark_action.setChecked(False); app.processEvents()
    win.nighttime_enable.setChecked(True); app.processEvents()
    win.dark_action.setChecked(False); app.processEvents()   # user forces light
    assert win._dark_auto is False
    win.nighttime_enable.setChecked(False); app.processEvents()
    assert not gui.is_dark(app)
    print("  [ok] user ownership: pre-chosen and mid-night overrides win")

    # dark theme IS a persisted preference (unlike nighttime mode itself)
    win.dark_action.setChecked(True); app.processEvents()
    cfg = win._collect_config()
    assert cfg["dark_theme"] is True
    win.dark_action.setChecked(False); app.processEvents()
    win._loading = True; win._apply_config(cfg); win._loading = False
    app.processEvents()
    assert win.dark_action.isChecked() and gui.is_dark(app)
    win.dark_action.setChecked(False); app.processEvents()
    win.on_run = orig_on_run2
    print("  [ok] dark theme persists in configs and re-applies on load")

    # --- day/night gate: the engine astronomy -------------------------------
    # (iers auto_max_age is already None via the gui import.) Fixed UTC
    # instants, checked against almanac Hawaii rise/set: July sunset ~19:10 /
    # sunrise ~05:55 HST; HST = UTC-10.
    assert not engine.is_night_at_keck(datetime(2026, 7, 21, 22, 0)), "noon HST"
    assert engine.is_night_at_keck(datetime(2026, 7, 21, 12, 0)), "2am HST"
    assert engine.is_night_at_keck(datetime(2026, 7, 22, 7, 0)), "9pm HST"
    assert not engine.is_night_at_keck(datetime(2026, 7, 21, 18, 0)), "8am HST"
    assert engine.is_night_at_keck(datetime(2026, 1, 15, 5, 0)), \
        "7pm HST in January (winter early sunset)"
    a_noon = engine.sun_altitude_deg(datetime(2026, 7, 21, 22, 0))
    assert a_noon > 60, f"July noon sun near-zenith in Hawaii, got {a_noon}"
    print("  [ok] engine: is_night_at_keck matches almanac day/night at Keck")

    # --- safety 1: refuse to ENABLE in daytime ------------------------------
    win.on_run = lambda: None
    win._nighttime_is_night = lambda: False          # daytime
    win.nighttime_enable.setChecked(True); app.processEvents()
    assert not win.nighttime_enable.isChecked(), \
        "daytime enable must be refused (MKWC only publishes at night)"
    assert not win._nighttime_timer.isActive()
    assert "daytime" in win.nighttime_status.text(), win.nighttime_status.text()
    assert win.mode_fetch.isEnabled() and win.fetch_date.isEnabled(), \
        "a refused enable must leave the source controls usable"
    print("  [ok] safety: daytime enable refused, with the reason in the status")

    # --- safety 2: auto-stop at sunrise -------------------------------------
    win._nighttime_is_night = lambda: True
    win.nighttime_enable.setChecked(True); app.processEvents()
    assert win.nighttime_enable.isChecked() and win._nighttime_timer.isActive()
    win._nighttime_is_night = lambda: False          # the sun comes up
    win._nighttime_pull()                            # what the next tick runs
    app.processEvents()
    assert not win.nighttime_enable.isChecked(), \
        "a tick after sunrise must shut nighttime mode off"
    assert not win._nighttime_timer.isActive()
    assert "sunrise" in win.nighttime_status.text(), win.nighttime_status.text()
    assert win.mode_fetch.isEnabled(), "sunrise stop must restore the controls"
    print("  [ok] safety: session left running past dawn stops itself "
          "(within one 5-min cycle)")

    # --- safety 3: a failed FIRST pull disarms; a later failure does not ----
    win._nighttime_is_night = lambda: True
    win._nighttime_last_pull = None                  # no success yet
    win.nighttime_enable.setChecked(True); app.processEvents()
    assert win.nighttime_enable.isChecked()
    orig_exec = QtWidgets.QMessageBox.exec           # keep the dialog out of
    QtWidgets.QMessageBox.exec = lambda self: 0      # the offline test
    try:
        win._on_failed("fetch failed: HTTP 404 (dimm)", "")
        app.processEvents()
        assert not win.nighttime_enable.isChecked(), \
            "a failed FIRST pull must disarm nighttime mode"
        assert not win._nighttime_timer.isActive()
        assert "try again later" in win.nighttime_status.text(), \
            win.nighttime_status.text()
        # ...but once a pull HAS succeeded, a later failure is transient:
        win.nighttime_enable.setChecked(True); app.processEvents()
        win._nighttime_last_pull = win._nighttime_now_hst()
        win._on_failed("fetch failed: transient blip", "")
        app.processEvents()
        assert win.nighttime_enable.isChecked(), \
            "a post-success failure must NOT disarm (timer retries next cycle)"
        assert win._nighttime_timer.isActive()
    finally:
        QtWidgets.QMessageBox.exec = orig_exec
    win.nighttime_enable.setChecked(False); app.processEvents()
    print("  [ok] safety: first-pull failure disarms with 'try again later'; "
          "post-success failures keep retrying")


if __name__ == "__main__":
    main()
