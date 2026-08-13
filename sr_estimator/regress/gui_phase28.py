#!/usr/bin/env python3
"""Free-typing TimeEdit + UTC display/entry mode (2026-07-23).

TimeEdit replaces QTimeEdit for the field-map and summary 'specific time'
fields: select-all/clear/retype just works ('2135', '935', '9', '21:35'
all parse; garbage reverts) -- QTimeEdit's section editing made that
impossible (Eduardo). UTC mode (Data tab 'Times:' checkbox) switches every
DISPLAYED and ENTERED wall time to UTC (+10 h) while everything internal
-- engine, config, saved windows -- stays HST: plot axes/annotations (via
plots.apply_utc_display, GUI-only post-processing; the CLI's frozen
outputs are untouched and the harness stays green), the observing-windows
list, the specific-time fields' interpretation, and the status readouts.
Run headless."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
from qtcompat import QtWidgets, QtCore
import keck_ao_estimator as engine
import keck_ao_estimator.gui as gui
DATA = os.path.join(HERE, "data")


def pump(cond, timeout=90):
    app = QtWidgets.QApplication.instance(); t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents(); QtCore.QThread.msleep(10)


def shift_text_engine_contract():
    """shift_hst_text: pure display helper, checked headlessly first."""
    assert engine.shift_hst_text("21:35 HST") == "07:35 UTC"
    assert engine.shift_hst_text("(00:14–03:20 HST)") == \
        "(10:14–13:20 UTC)"
    assert engine.shift_hst_text("no clock times HST") == "no clock times UTC"
    assert engine.shift_hst_text("plain text") == "plain text"
    print("  [ok] shift_hst_text: +10 h with midnight roll, HST->UTC tag")


def main():
    shift_text_engine_contract()
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.resize(1500, 950); win.show(); app.processEvents()

    # --- TimeEdit: free typing, every reasonable format, garbage reverts ----
    te = win.stats_time
    te.setEnabled(True)
    fired = []
    te.timeChanged.connect(lambda t: fired.append(t.toString("HH:mm")))
    for typed, want in (("2135", "21:35"), ("935", "09:35"), ("9", "09:00"),
                        ("21:35", "21:35"), ("7:5", "07:05"),
                        ("07.05", "07:05")):
        te.setText(typed); te._commit()
        assert te.text() == want and te.time().toString("HH:mm") == want, \
            (typed, te.text())
    for garbage in ("nonsense", "", "25:00", "12:75", "12345"):
        te.setText(garbage); te._commit()
        assert te.text() == "07:05", f"{garbage!r} must revert, not stick"
    assert fired and fired[-1] == "07:05"
    n = len(fired)
    te.setText("07:05"); te._commit()
    assert len(fired) == n, "recommitting the same time must not re-fire"
    win.fm_time.setTime(QtCore.QTime(23, 30))
    assert win.fm_time.time() == QtCore.QTime(23, 30), \
        "fm_time is a TimeEdit too and keeps the QTimeEdit setTime/time API"
    print("  [ok] TimeEdit: 2135/935/9/21:35/7:5/07.05 parse; garbage/"
          "invalid reverts; no duplicate timeChanged; QTime API intact")

    # --- load the bundled night ----------------------------------------------
    win.mode_local.setChecked(True)
    win.dimm_edit.setText(f"{DATA}/20260525_dimm.dat")
    win.mass_edit.setText(f"{DATA}/20260525_mass.dat")
    win.masspro_edit.setText(f"{DATA}/20260525_masspro.dat")
    win.target_enable.setChecked(True)
    win._validate(); win.on_run()
    pump(lambda: win.res is not None); app.processEvents()

    wins_hst = [win.windows_list.item(i).text()
                for i in range(win.windows_list.count())]
    assert wins_hst == ["00:14-03:20"], wins_hst
    fig = win._main_holder["canvas"].figure
    assert any((ax.get_xlabel() or "").startswith("HST (")
               for ax in fig.axes)

    # HST-mode reference: 'specific time' 03:49 HST resolves to a t_hst
    win.stats_cond.setCurrentText("specific time"); app.processEvents()
    win.stats_time.setTime(QtCore.QTime(3, 49))
    when, t_ref_hstmode = win._when_time_from(win.stats_cond, win.stats_time)
    assert when == "time" and t_ref_hstmode.hour == 3

    # --- toggle UTC -----------------------------------------------------------
    prev = win.res
    win.utc_cb.setChecked(True)
    pump(lambda: win.res is not prev, timeout=30); app.processEvents()

    # windows list DISPLAYS UT, label says so, engine still gets HST
    wins_ut = [win.windows_list.item(i).text()
               for i in range(win.windows_list.count())]
    assert wins_ut == ["10:14-13:20"], wins_ut
    assert win.windows_row_label.text() == "Windows (UT):"
    a = win.collect_args("x.png")
    assert a.window == ["00:14-03:20"], \
        "the ENGINE must receive HST windows regardless of display mode"
    print("  [ok] UTC mode: windows display 10:14-13:20 UT, engine gets "
          "00:14-03:20 HST")

    # plots: xlabel + window annotation + tick labels all UTC (+10 h)
    fig = win._main_holder["canvas"].figure
    xl = [ax.get_xlabel() for ax in fig.axes
          if (ax.get_xlabel() or "").startswith("UTC (")]
    assert xl and xl[0].endswith("HST)"), xl
    ann = [t.get_text() for ax in fig.axes for t in ax.texts
           if "observations" in t.get_text()]
    assert ann and "UTC" in ann[0] and "10:14" in ann[0]
    fig.canvas.draw()
    ax = next(a2 for a2 in fig.axes if "UTC" in (a2.get_xlabel() or ""))
    first_tick = [t.get_text() for t in ax.get_xticklabels()][0]
    assert first_tick == "06:00", \
        f"20:00 HST must tick as 06:00 UTC, got {first_tick}"
    print(f"  [ok] plots relabelled: xlabel {xl[0]!r}, annotation "
          f"'…10:14–13:20 UTC', ticks shifted (+10 h)")

    # advisory speaks UTC
    assert "UTC" in win.fa_advisory.text() and \
        "HST" not in win.fa_advisory.text()

    # 'specific time' typed as UT means the SAME instant: 13:49 UT == the
    # 03:49 HST reference resolved above
    win.stats_time.setTime(QtCore.QTime(13, 49))
    when, t_ref_utcmode = win._when_time_from(win.stats_cond, win.stats_time)
    assert when == "time" and t_ref_utcmode == t_ref_hstmode, \
        (t_ref_utcmode, t_ref_hstmode)
    print("  [ok] 'specific time' 13:49 UT resolves to the identical "
          "internal instant as 03:49 HST")

    # config: canonical HST windows + the mode flag itself
    cfg = win._collect_config()
    assert cfg["windows"] == ["00:14-03:20"], "configs store HST canonically"
    assert cfg["utc_times"] is True

    # --- toggle back: everything returns to HST exactly ----------------------
    prev = win.res
    win.utc_cb.setChecked(False)
    pump(lambda: win.res is not prev, timeout=30); app.processEvents()
    wins_back = [win.windows_list.item(i).text()
                 for i in range(win.windows_list.count())]
    assert wins_back == ["00:14-03:20"], wins_back
    assert win.windows_row_label.text() == "Windows (HST):"
    fig = win._main_holder["canvas"].figure
    assert any((ax.get_xlabel() or "").startswith("HST (")
               for ax in fig.axes)
    assert "HST" in win.fa_advisory.text()
    print("  [ok] toggling back restores HST display exactly (windows, "
          "xlabel, advisory)")

    # --- config round-trip re-applies the mode --------------------------------
    win.utc_cb.setChecked(True)
    pump(lambda: not win._busy, timeout=30); app.processEvents()
    cfg = win._collect_config()
    win2 = gui.MainWindow()
    win2._apply_config(cfg)
    app.processEvents()
    assert win2.utc_cb.isChecked()
    wins2 = [win2.windows_list.item(i).text()
             for i in range(win2.windows_list.count())]
    assert wins2 == ["10:14-13:20"], \
        f"a UTC-mode config must display UT windows on load: {wins2}"
    assert win2.windows_row_label.text() == "Windows (UT):"
    win2.close()
    print("  [ok] config round-trip: utc_times persists, HST-canonical "
          "windows re-display as UT on load")

    win.grab().save(os.path.join(HERE, "gui_phase28.png"))
    print("  [ok] screenshot saved")


if __name__ == "__main__":
    main()
