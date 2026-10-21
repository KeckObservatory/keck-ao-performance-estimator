#!/usr/bin/env python3
"""Measured SR summit hotfixes (Eduardo 2026-10-20, kept in sync with PyAO's
strehl_tool): Robust sky on by default (and the config fallback); a
re-measured frame is appended to the image log without the duplicate
question; every logged row is appended at once to tonight's session journal
($KECK_AO_SR_LOG_DIR/sr_tool_<UT date>.csv -- Export CSV's columns and
rounding, the header once; rows deleted in the table stay in the file); a
journal that cannot be written is reported once; the text log is appended
to sr_tool_<UT date>.log when the window closes. Offline;
QT_QPA_PLATFORM=offscreen.
"""
import csv
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
import warnings
warnings.filterwarnings("ignore")

LOG_DIR = tempfile.mkdtemp(prefix="gui53_logs_")
os.environ["KECK_AO_SR_LOG_DIR"] = LOG_DIR

from qtcompat import QtWidgets

import keck_ao_estimator.gui as gui
from keck_ao_estimator import session_log
from keck_ao_estimator.gui.tabs.nirc2_strehl import (DUPLICATE_POLICY,
                                                     NIRC2_CSV_COLUMNS)
from gui_phase29 import make_frame, pump


def _no_prompt(self):
    raise AssertionError("the duplicate question must not open")


def main():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow()
    win.show()
    app.processEvents()
    csv_path, log_path = win._nirc2_session_paths()
    assert os.path.dirname(csv_path) == LOG_DIR, csv_path
    tag = session_log.night_tag()
    assert os.path.basename(csv_path) == f"sr_tool_{tag}.csv"
    assert os.path.basename(log_path) == f"sr_tool_{tag}.log"
    assert f"session journal: {csv_path}" in win.n2_log.toPlainText()
    assert not os.path.exists(csv_path)
    print(f"  [ok] session paths announced in the log: {csv_path}")

    # ---- Robust sky on by default, also when a config lacks the key -------
    assert win.n2_robust_sky.isChecked(), "Robust sky must start on"
    c = win._collect_config()
    assert c["nirc2"]["robust_sky"] is True
    c["nirc2"].pop("robust_sky")
    win.n2_robust_sky.setChecked(False)
    win._apply_config(c)
    assert win.n2_robust_sky.isChecked(), "config fallback is on"
    c["nirc2"]["robust_sky"] = False
    win._apply_config(c)
    assert not win.n2_robust_sky.isChecked(), "a saved False is honoured"
    win.n2_robust_sky.setChecked(True)
    print("  [ok] Robust sky default on; config fallback on; saved off kept")

    # ---- duplicates are appended, never asked ----------------------------
    assert DUPLICATE_POLICY == "append"
    assert win._n2_dup_batch_policy == "append"
    QtWidgets.QMessageBox.exec = _no_prompt
    tmp = tempfile.mkdtemp(prefix="gui53_frames_")
    make_frame(tmp, 1)
    win.n2_path.setText(tmp)
    win.n2_im1.setValue(1)
    win.n2_nim.setValue(1)
    win.n2_nbg.setValue(0)
    win.n2_autofind.setChecked(True)
    win._on_nirc2_go()
    pump(lambda: win.n2_go.isEnabled())
    assert len(win._n2_csv_rows) == 1, win.n2_log.toPlainText()
    assert win._n2_dup_batch_policy == "append", "a run resets to append"
    with open(csv_path, newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == [label for _k, label in NIRC2_CSV_COLUMNS], rows[0]
    assert len(rows) == 2, rows
    win._on_nirc2_go()                       # the same frame again
    pump(lambda: win.n2_go.isEnabled())
    assert len(win._n2_csv_rows) == 2, "re-measurement appended, no prompt"
    with open(csv_path, newline="") as fh:
        rows = list(csv.reader(fh))
    assert len(rows) == 3, rows
    # the journal row carries the table's rounding (Export CSV's numbers)
    row = win._n2_csv_rows[-1]
    for (key, _label), cell in zip(NIRC2_CSV_COLUMNS, rows[-1]):
        v = row.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            assert float(cell) == win._csv_round(v), (key, cell, v)
        elif v is None:
            assert cell == "", (key, cell)
    # rows deleted in the table stay in the journal
    del win._n2_csv_rows[0]
    assert len(open(csv_path).read().splitlines()) == 3
    print("  [ok] GO! x2: 2 rows in the table, header + 2 rows in the "
          "journal, no duplicate prompt; table delete leaves the file")

    # ---- a journal that cannot be written is reported once --------------
    win._n2_session_log_dir = os.path.join(tmp, "n0001.fits", "sub")
    win._n2_journal_failed = False
    win._nirc2_append_row(dict(row))
    win._nirc2_append_row(dict(row))
    txt = win.n2_log.toPlainText()
    assert txt.count("session journal: cannot write") == 1, txt
    assert len(win._n2_csv_rows) == 3, "the table keeps working"
    assert win._nirc2_save_session_log() is None
    win._n2_session_log_dir = LOG_DIR
    win._n2_journal_failed = False
    print("  [ok] unwritable journal: reported once, table unaffected")

    # ---- closing saves the text log --------------------------------------
    win.n2_log.appendPlainText("gui_phase53 marker line")
    assert not os.path.exists(log_path)
    win.close()
    app.processEvents()
    saved = open(log_path).read()
    assert saved.startswith("=== measured-SR session closed "), saved[:80]
    assert "gui_phase53 marker line" in saved
    assert f"session journal: {csv_path}" in saved
    print(f"  [ok] close appended the text log to {log_path}")
    print("gui_phase53: PASS")


if __name__ == "__main__":
    main()
