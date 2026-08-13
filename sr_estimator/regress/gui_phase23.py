#!/usr/bin/env python3
"""Keck starlist picker (Target tab "Load starlist…" -> StarlistPickerMixin):
loading the bundled SYNTHETIC starlist (examples/synthetic_k1lgs.lst -- fake
targets, real-list formatting quirks; see examples/make_synthetic_starlist.py)
populates a pop-up table (same form as the guide-star ranking dialog),
double-clicking a row makes it the science target -- through the SAME
_add_target/_on_target_selected path the Targets dropdown uses, with the PM
fields zeroed (a starlist carries no proper motion) -- and every close path
drops the Qt references. Expected values are DERIVED from parse_starlist()
on the same file (a GUI-vs-parser cross-check), not hardcoded row numbers,
so regenerating the synthetic list cannot silently rot this test.
Run headless."""
import os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
from qtcompat import QtWidgets
import keck_ao_estimator as engine
from keck_ao_estimator.starlist import parse_starlist
import keck_ao_estimator.gui as gui

SAMPLE = os.path.join(ROOT, "examples", "synthetic_k1lgs.lst")


def main():
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.resize(1500, 950); win.show(); app.processEvents()

    # ground truth straight from the parser: the table must agree with it
    entries, skipped = parse_starlist(SAMPLE)
    n = len(entries)
    assert n > 20 and len(skipped) == 1, \
        f"synthetic list changed shape? {n} entries, {len(skipped)} skipped"

    # --- load the bundled synthetic starlist ---------------------------------
    win._open_starlist(SAMPLE)
    app.processEvents()
    assert win._starlist_dialog is not None and win._starlist_table is not None
    assert str(n) in win.status.text() and "synthetic_k1lgs.lst" in win.status.text()
    table = win._starlist_table

    def col(name):
        """Column index BY HEADER TEXT.  Hardcoded indices broke this
        test when an "HA" column was inserted at 4 (2026-07-28): every
        column after it shifted by one and the failure surfaced as an
        opaque assert on item(0, 8).  Look them up instead."""
        for i in range(table.columnCount()):
            h = table.horizontalHeaderItem(i)
            if h is not None and h.text() == name:
                return i
        raise AssertionError(
            f"no {name!r} column; headers are "
            f"{[table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]}")

    assert table.rowCount() == n
    assert table.item(0, 1).text() == entries[0]["name"]
    assert table.item(0, col("role")).text() == "target (lgs)"

    # a target=-linked row that also carries a bare K= key (the synthetic
    # list always has at least one); before any sort, table row == entry idx
    tt_i = next(i for i, e in enumerate(entries)
                if e["target"] and "K" in e["keys"])
    tt = entries[tt_i]
    assert table.item(tt_i, 1).text() == tt["name"]
    assert table.item(tt_i, col("role")).text() == f"TT star → {tt['target']}", \
        "a target=-linked row must be labelled as that target's TT star"
    assert table.item(tt_i, col("K")).text() == f"{float(tt['keys']['K']):.1f}", \
        "bare K= key must fill K col"
    print(f"  [ok] synthetic starlist loads: {n} rows, TT-star target= links "
          f"labelled ({table.item(tt_i, col('role')).text()})")

    # --- double-click picks a target (via the REAL signal, not the slot) -----
    # stale PM from a previously-loaded target MUST be zeroed by the pick: a
    # starlist has no PM, and a leaked one silently shifts the pointing.
    # "Syn Cluster A" is a J2000 science row whose NAME CONTAINS SPACES.
    pick_i = next(i for i, e in enumerate(entries)
                  if e["name"] == "Syn Cluster A")
    pick = entries[pick_i]
    win.pmra_spin.setValue(50.0); win.pmdec_spin.setValue(-20.0)
    table.cellDoubleClicked.emit(pick_i, 0)
    app.processEvents()
    assert win.tname_edit.text() == "Syn Cluster A"
    c = engine.parse_radec(win.ra_edit.text(), win.dec_edit.text())
    assert abs(c.ra.deg - pick["ra_deg"]) * 3600 < 0.01
    assert abs(c.dec.deg - pick["dec_deg"]) * 3600 < 0.01
    assert win.pmra_spin.value() == 0.0 and win.pmdec_spin.value() == 0.0, \
        "picking a starlist row must zero stale proper motion"
    assert win.target_select.currentText() == "Syn Cluster A", \
        "the pick must land in tonight's Targets list and be selected"
    assert "Syn Cluster A" in win.status.text()
    assert "equinox" not in win.status.text(), "J2000 rows must not warn"
    print(f"  [ok] double-click picks Syn Cluster A: fields + Targets dropdown "
          f"set, stale PM zeroed ({win.status.text()})")

    # picking a TT-star row directly also works (they're real stars)
    table.cellDoubleClicked.emit(tt_i, 0)
    app.processEvents()
    assert win.tname_edit.text() == tt["name"]
    assert win.target_select.currentText() == tt["name"]
    print("  [ok] a TT-star row can be picked directly too")

    # --- column sorting (2026-07-21): headers sort by VALUE, and picking
    #     still selects the DISPLAYED row after the re-ordering --------------
    from qtcompat import QtCore
    v_col = col("V")
    table.sortItems(v_col, QtCore.Qt.SortOrder.AscendingOrder)
    v_texts = [table.item(r, v_col).text() for r in range(table.rowCount())]
    nums = [float(s) for s in v_texts if s != "—"]
    assert nums == sorted(nums), "V column must sort numerically"
    n_dash = sum(1 for e in entries if "vmag" not in e["keys"])
    assert n_dash > 0 and all(t == "—" for t in v_texts[-n_dash:]), \
        f"the {n_dash} vmag-less rows must sort last ascending"
    brightest = min((e for e in entries if "vmag" in e["keys"]),
                    key=lambda e: float(e["keys"]["vmag"]))
    assert table.item(0, 1).text() == brightest["name"], \
        f"brightest V first, got {table.item(0, 1).text()}"
    table.cellDoubleClicked.emit(0, 0)
    app.processEvents()
    assert win.tname_edit.text() == brightest["name"], \
        "pick after sorting must select the displayed row, not entry[0]"
    assert abs(engine.parse_radec(win.ra_edit.text(), win.dec_edit.text())
               .ra.deg - brightest["ra_deg"]) * 3600 < 0.01
    print(f"  [ok] header sorting: numeric order, '—' rows last, pick after "
          f"sorting selects the displayed star ({brightest['name']})")

    # --- close path drops references; a reload reopens cleanly ---------------
    dlg = win._starlist_dialog
    dlg.close(); app.processEvents()
    assert win._starlist_dialog is None and win._starlist_table is None, \
        "closing the dialog must null the stored references"
    win._open_starlist(SAMPLE); app.processEvents()
    assert win._starlist_table is not None and \
        win._starlist_table.rowCount() == n
    win._starlist_dialog.close(); app.processEvents()
    print("  [ok] close nulls the references; reloading reopens cleanly")

    # --- non-J2000 equinox: flagged in the table AND warned on pick ----------
    tmp = os.path.join(tempfile.gettempdir(), "p23_b1950.lst")
    with open(tmp, "w") as fh:
        fh.write("OldEpochStar    01 02 03.4 +05 06 07.8 1950.0 vmag=9\n"
                 "junk that does not parse at all\n")
    win._open_starlist(tmp); app.processEvents()
    assert "1 entries" in win.status.text() and "1 malformed" in \
        win.status.text(), win.status.text()
    assert "equinox 1950.0" in win._starlist_table.item(
        0, col("role")).text()
    win._starlist_table.cellDoubleClicked.emit(0, 0); app.processEvents()
    assert "equinox 1950.0" in win.status.text(), win.status.text()
    win._starlist_dialog.close(); app.processEvents()
    print("  [ok] non-J2000 equinox: flagged in the role column, warned in "
          "the status bar on pick; malformed line counted, not fatal")

    # --- a file with nothing parseable reports, and opens no dialog ----------
    tmp2 = os.path.join(tempfile.gettempdir(), "p23_junk.lst")
    with open(tmp2, "w") as fh:
        fh.write("# only a comment\nutter garbage\n")
    win._open_starlist(tmp2); app.processEvents()
    assert win._starlist_dialog is None
    assert "No parseable starlist entries" in win.status.text()
    print("  [ok] an all-junk file reports and opens no dialog")

    win.close()
    print("  [ok] Keck starlist picker end-to-end")


if __name__ == "__main__":
    main()
