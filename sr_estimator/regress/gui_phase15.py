#!/usr/bin/env python3
"""Three-mode offset entry (NGS offset + TT-star offset): total arcsec,
sky-projected ΔRA/ΔDec, and offset-star RA/Dec -> separation. All resolve to
the scalar arcsec the engine consumes. Run headless."""
import os, sys, time, json, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
from qtcompat import QtWidgets, QtCore
import keck_ao_estimator as engine
import keck_ao_estimator.gui as gui
np = engine.np
DATA = os.path.join(HERE, "data")


def pump(cond, timeout=90):
    app = QtWidgets.QApplication.instance(); t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents(); QtCore.QThread.msleep(10)


def main():
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.resize(1500, 950); win.show(); app.processEvents()
    win.ra_edit.setText("15h49m57.7s"); win.dec_edit.setText("-03d55m16s")

    for name, entry in (("NGS", win.ngs_offset), ("TT", win.tt_offset)):
        # mode 0: total
        entry.setValue(7.0)
        assert abs(entry.value() - 7.0) < 1e-9 and entry.ok(), name
        # mode 1: sky-projected components -> hypot
        entry.mode.setCurrentIndex(1)
        entry.dra.setValue(3.0); entry.ddec.setValue(4.0)
        app.processEvents()
        assert abs(entry.value() - 5.0) < 1e-9, f"{name} hypot"
        # mode 2: offset-star coordinates -> separation (checked vs astropy)
        entry.mode.setCurrentIndex(2)
        entry.sra.setText("15h49m58.5s"); entry.sdec.setText("-03d55m16s")
        app.processEvents()
        from astropy.coordinates import SkyCoord
        sep = SkyCoord("15h49m57.7s", "-03d55m16s").separation(
            SkyCoord("15h49m58.5s", "-03d55m16s")).arcsec
        assert entry.ok() and abs(entry.value() - sep) < 1e-6, f"{name} sep"
        print(f"  [ok] {name} offset: total=7.0, (3,4)->5.0, star->{entry.value():.2f}\" (=astropy)")

    # star-coord mode with bad coordinates blocks Run
    win.mode_local.setChecked(True)
    win.dimm_edit.setText(os.path.join(DATA, "20260525_dimm.dat"))
    win.mass_edit.setText(os.path.join(DATA, "20260525_mass.dat"))
    win.masspro_edit.setText(os.path.join(DATA, "20260525_masspro.dat"))
    win._validate()
    assert win.run_btn.isEnabled(), "should run with valid coords"
    win.ngs_offset.sra.setText("not-a-coordinate"); win._validate()
    assert not win.ngs_offset.ok() and not win.run_btn.isEnabled(), \
        "bad star coords must block Run"
    print("  [ok] invalid star coordinates disable Run")
    win.ngs_offset.setValue(0.0)   # back to a clean total; TT still star-mode

    # the resolved offset actually reaches the engine via collect_args
    win.ngs_offset.mode.setCurrentIndex(1)
    win.ngs_offset.dra.setValue(6.0); win.ngs_offset.ddec.setValue(8.0)
    win.tt_offset.setValue(12.3)
    app.processEvents()
    a = win.collect_args("/tmp/x.png")
    assert abs(a.ngs_offset - 10.0) < 1e-9 and abs(a.tt_offset - 12.3) < 1e-9
    print(f"  [ok] collect_args: ngs_offset={a.ngs_offset:.1f} tt_offset={a.tt_offset:.1f}")

    # config round-trip preserves mode + fields
    cfg = os.path.join(tempfile.gettempdir(), "p15.json")
    win.ngs_offset.mode.setCurrentIndex(2)
    win.ngs_offset.sra.setText("15h49m58.5s"); win.ngs_offset.sdec.setText("-03d55m16s")
    app.processEvents()
    with open(cfg, "w") as fh:
        json.dump(win._collect_config(), fh)
    win.ngs_offset.setValue(0.0)                   # disturb
    with open(cfg) as fh:
        win._apply_config(json.load(fh))
    assert win.ngs_offset.mode.currentIndex() == 2, "mode not restored"
    assert win.ngs_offset.sra.text() == "15h49m58.5s", "coords not restored"
    assert abs(win.ngs_offset.value() - sep) < 1e-6, "resolved value not restored"
    print("  [ok] config round-trip preserves offset mode + coordinates")

    # a run works end-to-end with an offset supplied by coordinates
    win.tel_k1.setChecked(True); win._validate(); win.on_run()
    pump(lambda: win.res is not None, timeout=90)
    assert win.res is not None and win.args_cached.ngs_offset > 0
    print(f"  [ok] full run with coordinate-derived NGS offset "
          f"({win.args_cached.ngs_offset:.2f}\")")


if __name__ == "__main__":
    main()
