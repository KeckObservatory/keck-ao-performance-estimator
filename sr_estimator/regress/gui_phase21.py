#!/usr/bin/env python3
"""TT-star follow-ups to the guide-star catalogue overlay (gui_phase20):

  * a catalogue pick sets the TT/NGS offset's ABSOLUTE RA/Dec (star RA/Dec
    mode), not a delta offset from the target -- a planning user needs real
    coordinates, and they must survive the target itself later moving.
  * switching the TT sensor's band (STRAP R <-> TRICK H/K) re-derives the
    magnitude from the tracked catalogue star instead of leaving the old
    band's number in place -- a real star's R/H/K magnitudes are never
    numerically equal -- and WARNS (rather than silently doing nothing) when
    there's no tracked star, or it has no derivable magnitude in the new
    band. A manual tt_mag edit stops the tracking.

Run headless."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
from qtcompat import QtWidgets, QtCore
import keck_ao_estimator as engine
import keck_ao_estimator.gui as gui
import astropy.units as u
from astropy.coordinates import SkyCoord
DATA = os.path.join(HERE, "data")


def pump(cond, timeout=90):
    app = QtWidgets.QApplication.instance(); t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents(); QtCore.QThread.msleep(10)


def main():
    app = QtWidgets.QApplication(sys.argv[:1])
    win = gui.MainWindow(); win.resize(1500, 950); win.show(); app.processEvents()
    win.mode_local.setChecked(True)
    win.dimm_edit.setText(f"{DATA}/20260525_dimm.dat")
    win.mass_edit.setText(f"{DATA}/20260525_mass.dat")
    win.masspro_edit.setText(f"{DATA}/20260525_masspro.dat")
    win.tel_k1.setChecked(True)
    win.ra_edit.setText("10h00m00s"); win.dec_edit.setText("+20d00m00s")
    win._validate(); win.on_run()
    pump(lambda: win.res is not None)

    fc = win._field_center_deg(); assert fc is not None
    cen = SkyCoord(fc[0] * u.deg, fc[1] * u.deg)
    p = cen.spherical_offsets_by(6 * u.arcsec, 4 * u.arcsec)
    # a real star: distinct R, H, K so a stale cross-band value is obviously
    # wrong if it ever leaks through
    star = {"id": "S1", "ra": float(p.ra.deg), "dec": float(p.dec.deg),
            "mags": {"R": 13.5, "H": 10.2, "K": 9.8}}
    # estimate_sensing_mag only returns None with ZERO usable magnitudes --
    # any real photometry (even in a different band) gets a flagged rough
    # nearest-band fallback instead (see estimate_sensing_mag / gui_phase20),
    # which _sync_tt_mag_for_band treats as a successful (if rough) update,
    # same as a direct catalogue selection already does. So the genuine
    # "not derivable" case is a star with no usable magnitude at all.
    no_mag = {"id": "S2", "ra": float(p.ra.deg), "dec": float(p.dec.deg),
              "mags": {"G": None}}

    # --- catalogue pick stores ABSOLUTE coordinates, not a delta offset ----
    win.tt_sensor.setCurrentText("STRAP (R)")
    win._fm_select_star(star, "tt")
    assert win.tt_offset.mode.currentIndex() == 2, "must be star RA/Dec mode"
    assert win.tt_offset.sra.text() and win.tt_offset.sdec.text()
    assert abs(win.tt_mag.value() - 13.5) < 1e-6, win.tt_mag.value()
    # moving the target must NOT change the stored star position (it's
    # absolute) -- only the resolved offset from the (now different) target
    win.ra_edit.setText("10h01m00s")
    resolved = engine.parse_radec(win.tt_offset.sra.text(), win.tt_offset.sdec.text())
    # tolerance set by the displayed text's own precision (hms/dms to a few
    # decimals), not exactness -- just confirms it's the star's position, not
    # something that silently tracked the target
    assert resolved.separation(p).arcsec < 0.05, "stored star position must be absolute"
    win.ra_edit.setText("10h00m00s")           # restore
    print("  [ok] catalogue pick: absolute RA/Dec stored, survives a target move")

    # --- sensor switch re-derives the magnitude from the tracked star ------
    win.tt_sensor.setCurrentText("TRICK (K)")
    assert abs(win.tt_mag.value() - 9.8) < 1e-6, \
        f"must re-derive the K mag from the tracked star, not keep R's 13.5: {win.tt_mag.value()}"
    assert "updated" in win.fm_catalog_status.text().lower()
    win.tt_sensor.setCurrentText("TRICK (H)")
    assert abs(win.tt_mag.value() - 10.2) < 1e-6, win.tt_mag.value()
    win.tt_sensor.setCurrentText("STRAP (R)")
    assert abs(win.tt_mag.value() - 13.5) < 1e-6, win.tt_mag.value()
    print("  [ok] TT magnitude re-derived from the tracked catalogue star on every "
          "STRAP<->TRICK band switch (R/H/K all distinct, none stale)")

    # the retired "STRAP legacy (R)" combo entry is GONE (2026-08-07); an old
    # saved config naming it must fall back to refined STRAP, not silently
    # leave whatever sensor was previously selected
    assert win.tt_sensor.findText("STRAP legacy (R)") < 0
    cfg = win._collect_config()
    cfg["tt_sensor"] = "STRAP legacy (R)"
    win.tt_sensor.setCurrentText("TRICK (K)")
    win._apply_config(cfg)
    assert win.tt_sensor.currentText() == "STRAP (R)", win.tt_sensor.currentText()
    # _apply_config swaps the combo with signals blocked, so _tt_wfs_band is
    # stale ("K") here -- resync it before the sections below rely on band-
    # change detection (pre-existing config-load behavior, not this feature's)
    win._on_tt_sensor_changed()
    assert abs(win.tt_mag.value() - 13.5) < 1e-6, win.tt_mag.value()
    print("  [ok] retired STRAP-legacy combo entry: an old config naming it "
          "falls back to STRAP (R)")

    # --- no derivable magnitude in the new band -> warn, don't touch tt_mag ---
    before = win.tt_mag.value()
    win._fm_select_star(no_mag, "tt")          # no usable mags at all -> no update
    assert abs(win.tt_mag.value() - before) < 1e-9
    assert "derivable" in win.fm_catalog_status.text().lower()
    assert win._tt_star_ref is no_mag, "still tracked, in case a later band derives it"
    win.tt_sensor.setCurrentText("TRICK (K)")
    assert abs(win.tt_mag.value() - before) < 1e-9, \
        "no derivable K magnitude -> must NOT silently change tt_mag"
    txt = win.fm_catalog_status.text().lower()
    assert "no derivable" in txt or "not updated" in txt, win.fm_catalog_status.text()
    win.tt_sensor.setCurrentText("STRAP (R)")
    print("  [ok] switching to a band the catalogue star has no magnitude for "
          "warns and leaves tt_mag untouched")

    # --- manual edit clears the tracked star -> next switch warns ----------
    win._fm_select_star(star, "tt")
    assert win._tt_star_ref is star
    win.tt_mag.setValue(14.7)                 # simulates a real user edit
    assert win._tt_star_ref is None, "a manual tt_mag edit must drop the tracked star"
    before = win.tt_mag.value()
    win.tt_sensor.setCurrentText("TRICK (K)")
    assert abs(win.tt_mag.value() - before) < 1e-9, \
        "no tracked star -> must NOT silently change tt_mag"
    txt = win.fm_catalog_status.text().lower()
    assert "no catalogue star" in txt or "not updated" in txt, win.fm_catalog_status.text()
    win.tt_sensor.setCurrentText("STRAP (R)")
    print("  [ok] a manual tt_mag edit stops tracking; the next band switch warns "
          "instead of silently leaving a stale value")

    print("  [ok] TT-star coordinate storage + sensor-switch magnitude handling")


if __name__ == "__main__":
    main()
