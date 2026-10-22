#!/usr/bin/env python3
"""Instrument parameter files (Eduardo 2026-10-21, kept in sync with PyAO's
strehl_tool): the Measured SR menu lists data/instruments/*.yaml plus files
in $KECK_AO_INSTRUMENTS; NIRC2 and OSIRIS keep their code paths (same
parameters, reduction and DL PSF); an invented 8 m telescope described
only by a file is listed, its frames are listed with raw copies skipped,
its PREFIX fills in, GO! measures its Strehl from its circular pupil, and
Latest finds its night directory; a broken file is reported in the log.
Offline; QT_QPA_PLATFORM=offscreen.
"""
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

import numpy as np
from astropy.io import fits

TMP = tempfile.mkdtemp(prefix="gui54_")
INST_DIR = os.path.join(TMP, "instruments")
DATA = os.path.join(TMP, "data")
os.makedirs(INST_DIR)
with open(os.path.join(INST_DIR, "myinst.yaml"), "w") as fh:
    fh.write(f"""key: myinst
name: MYINST
telescope: Test 8 m
order: 50
identify:
  - {{keyword: INSTRUME, contains: MYINST}}
frames:
  pattern: '^img(\\d{{5}})\\.fits$'
  digits: 5
  prefix: img
  skip: '_raw\\.'
night:
  model: directories
  roots: ['{DATA}/*']
  name_format: '%Y%m%d'
  subdir: raw
params:
  model: generic
  plate_scale_mas: 12.0
  wavelength:
    filter_keyword: FILTER
    filters: {{K: 2.2}}
  pupil_rotation_deg: 10.0
reduce:
  model: generic
pupil:
  model: circular
  diameter_m: 8.0
  obscuration_m: 1.2
  spiders: {{count: 4, width_m: 0.05, angle_deg: 45.0}}
""")
with open(os.path.join(INST_DIR, "broken.yaml"), "w") as fh:
    fh.write("key: [unclosed\n")
os.environ["KECK_AO_INSTRUMENTS"] = INST_DIR
os.environ.setdefault("KECK_AO_SR_LOG_DIR", os.path.join(TMP, "logs"))

from qtcompat import QtWidgets

import keck_ao_estimator.gui as gui
from keck_ao_estimator import image_strehl, instruments
from keck_ao_estimator.nirc2 import nirc2_frame_params
from keck_ao_estimator.nirc2_psf import nirc2_dl_psf
from keck_ao_estimator.pupil_psf import circular_dl_psf
from gui_phase29 import pump


def check_parity():
    h = fits.Header({"INSTRUME": "NIRC2", "CAMNAME": "narrow",
                     "PMSNAME": "largehex", "EFFWAVE": 2.12, "ROTPPOSN": 10.0,
                     "EL": 55.0, "COADDS": 3})
    n2 = instruments.get("nirc2")
    assert instruments.identify(h) is n2
    p = n2.frame_params(h)
    assert p == nirc2_frame_params(h)
    assert np.array_equal(n2.dl_psf(p, npix=128), nirc2_dl_psf(
        p.camname, p.pmsname, p.effwave_um, p.pmrangl_deg, npix=128,
        daytime=p.daytime, sfp=p.sfp))
    raw = np.random.default_rng(1).normal(100, 5, (2048, 2048))
    assert np.array_equal(instruments.get("osiris").reduce_frame(raw),
                          image_strehl.osiris_reduce(raw, crop=False))
    print("  [ok] NIRC2 / OSIRIS files reach the same parameters, reduction "
          "and DL PSF")


def make_frame(path):
    yy, xx = np.mgrid[0:256, 0:256]
    halo = np.exp(-((xx - 128) ** 2 + (yy - 128) ** 2) / (2 * 30.0 ** 2))
    star = 0.6 * circular_dl_psf(8.0, 12.0, 2.2, npix=256, obscuration_m=1.2,
                                 spiders=4, spider_width_m=0.05,
                                 spider_angle_deg=45.0, rotation_deg=10.0,
                                 pos=(0.3, 0.2)) + 0.4 * halo / halo.sum()
    img = np.random.default_rng(4).normal(0, 0.3, (256, 256)) + 5e5 * star
    fits.PrimaryHDU(img.astype(np.float32), header=fits.Header(
        {"INSTRUME": "MYINST", "FILTER": "K", "OBJECT": "synthetic"})
    ).writeto(path)


def main():
    check_parity()
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication(sys.argv[:1])
    night = os.path.join(DATA, "acct", "20261021", "raw")
    os.makedirs(night)
    make_frame(os.path.join(night, "img00003.fits"))
    make_frame(os.path.join(night, "img00003_raw.fits"))
    win = gui.MainWindow()
    win._nirc2_prefetch_guide_stars = lambda seq, cont: cont()
    log = win.n2_log.toPlainText()
    assert "! instrument file skipped: " in log and "broken.yaml" in log
    items = [win.n2_instrument.itemText(i)
             for i in range(win.n2_instrument.count())]
    assert items == ["NIRC2", "OSIRIS", "ORKID", "MYINST"], items
    win.n2_instrument.setCurrentText("MYINST")
    assert win._nirc2_kind() == "myinst" and win._nirc2_digits() == 5
    assert win.n2_prefix.text() == "img"
    print("  [ok] menu lists the files; a broken file is reported")
    win.n2_prefix.setText("")
    win.n2_path.setText(night)
    names = [win.n2_files.item(i).text() for i in range(win.n2_files.count())]
    assert names == ["img00003.fits"], names
    assert win.n2_prefix.text() == "img", "PREFIX fills in from the frames"
    win.n2_im1.setValue(3)
    win.n2_nim.setValue(1)
    win.n2_nbg.setValue(0)
    win._n2_loaded_files = None
    win._on_nirc2_go()
    pump(lambda: win.n2_go.isEnabled())
    sr = float(win.n2_strehl_out.text())
    assert 0.5 < sr < 0.7, (sr, win.n2_log.toPlainText()[-600:])
    print(f"  [ok] MYINST frame measured from its circular pupil: S={sr:.3f} "
          "(built with 0.6 of the DL core)")
    win.n2_path.setText("")
    win._on_nirc2_latest()
    pump(lambda: win.n2_latest.isEnabled())
    assert win.n2_path.text() == night, win.n2_path.text()
    print("  [ok] Latest finds MYINST's night directory from its file")
    win.close()
    print("gui_phase54: PASS")


if __name__ == "__main__":
    main()
