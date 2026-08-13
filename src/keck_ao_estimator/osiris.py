"""OSIRIS imager constants and FITS-header parameter extraction.

Python port of the K1 OSIRIS Strehl tool's instrument knowledge (the IDL
fork on k1aoserver: strehl_widget.pro MvD 2025 + find_strehl.pro Ragland
2014).  The measurement pipeline is shared with NIRC2 (image_strehl.py);
only the front end differs:

- camera 'osiris' at 9.95 mas/pixel (the post-2016 imager; the historic
  'osimg' 20.34 mas entry in the PSF model is the pre-upgrade imager);
- wavelength from the IFILTER header card via the tool's inline table
  (from the OSIRIS filter index page), NOT EFFWAVE;
- the reference PSF is always the OPEN pupil at the tool's pinned 38 deg
  (its own TODOs mark pupil handling unfinished -- we mirror, not
  improve), and the K1 white-light source (sfp) is a plain 11.14 m
  circular pupil;
- no flat, no bad-pixel mask, and no saturation ceiling: reduction is
  (image - background) followed by fix_image auto bad-pixel repair.
"""
from datetime import datetime

from .nirc2 import (
    Nirc2FrameParams, opt_header_float, opt_header_int, opt_lbwfs_fwhm,
    trick_sensor_active,
)

OSIRIS_PLATE_SCALE_MAS = 9.95           # post-2016 imager
OSIRIS_PMRANGL_DEG = 38.0               # find_strehl.pro's pinned value
OSIRIS_WL_PUPIL_M = 11.14               # sfp: white-light-source pupil diam

# IFILTER (lowercased) -> central wavelength in microns
# (find_strehl.pro; source: OSIRIS filter index page)
OSIRIS_FILTER_WAVELENGTH_UM = {
    "jn1": 1.2029, "jn2": 1.2581, "jn3": 1.3060,
    "hn1": 1.5037, "hn2": 1.5700, "hn3": 1.6351, "hn4": 1.6951,
    "hn5": 1.7642,
    "feii": 1.6475, "hcont": 1.5832,
    "zn3": 1.0868, "y": 1.0251, "j": 1.2429, "hbb": 1.6383,
    "kp": 2.1145,
    "kn1": 2.0044, "kn2": 2.0905, "kn3": 2.1756, "kn4": 2.2648,
    "kn5": 2.3498,
    "brgamma": 2.169, "brgamma-lhex": 2.169,
    "kcont": 2.2700, "hei_b": 2.0605, "zbb": 1.0915,
}
OSIRIS_UNKNOWN_FILTER_UM = 2.2          # the tool's fallback


def detect_instrument(header):
    """'nirc2' / 'osiris' / '' from a frame header.  NIRC2 writes
    INSTRUME; OSIRIS imager frames carry CURRINST (INSTRUME absent)."""
    instrume = str(header.get("INSTRUME", "") or "").upper()
    if "NIRC2" in instrume:
        return "nirc2"
    currinst = str(header.get("CURRINST", "") or "").upper()
    if "OSIRIS" in currinst or "OSIRIS" in instrume:
        return "osiris"
    return ""


def osiris_frame_params(header, sfp=False):
    """Extract measurement parameters from an OSIRIS imager header,
    mirroring the K1 tool's find_strehl.pro.  Returns the same params
    object the shared measurement core consumes."""
    filt = str(header.get("IFILTER", "")).strip().lower()
    effwave = OSIRIS_FILTER_WAVELENGTH_UM.get(filt, OSIRIS_UNKNOWN_FILTER_UM)

    coadds = int(header.get("COADDS", 1) or 1)

    utc = None
    date_obs = header.get("DATE-OBS")
    tval = header.get("UTC", header.get("UT"))
    if date_obs and tval:
        try:
            utc = datetime.fromisoformat(f"{date_obs}T{str(tval).split('.')[0]}")
        except ValueError:
            utc = None

    lsprop = str(header.get("LSPROP", "")).strip().lower()
    lgs = {"yes": True, "no": False}.get(lsprop)

    return Nirc2FrameParams(
        camname="osiris",
        pmsname="open",                     # the tool's pinned pupil call
        effwave_um=effwave,
        pmrangl_deg=OSIRIS_PMRANGL_DEG,
        coadds=coadds,
        max_counts=float("inf"),            # no saturation check in the tool
        daytime=False,
        plate_scale_mas=OSIRIS_PLATE_SCALE_MAS,
        utc=utc,
        object_name=str(header.get("OBJECT", "")).strip(),
        lgs=lgs,
        ra=str(header.get("RA", "") or "").strip(),
        dec=str(header.get("DEC", "") or "").strip(),
        sfp=bool(sfp),
        # same shared K1 AO-bench/telescope telemetry as NIRC2 (item 4's
        # structured Measured-SR log) -- None if this particular header
        # doesn't carry a given keyword
        az_deg=opt_header_float(header, "AZ"),
        el_deg=opt_header_float(header, "EL"),
        airmass=opt_header_float(header, "AIRMASS"),
        filter_name=str(header.get("IFILTER", "")).strip(),
        lbwfs_fwhm=opt_lbwfs_fwhm(header),
        aoopsmod=opt_header_int(header, "AOOPSMOD"),
        trick_active=trick_sensor_active(header),
    )
