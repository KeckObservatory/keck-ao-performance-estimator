"""NIRC2 instrument constants and FITS-header parameter extraction.

Python port of the K2 NIRC2 Strehl tool's instrument knowledge (the IDL
codebase by M. van Dam, A. Bouchez et al., kroot kss/nirc2/strehl): camera
plate scales, filter lookup tables, pupil-stop dimensions from the KAON 253
engineering drawings, and the header conventions (pupil-drive angle,
detector gain / saturation, daytime PCU source detection).

Everything here is pure data + tiny pure functions; the PSF machinery
lives in nirc2_psf.py and the measurement pipeline in image_strehl.py.
"""
from dataclasses import dataclass
from datetime import datetime

# camera plate scales in mas/pixel (find_strehl.pro; the more precise
# astrometric values, not the rounded 9.94 of the batch path)
NIRC2_PLATE_SCALE_MAS = {
    "narrow": 9.942,
    "medium": 19.829,
    "wide": 39.686,
}

# FITS FILTER card (spaces stripped) -> canonical filter name (which_filter.pro)
NIRC2_FILTER_NAMES = {
    "J+clear": "j",
    "K+clear": "k",
    "H+clear": "h",
    "Kp+clear": "kprime",
    "Ks+clear": "ks",
    "PK50_1.5+Kcont": "kcont",
    "PK50_1.5+Hcont": "hcont",
    "PK50_1.5+FeII": "feii",
    "PK50_1.5+Br_gamma": "brgamma",
    "PK50_1.5+Jcont": "jcont",
    "Lp+clear": "lprime",
    "Ms+clear": "ms",
    "PK50_1.5+NB2.108": "nb2.108",
    "clear+PAH": "pabeta",
    "PK50_1.5+He1_B": "heib",
}

# canonical filter name -> central wavelength in microns (central_wavelength.pro)
NIRC2_FILTER_WAVELENGTH_UM = {
    "hcont": 1.5804,
    "kcont": 2.2706,
    "j": 1.248,
    "h": 1.633,
    "k": 2.196,
    "ks": 2.146,
    "kprime": 2.124,
    "feii": 1.6455,
    "brgamma": 2.1686,
    "lprime": 3.776,
    "ms": 4.670,
    "nb2.108": 2.108,
    "jcont": 1.2132,
    "pabeta": 1.2903,
    "heib": 2.0563,
    "co": 2.2782,
}

# pupil-stop dimensions in inches at the pupil plane (nirc2pupil.pro, from
# the KAON 253 / NIRC2 pupil-stop engineering drawings).  For the hex stops
# the six numbers are [outer flat-to-flat/2 .. inner web] as used by the
# sextant vertex construction; INCIRCLE is [outer radius, inner radius,
# spider half-width].
NIRC2_PUPIL_STOPS = {
    "open": [0.49, 0.42, 0.350, 0.280, 0.0, 0.0],
    "largehex": [0.479, 0.4090, 0.3390, 0.2690, 0.1170, 0.0020],
    "mediumhex": [0.471, 0.4010, 0.3310, 0.2610, 0.1250, 0.0030],
    "smallhex": [0.451, 0.3810, 0.3110, 0.2410, 0.1450, 0.0030],
    "fixedhex": [0.474, 0.4090, 0.3350, 0.2690, 0.0000, 0.0100],
    "incircle": [0.392, 0.1325, 0.0030],
}
NIRC2_PMS_M_PER_INCH = 0.0899   # pupil-plane scale, m/inch (KAON 253)
NIRC2_PMR_ZERO_DEG = 38.0       # pupil-drive angle zeropoint (rough)
NIRC2_OPEN_SECONDARY_M = 1.30   # open-stop secondary cut radius, m
NIRC2_DAYTIME_PUPIL_M = 11.0    # circular pupil diameter for PCU source

# detector: ADU ceiling and transputer-era default gain (find_strehl.pro)
NIRC2_SATURATION_ADU = 32000.0
NIRC2_DEFAULT_DETGAIN = 4.0

# widget measurement defaults (strehl_widget.pro), radii in arcsec
NIRC2_PHOTOMETRY_RADIUS_ARCSEC = 1.0
NIRC2_BG_INNER_RADIUS_ARCSEC = 1.2
NIRC2_BG_OUTER_RADIUS_ARCSEC = 1.4
NIRC2_PEAK_RADIUS_ARCSEC = 0.1


@dataclass(frozen=True)
class Nirc2FrameParams:
    """Everything the Strehl measurement needs from a NIRC2 FITS header."""
    camname: str            # 'narrow' | 'medium' | 'wide'
    pmsname: str            # pupil stop, e.g. 'largehex'
    effwave_um: float       # effective wavelength, microns
    pmrangl_deg: float      # pupil rotation for the PSF model
    coadds: int
    max_counts: float       # saturation threshold per coadd, ADU
    daytime: bool           # PCU fiber source (calibration) vs on-sky
    plate_scale_mas: float
    utc: datetime | None    # frame timestamp, if the header carries one
    object_name: str
    lgs: bool | None = None     # LSPROP: laser propagating (None = unknown)
    ra: str = ""                # header RA/DEC, sexagesimal strings as
    dec: str = ""               #   NIRC2 writes them (FK5 J2000)
    sfp: bool = False           # OSIRIS: K1 white-light source in the beam
    # telemetry carried through for the Measured-SR structured log (item 4):
    # none of these feed the measurement itself, only the CSV export
    az_deg: float | None = None       # AZ: telescope azimuth
    el_deg: float | None = None       # EL: telescope elevation (also feeds
                                      #   pmrangl_deg above)
    airmass: float | None = None      # AIRMASS header value
    filter_name: str = ""             # raw FILTER string (effwave_um above
                                      #   is the resolved wavelength)
    lbwfs_fwhm: float | None = None   # AOLBFWHM: AO LBWFS avg fwhm
    aoopsmod: int | None = None       # AOOPSMOD: AO ops mode code, see
                                      #   decode_ao_ops_mode() below
    trick_active: bool | None = None  # DYYMASTR/DTSENSOR: TRICK (vs STRAP)
                                      #   was the real TT sensor for this
                                      #   frame, see trick_sensor_active()


def opt_header_float(header, key):
    """header[key] as a float, or None if absent/blank/unparseable -- for
    telemetry fields that are carried through for logging only (never fed
    into the measurement itself), where None must mean "not in this
    header" rather than a silently wrong 0.0."""
    v = header.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def opt_header_int(header, key):
    """header[key] as an int, or None if absent/blank/unparseable.

    Deliberately reads ONLY the exact key given -- callers must NOT add
    a fallback to a similarly-named key. AOOPSMOD is the concrete reason:
    a buggy header writer has, on at least one real frame, ALSO written a
    literal 9-character `AOOPSMODE` card (over FITS's 8-char keyword
    limit; astropy warns "invalid or non-standard convention" and stores
    the value as a quoted STRING, unlike the correctly-truncated
    `AOOPSMOD` card's bare int) -- ground-truthed in
    keck_ao_experiments/m79_slgs_vs_ltao_20260112/NIGHT_LOG_20260112.md.
    Falling back to it would silently read the wrong, unreliable card."""
    v = header.get(key)
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# AO Ops mode (AOOPSMOD): ground-truthed against real header/log data
# across multiple K1 nights in keck_ao_experiments (not guessed) -- see
# p5_conditions/LTAO_VS_SLGS_STORY.md, hd18770_recon_20251204/
# NIGHT_LOG_20251204.md, m3_poorcond_20260425/NIGHT_LOG_20260425.md,
# and THETA0_EVOLUTION.md's GLAO-naming note:
#   0 = NGS (also the code seen during pre-lock/checkout on some nights)
#   2 = single-LGS (sLGS)
#   3 = LTAO -- ALSO covers manual GLAO-recon configs run on LTAO
#       hardware pre-2026, and WFS2-reconstructor configs: "invisible to
#       the keyword" per the 1204 log, so this reports plain "LTAO" for
#       all of them, same as the experiments repo's own convention
#   4 = facility GLAO ("LTAO-wide-field"), established 2026+; did not
#       exist as a code in 2025 data (those nights' GLAO-like configs
#       are the ops=3 case above)
# Any other value -- AOOPSMOD=1 is on record on one out-of-scope daytime
# frame, "not catalogued, not guessed at" per that night's own log -- is
# reported as unknown rather than invented.
AO_OPS_MODE_NAMES = {0: "NGS", 2: "single-LGS", 3: "LTAO", 4: "GLAO"}


def decode_ao_ops_mode(code):
    """Human label for a raw AOOPSMOD code, or None if code is None."""
    if code is None:
        return None
    return AO_OPS_MODE_NAMES.get(code, f"unknown (AOOPSMOD={code})")


def opt_lbwfs_fwhm(header):
    """AOLBFWHM if the LBWFS was actually in the loop for this frame, else
    None. Eduardo (2026-07-28): the LBWFS isn't in use during NGS -- it's
    part of the laser path -- so its "avg fwhm" during an NGS frame isn't
    a real measurement, just a stale/residual reading from whenever light
    last reached it. Ground-truthed rather than inferred from AO mode:
    AOFCLBCT ("AO FC LBWFS control") is 'OFF' on every one of 382 real NGS
    frames and 'ON' on every one of 67 real laser-propagating frames in
    the datasets checked -- gate on that directly, the actual control-loop
    state, rather than re-deriving it from AOOPSMOD."""
    if str(header.get("AOFCLBCT", "")).strip().upper() != "ON":
        return None
    return opt_header_float(header, "AOLBFWHM")


def trick_sensor_active(header):
    """True if TRICK (not STRAP/WFS) was the real LGS tip-tilt sensor for
    THIS frame, per the AO bench's own telemetry: DYYMASTR=1 and
    DTSENSOR=3 (Eduardo 2026-07-28). None if either keyword is absent or
    non-numeric -- e.g. K2 frames, where DTSENSOR is a plain string enum
    ('STRAP'/'WFS') and TRICK does not exist as hardware at all -- so
    "no evidence either way" is never confused with "confirmed STRAP".

    This is a per-frame fact, unlike the live TT-sensor combo
    (_tt_sensor_band()): that combo is current UI state and can be left
    on a stale TRICK H/K selection from an earlier, unrelated target --
    the same staleness trap the NGS-band bug was. Ground truth here only
    ever narrows the STRAP-vs-TRICK question, not TRICK's H vs K -- these
    two keywords don't say which, so that half still needs the combo."""
    dyymastr = opt_header_int(header, "DYYMASTR")
    dtsensor = opt_header_int(header, "DTSENSOR")
    if dyymastr is None or dtsensor is None:
        return None
    return dyymastr == 1 and dtsensor == 3


def nirc2_frame_params(header):
    """Extract measurement parameters from a NIRC2 FITS header.

    Mirrors find_strehl.pro's 'nirc2' branch: EFFWAVE is trusted directly,
    the pupil angle is reconstructed from the rotator and elevation
    (pmrangl = -(ROTPPOSN - EL) + 38 + 90, the convention the pupil model
    was built for), the saturation ceiling is 32000/DETGAIN (transputer
    gain 4 assumed when DETGAIN is absent), and the frame counts as a
    daytime/PCU calibration only when the PCU is not at 'telescope' AND
    the AO hatch is closed.
    """
    camname = str(header.get("CAMNAME", "narrow")).strip().lower()
    if camname not in NIRC2_PLATE_SCALE_MAS:
        raise ValueError(f"unrecognized CAMNAME {camname!r}")
    pmsname = str(header.get("PMSNAME", "largehex")).strip().lower()

    effwave = float(header.get("EFFWAVE", 0.0) or 0.0)
    if effwave <= 0.0:
        filt = str(header.get("FILTER", "")).replace(" ", "")
        name = NIRC2_FILTER_NAMES.get(filt, "hcont")
        effwave = NIRC2_FILTER_WAVELENGTH_UM[name]

    rotpposn = float(header.get("ROTPPOSN", 0.0) or 0.0)
    el = float(header.get("EL", 0.0) or 0.0)
    pmrangl = -(rotpposn - el) + NIRC2_PMR_ZERO_DEG + 90.0

    detgain = float(header.get("DETGAIN", 0.0) or 0.0)
    if detgain == 0.0:
        detgain = NIRC2_DEFAULT_DETGAIN
    coadds = int(header.get("COADDS", 1) or 1)

    pcuname = str(header.get("PCUNAME", "")).replace(" ", "").upper()
    aohatch = str(header.get("AOHATCH", "")).replace(" ", "").upper()
    daytime = True
    if pcuname == "TELESCOPE":
        daytime = False
    if aohatch != "CLOSED":
        daytime = False

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
        camname=camname,
        pmsname=pmsname,
        effwave_um=effwave,
        pmrangl_deg=pmrangl,
        coadds=coadds,
        max_counts=NIRC2_SATURATION_ADU / detgain,
        daytime=daytime,
        plate_scale_mas=NIRC2_PLATE_SCALE_MAS[camname],
        utc=utc,
        object_name=str(header.get("OBJECT", "")).strip(),
        lgs=lgs,
        ra=str(header.get("RA", header.get("TARGRA", "")) or "").strip(),
        dec=str(header.get("DEC", header.get("TARGDEC", "")) or "").strip(),
        az_deg=opt_header_float(header, "AZ"),
        el_deg=el if header.get("EL") is not None else None,
        airmass=opt_header_float(header, "AIRMASS"),
        filter_name=str(header.get("FILTER", "")).strip(),
        lbwfs_fwhm=opt_lbwfs_fwhm(header),
        aoopsmod=opt_header_int(header, "AOOPSMOD"),
        trick_active=trick_sensor_active(header),
    )
