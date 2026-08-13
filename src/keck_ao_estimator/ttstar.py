"""TT-star position (and identity) from the AO headers: the
TSS-vs-pointing-origin odometer.

The tip/tilt sensor stage (AOTSX/AOTSY, mm) and the ACTIVE DCS pointing
origin (POXPOS/POYPOS, metres; PONAME names it) live in the same f/15
focal-plane coordinate frame, 1.375 arcsec/mm.  The stage physically
sits on the TT star, the pointing origin is where the telescope puts
the field -- so their per-frame difference IS the TT-star offset from
the field centre.  Ground-truthed on two OSIRIS nights (2026-07-25):

  * on-axis identity: with the TT star on-axis, TSS == PO to readback
    precision (M79 2026-01-31 seq 23: 0.001 mm = 1.4 mas; the GLAO
    night's base pointings likewise), and the per-frame values visibly
    track the bxy4 dither pattern (a corner-anchored 1" box reads as a
    0.70" mean displacement -- the pattern centroid, not a GS offset);
  * scale: commanded 15"/30" offsets read back 14.96"/29.93" (GLAO
    night, large diagonal moves) at 1.375 "/mm;
  * the M79 night's readbacks show the known TT-stage undershoot on
    small RA-axis increments (14.61" per commanded 15" step), matching
    the mosaic-measured 0.38"/step field creep -- the odometer reports
    DELIVERED position, which is exactly what makes it trustworthy.

The SEPARATION is rotation-free and always valid.  The DIRECTION is
not derivable from one frame: the bench-to-sky rotation differed by
exactly 60.0 deg between the two calibration nights at identical
ROTPOSN and PA_IMAG (stable within each night) -- a rotator-
configuration degeneracy that two nights cannot break.  Identity and
magnitude therefore come from RING MATCHING: query a catalogue around
the frame's target coordinates and keep stars whose radial separation
matches the measured one (direction-free; the header RA/DEC is
boresight-referenced, hence the default tolerance).
"""
import math

from .osiris import detect_instrument

TSS_ARCSEC_PER_MM = 1.375        # Keck f/15 focal-plane scale
TT_ONAXIS_MAX_ARCSEC = 1.0       # below this the star is called on-axis
# ring half-width for catalogue matching: dominated by the header RA/DEC
# being boresight-referenced, not field-centre-accurate -- measured 3.8"
# off on an on-axis 2026-02-26 frame (the M79 mosaics hit the same thing)
RING_TOL_ARCSEC = 5.0


def tt_star_offset(header):
    """TT-star offset from the field centre, from one frame's header.

    Returns dict(dx_mm, dy_mm, sep_arcsec, bench_pa_deg, on_axis,
    po_name) or None when the frame carries no TSS / pointing-origin
    keywords (non-AO data, or another facility's headers).  dx/dy are
    bench-frame millimetres (TSS minus PO); bench_pa_deg is the vector's
    angle in that frame -- NOT a sky direction (see module docstring).

    POXPOS/POYPOS's units are INSTRUMENT-DEPENDENT, not universal --
    the same K1-vs-K2 keyword-server gap this repo has hit before
    (AOOPSMOD/AOOPSMODE, DTSENSOR string-vs-int). The OSIRIS ground
    truth this module was built from used real metres (the x1000
    below). A real K2/NIRC2 night (2026-07-27) showed the identical
    POXPOS/POYPOS value (-0.1, 9.16) on EVERY one of 449 frames all
    night regardless of target/mode, with its own header card
    explicitly commented "[mm] Pointing origin X/Y position" -- and at
    a scale matching AOTSX/AOTSY/AOFMX/AOFMY (also mm) directly, no
    x1000 needed. Applying x1000 anyway inflated dy by ~9 metres,
    producing a nonsense ~12600" "separation" that could never match
    any real starlist entry -- the reason single-LGS guide-star
    resolution was silently failing to ASSUMED on every NIRC2 frame
    (ground-truthed on the 2026-07-27 on-sky session). NIRC2/K2 skips
    the x1000; anything else (OSIRIS/K1) keeps the original, still
    genuinely metres per that ground truth."""
    po_scale = 1.0 if detect_instrument(header) == "nirc2" else 1000.0
    try:
        dx = float(header["AOTSX"]) - po_scale * float(header["POXPOS"])
        dy = float(header["AOTSY"]) - po_scale * float(header["POYPOS"])
    except (KeyError, TypeError, ValueError):
        return None
    sep = math.hypot(dx, dy) * TSS_ARCSEC_PER_MM
    return dict(dx_mm=dx, dy_mm=dy, sep_arcsec=sep,
                bench_pa_deg=math.degrees(math.atan2(dy, dx)),
                on_axis=sep <= TT_ONAXIS_MAX_ARCSEC,
                po_name=str(header.get("PONAME", "")).strip())


TRICK_PLATE_SCALE_MAS = 50.0     # TRICK detector, Eduardo 2026-07-28
TRICK_ROI_CENTER_PX = 1020.0     # 102"/50mas = 2040 px, centred on target


def trick_roi_offset(header):
    """TRICK guide-star offset from the field centre, from the ROI
    position keywords -- an independent, TRICK-specific alternative to
    tt_star_offset's TSS-vs-pointing-origin odometer, for frames where
    TRICK (not STRAP) is the active TT sensor (see
    nirc2.trick_sensor_active). The TSS odometer tracks the STRAP stage
    specifically; there's no reason to expect it means anything while
    STRAP isn't the sensor in the loop, so this reads TRICK's own ROI
    instead: TRKRO1XP/YP is the ROI's CENTER pixel (Eduardo 2026-07-28),
    on a 102"x102" / 50 mas-per-pixel detector CENTERED ON THE TARGET
    (2040 px across, so pixel 1020 = zero offset).

    Returns dict(dx_px, dy_px, sep_arcsec, on_axis) or None when the
    frame carries no TRKRO1XP/YP (non-TRICK data). Like the TSS
    odometer, the SEPARATION alone is trustworthy standalone -- plain
    Pythagorean distance from a fixed, confirmed centre pixel and plate
    scale, no rotation/parity assumption involved -- and safe to feed
    directly into the physics (lgs_strehl's tt_offset). The DIRECTION
    (needed only to query a catalogue at the star's real RA/Dec) is a
    SEPARATE, less-certain question -- see trick_roi_sky_offset."""
    try:
        xp = float(header["TRKRO1XP"])
        yp = float(header["TRKRO1YP"])
    except (KeyError, TypeError, ValueError):
        return None
    dx_px = xp - TRICK_ROI_CENTER_PX
    dy_px = yp - TRICK_ROI_CENTER_PX
    sep = math.hypot(dx_px, dy_px) * (TRICK_PLATE_SCALE_MAS / 1000.0)
    return dict(dx_px=dx_px, dy_px=dy_px, sep_arcsec=sep,
                on_axis=sep <= TT_ONAXIS_MAX_ARCSEC)


def trick_roi_sky_offset(header):
    """(d_ra_east_arcsec, d_dec_north_arcsec) of the TRICK ROI relative
    to the field centre, for querying a catalogue to identify the guide
    star and its magnitude -- the one piece trick_roi_offset's docstring
    flags as NOT fully confirmed.

    *** ORIENTATION IS UNCONFIRMED -- READ BEFORE TRUSTING A TRICK
    OFF-AXIS IDENTITY MATCH. *** Eduardo 2026-07-28: "same orientation
    as OSIRIS's science image, though I am not 100% sure." Checked and
    RULED OUT as the reference for that: `OSIRIS_PMRANGL_DEG` (osiris.py,
    38 deg) is the PUPIL SEGMENT rotation for the diffraction-PSF model
    (nirc2_psf.py's hexagon geometry) -- a completely different physical
    angle from detector-to-sky image orientation. No confirmed OSIRIS/
    TRICK image position-angle constant exists elsewhere in this repo
    to check this against.

    Working assumption used here, absent anything better: TRICK's pixel
    axes are North-up/East-left with NO rotation -- i.e. the same
    ΔRA-East/ΔDec-North sign convention this GUI's own manual offset
    entries already use elsewhere (fieldmap_overlays.py's "Put TT/NGS
    star here": ΔRA-East = -x, ΔDec-North = y).

    IF A TRICK OFF-AXIS MATCH LOOKS WRONG, START HERE: the failure mode
    is a star assigned a plausible-looking but WRONG catalogue identity/
    magnitude (the ROI separation is still correct via trick_roi_offset;
    only the compass direction would be off), not an obvious crash --
    same class of bug as the 2026-07-18 WCS parity lesson. To verify:
    take a night where the guide star's real identity is independently
    known off-axis (on-axis frames give no signal either way -- sep~0),
    e.g. keck_ao_experiments/m92_ttmag_20260501/NIGHT_LOG_20260501.md's
    ROI/H-mag rows, compute this function's predicted RA/Dec, and check
    it against that star's real catalogue position. If wrong, there are
    only 4 candidate parities (swap X/Y, and/or flip either sign) --
    try them in turn against that same confirmed row."""
    try:
        xp = float(header["TRKRO1XP"])
        yp = float(header["TRKRO1YP"])
    except (KeyError, TypeError, ValueError):
        return None
    scale = TRICK_PLATE_SCALE_MAS / 1000.0
    d_ra_east = -(xp - TRICK_ROI_CENTER_PX) * scale
    d_dec_north = (yp - TRICK_ROI_CENTER_PX) * scale
    return d_ra_east, d_dec_north


def tt_ring_match(stars, ra_deg, dec_deg, sep_arcsec,
                  tol_arcsec=RING_TOL_ARCSEC):
    """Catalogue stars whose separation from (ra_deg, dec_deg) matches
    the TSS-measured ring, sorted by |mismatch|.

    stars: catalogs.query_guide_stars dicts ({id, ra, dec, mags}).
    Returns [{star, sep_arcsec, dsep_arcsec}, ...]; on-axis (sep ~ 0)
    degenerates naturally to nearest-the-centre matching."""
    cosd = math.cos(math.radians(dec_deg))
    out = []
    for s in stars:
        d_ra = (float(s["ra"]) - ra_deg) * cosd * 3600.0
        d_de = (float(s["dec"]) - dec_deg) * 3600.0
        sep = math.hypot(d_ra, d_de)
        dsep = sep - sep_arcsec
        if abs(dsep) <= tol_arcsec:
            out.append(dict(star=s, sep_arcsec=sep, dsep_arcsec=dsep))
    out.sort(key=lambda r: abs(r["dsep_arcsec"]))
    return out


def best_mag(star, prefer=("R", "V", "r", "G", "B")):
    """(band, mag) of the star's best-available magnitude, preferring
    STRAP's R band; (None, None) when the catalogue row has none."""
    mags = star.get("mags", {}) or {}
    for band in prefer:
        if mags.get(band) is not None:
            return band, float(mags[band])
    for band, v in mags.items():
        if v is not None:
            return band, float(v)
    return None, None
