"""Tilt-Sensor-Stage (TSS) reachability and vignetting for the K1 tip-tilt
sensor -- how far off axis a guide star can be picked up at all, and how much
light the sensor loses when it is.

*** THIS IS A MODEL, NOT THE MEASURED MAP ***
The real vignetting map is three data files the AO Guide Star Tool and the TSS
widget read -- illsubaps_tssx.dat, illsubaps_tssy.dat, illsubaps.dat. KAON 913
says the K1 versions are tabulated in it; the copy we have
(skycoverage/"Keck I Low Bandwidth Wavefront Sensor Vignetting and Field
Curvature.docx") does NOT contain those tables, and the files are not on this
machine. Everything below is reconstructed from what the note does carry, and
the reconstruction is deliberately coarse where the evidence is thin. Eduardo
2026-08-07: "for now just code it as a model I dont know where these data
files are." If the .dat files turn up, `vignetting_fraction` is the ONE
function to replace -- everything else keys off it.

PROVENANCE -- KAON 913, "Keck I Low Bandwidth Wavefront Sensor Vignetting and
Field Curvature, and Tilt Sensor Stage Properties", P. Wizinowich & J. Chin,
2012-04-24. Two caveats that belong on every number here:

  * It is an LBWFS note. Its applicability to STRAP rests on one sentence --
    "These results also apply to the tip-tilt sensor (STRAP)" -- and TSS x/y/z
    are STACKED stages carrying both sensors, so the travel limits are shared
    but the optical vignetting need not be identical.
  * The 60" unvignetted radius is set by the ROTATOR, and the measurements
    were taken at rotator = 45 deg. The note itself says vignetting "may" need
    re-measuring at rotator + 90 deg. So the true map is probably
    rotator-dependent, and a static model cannot see that.

MEASURED CONTENT actually used (TSS device coordinates, the Maori-GUI frame):
  * travel limits          x in [-44, +59] mm, y in [-69, +52] mm
  * unvignetted radius     43.6 mm = 60" (rotator-limited)
  * vignetting samples     (1.59, -0.04) -> 0 %, (-14.26, -30.03) -> 5 %,
                           (40.75, +19.97) -> 15 %   [3 points, that is all]
  * instrument on-axis     OSIRIS imager (1.58, -10.04) mm,
                           spectrograph (1.56, +5.46) mm  -- 21.3" apart
  * plate scale            43.6 mm <-> 60" => 1.376 "/mm, which independently
                           confirms ttstar.TSS_ARCSEC_PER_MM = 1.375

WHY THIS IS RADIAL. The travel box is in the BENCH frame; the field map and
the guide-star catalogue are in the SKY frame, and the bench-to-sky rotation
needs a rotator angle we do not carry (and which this project has settled,
painfully, must never be taken from recollection -- see the 2026-08-01
detector-orientation notes). So the default model is the rotation-INVARIANT
reduction of the box, which needs no angle and cannot be wrong about one:

    r <= INSCRIBED (60.5")  reachable at EVERY rotator angle, unvignetted
    r <= CIRCUMSCRIBED (95") reachable at SOME rotator angles, vignetted
    r >  CIRCUMSCRIBED       unreachable at any angle

The inscribed radius of the travel box is min(44, 59, 69, 52) = 44 mm =
60.5", which lands on the rotator's own 60" unvignetted radius from a
completely independent direction. That coincidence is what makes the radial
reduction cheap: inside 60" both limits agree, and outside it the honest
statement is "depends on the rotator", which is exactly what the model says.
`tss_box_reachable()` is available for a caller that DOES know the bench
angle.
"""
import numpy as np

from .ttstar import TSS_ARCSEC_PER_MM

# --- measured, exact (KAON 913) ---------------------------------------------
TSS_TRAVEL_MM = {"x": (-44.0, 59.0), "y": (-69.0, 52.0)}   # device coords
TSS_UNVIGNETTED_RADIUS_MM = 43.6            # = 60", rotator-limited
INSTRUMENT_CENTRE_MM = {                    # TSS device coords of "on axis"
    "osiris-imager": (1.58, -10.04),
    "osiris-spec":   (1.56, 5.46),
    "optical-axis":  (1.59, -0.04),         # the 0 %-vignetting test point
}
DEF_INSTRUMENT = "optical-axis"

#  Rotation-invariant reduction of the travel box (see the module header).
TSS_INSCRIBED_MM = min(abs(TSS_TRAVEL_MM["x"][0]), TSS_TRAVEL_MM["x"][1],
                       abs(TSS_TRAVEL_MM["y"][0]), TSS_TRAVEL_MM["y"][1])
TSS_CIRCUMSCRIBED_MM = max(abs(TSS_TRAVEL_MM["x"][0]), TSS_TRAVEL_MM["x"][1],
                           abs(TSS_TRAVEL_MM["y"][0]), TSS_TRAVEL_MM["y"][1])
TSS_INSCRIBED_ARCSEC = TSS_INSCRIBED_MM * TSS_ARCSEC_PER_MM          # 60.5"
TSS_CIRCUMSCRIBED_ARCSEC = TSS_CIRCUMSCRIBED_MM * TSS_ARCSEC_PER_MM  # 94.9"
UNVIGNETTED_RADIUS_ARCSEC = TSS_UNVIGNETTED_RADIUS_MM * TSS_ARCSEC_PER_MM

#  --- the fitted curve: TWO parameters through TWO points ---------------
#  KAON 913's three vignetting samples, converted to field radius:
#      2.2" -> 0 %,   45.7" -> 5 %,   62.4" -> 15 %
#  The on-axis point only confirms the origin, so the curve
#      v(r) = VIGNETTE_COEFF * (r / UNVIGNETTED_RADIUS_ARCSEC) ** VIGNETTE_EXP
#  has ZERO degrees of freedom left: both parameters are consumed by the two
#  off-axis points, and NOTHING in the note validates the shape between or
#  beyond them. A power law is chosen only because vignetting must be
#  monotonic, zero at the origin, and accelerating (the pupil is being
#  progressively occulted) -- not because the data prefers it over anything
#  else. Treat the 5 % and 15 % radii as the trustworthy part and the curve
#  between them as interpolation.
#  Sanity: at the outer travel limit (94.9") it extrapolates to ~0.66, i.e. it
#  never claims total occultation inside the reachable field. The star becomes
#  UNUSABLE by leaving the box, not by this curve reaching 1.
VIGNETTE_EXP = 3.53      # ln(0.15/0.05) / ln(62.4/45.7)
VIGNETTE_COEFF = 0.131   # pins v(62.4") = 0.15
VIGNETTE_SAMPLES = ((2.2, 0.00), (45.7, 0.05), (62.4, 0.15))   # for tests/docs

#  A star this heavily vignetted is not worth ranking: the sensor is seeing
#  well under half the flux and the note has no measurement out here anyway.
VIGNETTE_UNUSABLE_FRAC = 0.50


def field_centre_mm(instrument=DEF_INSTRUMENT):
    """TSS device coordinates (mm) of the on-axis position for `instrument`.
    The OSIRIS imager and spectrograph sit 21.3" apart, so which one is in use
    genuinely moves the field centre -- and therefore which guide stars are
    reachable. Unknown names fall back to the optical axis."""
    return INSTRUMENT_CENTRE_MM.get(instrument,
                                    INSTRUMENT_CENTRE_MM[DEF_INSTRUMENT])


def instrument_centre_offset_arcsec(instrument=DEF_INSTRUMENT):
    """How far `instrument`'s on-axis position sits from the optical axis
    (arcsec) -- the amount every field radius shifts when it is selected."""
    x, y = field_centre_mm(instrument)
    x0, y0 = INSTRUMENT_CENTRE_MM[DEF_INSTRUMENT]
    return float(np.hypot(x - x0, y - y0) * TSS_ARCSEC_PER_MM)


def vignetting_fraction(offset_arcsec):
    """Fraction of the tip-tilt sensor's subapertures vignetted (0..1) for a
    guide star `offset_arcsec` from the field centre.

    THE MODEL, not the measured map -- see the module header. Radial by
    construction (the bench-to-sky rotation is unknown), monotonic, 0 on axis,
    and clipped to 1. Beyond the stage travel this still returns a number;
    ask `tss_reachable()` whether the star can be acquired at all, because a
    modest vignetting fraction out there is meaningless if the stage cannot
    get to it."""
    r = np.asarray(offset_arcsec, dtype=float)
    v = VIGNETTE_COEFF * (np.maximum(r, 0.0)
                          / UNVIGNETTED_RADIUS_ARCSEC) ** VIGNETTE_EXP
    return np.clip(v, 0.0, 1.0)


def vignetting_mag_penalty(offset_arcsec):
    """The vignetting expressed as magnitudes of lost flux, so it can be added
    straight to a guide-star magnitude before the sensor's own noise model
    runs: dm = -2.5 log10(1 - v).

    This is the honest way to charge vignetting in this engine. Every TT
    measurement row (STRAP and TRICK alike) is parameterized by SENSING-BAND
    MAGNITUDE, so a flux loss belongs in the magnitude, not as a second
    ad-hoc term bolted onto the error budget -- and it then propagates through
    whichever row is active without being calibrated twice. Returns inf for
    total occultation."""
    v = vignetting_fraction(offset_arcsec)
    with np.errstate(divide="ignore"):
        return -2.5 * np.log10(np.clip(1.0 - v, 0.0, 1.0))


def tss_reachable(offset_arcsec):
    """Can the TSS put the tip-tilt sensor on a star this far off axis?

    Returns (reachable, certainty, why) where certainty is:
      'always'  -- inside the inscribed radius; true at every rotator angle
      'depends' -- inside the box's circumscribed radius but outside the
                   inscribed one, so it depends on the bench angle, which
                   this engine does not carry
      'never'   -- outside the stage travel at any angle
    `reachable` is True for 'always' and 'depends' (refusing a star that is
    merely angle-dependent would hide usable guide stars), so callers should
    branch on `certainty` when they want to warn."""
    r = float(offset_arcsec)
    if r <= TSS_INSCRIBED_ARCSEC:
        return True, "always", ""
    if r <= TSS_CIRCUMSCRIBED_ARCSEC:
        return True, "depends", (
            f"{r:.0f}\" is outside the TSS travel's guaranteed "
            f"{TSS_INSCRIBED_ARCSEC:.0f}\" circle -- reachable only at "
            f"favourable rotator angles (stage box "
            f"{TSS_TRAVEL_MM['x'][0]:g}..{TSS_TRAVEL_MM['x'][1]:g} x "
            f"{TSS_TRAVEL_MM['y'][0]:g}..{TSS_TRAVEL_MM['y'][1]:g} mm)")
    return False, "never", (
        f"{r:.0f}\" exceeds the TSS stage travel "
        f"({TSS_CIRCUMSCRIBED_ARCSEC:.0f}\" at the most favourable rotator "
        "angle) -- the tip-tilt sensor cannot be placed on this star")


def tss_box_reachable(dx_mm, dy_mm):
    """Exact travel-limit test in TSS DEVICE coordinates -- the real,
    asymmetric box, for a caller that knows the bench-frame position (e.g.
    the TSS odometer in ttstar.py, which reads dx/dy straight out of the
    header). No rotation assumption is made or needed here."""
    xlo, xhi = TSS_TRAVEL_MM["x"]
    ylo, yhi = TSS_TRAVEL_MM["y"]
    return bool(xlo <= float(dx_mm) <= xhi and ylo <= float(dy_mm) <= yhi)


def vignetting_note(offset_arcsec):
    """One human sentence for a log / tooltip, or "" when there is nothing to
    say (unvignetted and safely reachable)."""
    v = float(vignetting_fraction(offset_arcsec))
    ok, certainty, why = tss_reachable(offset_arcsec)
    bits = []
    if v >= 0.01:
        bits.append(f"~{100 * v:.0f}% vignetted "
                    f"(+{float(vignetting_mag_penalty(offset_arcsec)):.2f} mag "
                    "of lost flux; modelled, see KAON 913)")
    if why:
        bits.append(why)
    return "; ".join(bits)
