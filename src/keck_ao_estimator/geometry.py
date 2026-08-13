"""Telescope-pointing geometry: airmass/alt-az from astropy, and the Keck
horizon / Nasmyth-deck-shadow pointing-limit classification."""
import re
from datetime import timedelta

from astropy.coordinates import Angle, SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u

# Reference epoch proper motion is measured FROM. J2000.0 is the standard
# SIMBAD/Gaia catalogue-position epoch, so a PM resolved via SIMBAD (or typed
# by hand against a J2000 catalogue value) propagates correctly with no
# further epoch bookkeeping needed from the caller.
PM_REF_EPOCH = "J2000"

from .constants import KECK_LAT_DEG, KECK_LON_DEG, KECK_HEIGHT_M, \
    HST_TO_UTC_HOURS, POINTING_LIMITS

_HMS_DMS_RE = re.compile(r"[hdms]", re.IGNORECASE)


def _parse_angle(value, colon_unit):
    """One coordinate component, in whichever of the three TEXT formats it's
    written in (detected independently per component, so RA and Dec need not
    match each other's format). A non-string (already an Angle/Quantity, as
    callers that build a SkyCoord programmatically may pass) goes straight to
    Angle() unchanged -- this must keep accepting exactly what plain
    SkyCoord(ra=, dec=) always has."""
    if not isinstance(value, str):
        return Angle(value)
    s = value.strip()
    if _HMS_DMS_RE.search(s):
        return Angle(s)                       # self-describing, e.g. 15h49m57.7s
    if ":" in s:
        return Angle(s, unit=colon_unit)       # colon sexagesimal, e.g. 15:49:57.7
    return Angle(float(s), unit=u.deg)         # plain decimal degrees, e.g. 237.49


def parse_radec(ra, dec):
    """Parse a target RA/Dec pair into a SkyCoord. Each component accepts,
    independently:
      * self-describing hms/dms, e.g. "15h49m57.7s" / "-03d55m16s"
      * colon-separated sexagesimal, e.g. "15:49:57.7" (RA, hours) /
        "-03:55:16" (Dec, degrees) -- the common DS9/pipeline convention
      * plain decimal degrees, e.g. "237.49" (RA) / "-3.92" (Dec)
      * an already-parsed Angle/Quantity (passed through unchanged)
    Raises the same way SkyCoord/Angle would on unparseable input."""
    return SkyCoord(ra=_parse_angle(ra, u.hourangle),
                    dec=_parse_angle(dec, u.deg), frame="icrs")


def apply_proper_motion(coord, pm_ra_cosdec_masyr, pm_dec_masyr, obs_date,
                        ref_epoch=PM_REF_EPOCH):
    """Propagate an ICRS SkyCoord from ref_epoch (default J2000) to obs_date
    by proper motion alone -- no parallax/radial velocity needed. Astropy's
    apply_space_motion works without a distance (it treats the star as
    effectively at infinite distance), which is exactly the pure angular-rate
    correction wanted here: "where has this target's position on the sky
    drifted to by observation time", not a 3-D space position.

    pm_ra_cosdec_masyr / pm_dec_masyr: proper motion in mas/yr, SIMBAD/Gaia's
    PMRA/PMDEC convention (PMRA is already *cos(dec) -- the rate of change of
    RA angle on the sky, not of the RA coordinate itself).
    obs_date: a date/datetime with .isoformat() (only the calendar date
    matters -- PM effects accumulate over YEARS, so sub-day timing is
    irrelevant).

    Both 0 (or None) returns coord UNCHANGED -- not even re-wrapped through
    SkyCoord -- so a target with no known proper motion is byte-identical to
    every call site from before this function existed."""
    if not pm_ra_cosdec_masyr and not pm_dec_masyr:
        return coord
    import warnings
    from erfa import ErfaWarning
    c = SkyCoord(ra=coord.ra, dec=coord.dec, frame="icrs",
                pm_ra_cosdec=(pm_ra_cosdec_masyr or 0.0) * u.mas / u.yr,
                pm_dec=(pm_dec_masyr or 0.0) * u.mas / u.yr,
                obstime=Time(ref_epoch))
    with warnings.catch_warnings():
        # ERFA warns that it substituted a large default distance because
        # none was given -- exactly the intended no-parallax, angle-only
        # propagation (see docstring), not a real problem to surface.
        warnings.simplefilter("ignore", ErfaWarning)
        return c.apply_space_motion(new_obstime=Time(obs_date.isoformat()))


def compute_airmass_curve(ra, dec, grid_dts_hst):
    """Airmass of the target at each HST datetime, as seen from Keck.
    Returns (airmass_array, elevation_deg_array, azimuth_deg_array)."""
    target = parse_radec(ra, dec)
    keck   = EarthLocation(lat=KECK_LAT_DEG * u.deg,
                           lon=KECK_LON_DEG * u.deg,
                           height=KECK_HEIGHT_M * u.m)
    times_utc = Time([t + timedelta(hours=HST_TO_UTC_HOURS) for t in grid_dts_hst])
    altaz = target.transform_to(AltAz(obstime=times_utc, location=keck))
    return altaz.secz.value, altaz.alt.deg, altaz.az.deg


def sun_altitude_deg(when_utc=None):
    """Sun altitude above the horizon at Keck, degrees, at `when_utc` (a naive
    UTC datetime; None = now). Positive = sun up."""
    from datetime import datetime, timezone
    from astropy.coordinates import get_sun
    if when_utc is None:
        when_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    keck = EarthLocation(lat=KECK_LAT_DEG * u.deg, lon=KECK_LON_DEG * u.deg,
                         height=KECK_HEIGHT_M * u.m)
    t = Time(when_utc)
    return float(get_sun(t).transform_to(
        AltAz(obstime=t, location=keck)).alt.deg)


def moon_separation_deg(ra, dec, when_utc=None):
    """Angular separation (deg) between a target and the Moon, as seen from
    Keck, at `when_utc` (a naive UTC datetime; None = now). Geocentric moon
    position (get_body("moon", ...) without `location`) is accurate enough
    for a starlist-picker "how close is the Moon" readout -- the parallax
    shift from Earth's radius vs the Moon's ~384000 km distance is at most
    ~1 deg, immaterial for the coarse avoid-the-Moon judgement this serves."""
    from datetime import datetime, timezone
    from astropy.coordinates import get_body
    if when_utc is None:
        when_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    t = Time(when_utc)
    target = parse_radec(ra, dec)
    moon = get_body("moon", t)
    return float(target.separation(moon).deg)


def moon_illumination_fraction(when_utc=None):
    """Illuminated fraction of the Moon's disk at `when_utc` (naive UTC
    datetime; None = now): 0.0 = new moon, 1.0 = full moon. Standard low-
    precision approximation (Meeus, "Astronomical Algorithms" ch. 48):
    k = (1 - cos(psi)) / 2, where psi is the geocentric Sun-Moon angular
    separation -- exact in the limit Earth-Sun distance >> Earth-Moon
    distance (true to ~0.1%, more than enough for a planning-tool percentage
    readout). Geocentric (no Keck-specific parallax correction), same
    precision level as moon_separation_deg above."""
    import math
    from datetime import datetime, timezone
    from astropy.coordinates import get_body
    if when_utc is None:
        when_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    t = Time(when_utc)
    sun = get_body("sun", t)
    moon = get_body("moon", t)
    psi = sun.separation(moon).radian
    return float((1.0 - math.cos(psi)) / 2.0)


def hour_angle_hours(ra, when_utc=None):
    """Hour angle (hours, signed to (-12, +12]) of `ra` at Keck's longitude,
    at `when_utc` (naive UTC datetime; None = now). HA = LST - RA: negative
    means the target hasn't yet crossed the meridian (rising), positive
    means it has (setting/past transit). Depends only on RA and time/
    longitude -- unlike az/el, no Dec or latitude is involved, so this
    doesn't need a target-Dec argument at all."""
    from datetime import datetime, timezone
    if when_utc is None:
        when_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    t = Time(when_utc)
    lst = t.sidereal_time("apparent", longitude=KECK_LON_DEG * u.deg)
    ra_ang = _parse_angle(ra, u.hourangle)
    ha = (lst - ra_ang).wrap_at("180d")
    return float(ha.hour)


def is_night_at_keck(when_utc=None, horizon_deg=-0.833):
    """True between sunset and sunrise at Keck: the sun's altitude is below
    the standard rise/set horizon of -0.833 deg (atmospheric refraction +
    the solar semi-diameter -- the same definition almanac sunset/sunrise
    times use). Used by the GUI's Nighttime mode as its day/night gate."""
    return sun_altitude_deg(when_utc) < horizon_deg


def in_wedge(az_deg, telescope):
    """True if the azimuth falls in the telescope's Nasmyth-deck shadow wedge."""
    a0, a1 = POINTING_LIMITS[telescope]["wedge"]
    return a0 <= az_deg <= a1


def pointing_state(elev_deg, az_deg, telescope):
    """Classify a target position for the given telescope into one of:
        'open'     : unvignetted (elev >= the az-dependent floor, <= ceiling)
        'vignetted': 0 <= elev < 18 deg and NOT blocked by the deck
        'blocked'  : below the wedge floor while inside the wedge, or below 0,
                     or above the ceiling -> not observable / off-limits
    """
    lim = POINTING_LIMITS[telescope]
    if elev_deg > lim["ceiling"] or elev_deg < 0:
        return "blocked"
    floor = lim["wedge_floor"] if in_wedge(az_deg, telescope) else lim["open_floor"]
    if elev_deg >= floor:
        return "open"
    # below the unvignetted floor:
    if in_wedge(az_deg, telescope):
        # inside the wedge, below the (high) deck floor -> fully blocked
        return "blocked"
    # outside the wedge, 0..18 deg -> vignetted
    return "vignetted"
