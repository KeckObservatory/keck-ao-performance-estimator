"""Which star was the MKAM MASS/DIMM actually watching?

The Maunakea seeing monitor (the MASS/DIMM on the MKAM tower by CFHT)
observes ONE bright star at a time from a fixed target catalog, and its
pointing is not logged with the seeing data.  That pointing sets the
monitor's turbulence pierce points, so any advection lead/lag argument
(fa_advisory.event_lead_lag) needs it.  This module replaces a
hand-postulated pointing with a PROBABILISTIC model driven by the real
catalog: given the observation times, rank every catalog star by how
strongly a stay-near-zenith scheduler would prefer it, and hand the top
few (az, el) orientations to the FA-geometry plot.

Catalog
-------
data/catalog_MKAM.zip carries the monitor software's target lists:

  catalog.MASS  -- the operational whole-sky list: ~130 HR-catalog stars
                   (V <~ 3) with RA/Dec (J2000), V, B-V, an SED bin used
                   by MASS spectral weighting, and duplicity notes.
                   This is the list the model draws from.
  catalog.robo  -- IGNORED: an 8-star all-southern (dec -34..-61) example
                   list from the software distribution (a southern-site
                   RoboDIMM file); those stars never rise near zenith at
                   +19.8 deg latitude.

Parsing notes (ground-truthed against the shipped file): the name field
is fixed-width (cols 5-12, "Gam Gem" / "Mu 1Sco" / "Omi2CMa"), the rest
token-splits.  A hand-edited header block duplicates six bright stars
with "---" HR numbers and placeholder B-V/duplicity; the loader DEDUPES
by name (preferring the HR-numbered row) so one physical star cannot
split its scheduler probability across two rows.  One line carries a
sign typo ("- 11 -9 41" for Spica's declination): dec components are
parsed as magnitudes under the leading sign.

Scheduler model
---------------
The monitor "optimizes to stay near zenith" (its documented operating
practice) within an elevation limit, on single stars:

  * hard eligibility: el >= min_el_deg (default 45, matching the pierce
    region drawn in fa_geometry_plot) and no comparably-bright companion
    (sep <= max_companion_sep_arcsec with dmag < min_companion_dmag)
    that would corrupt the differential/scintillation measurement;
  * soft preference: at each sample time the eligible stars are weighted
    exp(-(ZD - ZD_best)/softness_deg).  The softness absorbs what we do
    NOT know about the real scheduler -- switch hysteresis (it stays on
    the last star while a marginally better one rises), acquisition
    timing, and operator overrides -- instead of pretending the argmin
    star is certain;
  * the per-time weights are normalized and averaged over the window:
    P(star) = mean_t w_i(t).

Alt/az here is plain-math (GMST + hour angle on the J2000 coordinates,
no precession/nutation/refraction): worst-case ~0.4 deg against a full
transform in 2026, irrelevant for ranking stars degrees apart but NOT a
pointing-grade transform.  Stdlib-only on purpose.
"""
import datetime as dt
import io
import math
import os
import re
import zipfile

from .constants import KECK_LAT_DEG, KECK_LON_DEG

MKAM_CATALOG_ZIP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "data", "catalog_MKAM.zip")
MKAM_CATALOG_MEMBER = "catalog.MASS"

# Bayer-abbreviation -> greek letter, for figure labels ("Gam Gem" -> "γ Gem")
_GREEK = {
    "Alp": "α", "Bet": "β", "Gam": "γ", "Del": "δ", "Eps": "ε", "Zet": "ζ",
    "Eta": "η", "The": "θ", "Iot": "ι", "Kap": "κ", "Lam": "λ", "Mu": "μ",
    "Nu": "ν", "Xi": "ξ", "Omi": "ο", "Pi": "π", "Rho": "ρ", "Sig": "σ",
    "Tau": "τ", "Ups": "υ", "Phi": "φ", "Chi": "χ", "Psi": "ψ", "Ome": "ω",
}
_SUPERSCRIPT = {"1": "¹", "2": "²", "3": "³"}

_SEP_RE = re.compile(r'([\d.]+)"')
_DM_RE = re.compile(r"dm=([\d.]+)")


# the 7-char name field packs Bayer + optional component + constellation
# with shifting spacing ("Gam Gem", "Mu1 Sco", "Bet1Sco", "Mu 1Sco")
_NAME_RE = re.compile(r"^([A-Za-z]+?)\s*([123])?\s*([A-Z][A-Za-z]{2})$")


def _pretty_name(name7):
    """'Gam Gem' -> 'γ Gem', 'Mu1 Sco'/'Mu 1Sco' -> 'μ¹ Sco'; unknown
    forms pass through whitespace-collapsed."""
    collapsed = " ".join(name7.split())
    m = _NAME_RE.match(collapsed)
    if m and m.group(1) in _GREEK:
        comp = _SUPERSCRIPT.get(m.group(2) or "", "")
        return _GREEK[m.group(1)] + comp + " " + m.group(3)
    return collapsed


def _parse_line(line):
    """One catalog.MASS row -> star dict, or None for headers/blanks.
    Raises ValueError on a structurally broken line."""
    if not line.strip() or line.lstrip().startswith("#"):
        return None
    hr = line[:4].strip()
    name = line[5:12]
    toks = line[12:].split()
    if len(toks) < 9:
        raise ValueError(f"short catalog line: {line!r}")
    ra_deg = 15.0 * (float(toks[0]) + float(toks[1]) / 60.0
                     + float(toks[2]) / 3600.0)
    sign = -1.0 if toks[3] == "-" else 1.0
    # abs() eats the file's stray inner minus ("- 11 -9 41" for Spica)
    dec_deg = sign * (abs(float(toks[4])) + abs(float(toks[5])) / 60.0
                      + abs(float(toks[6])) / 3600.0)
    dup = " ".join(toks[11:]) if len(toks) > 11 else ""
    sep_m = _SEP_RE.search(dup)
    dm_m = _DM_RE.search(dup)
    return {
        "hr": int(hr) if hr.isdigit() else None,
        "name": " ".join(name.split()),
        "pretty": _pretty_name(name),
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "vmag": float(toks[7]),
        "bv": float(toks[8]),
        "sptype": toks[10] if len(toks) > 10 else "",
        "sep_arcsec": float(sep_m.group(1)) if sep_m else None,
        "dmag": float(dm_m.group(1)) if dm_m else None,
    }


def load_mkam_catalog(zip_path=None, member=MKAM_CATALOG_MEMBER):
    """Parse the MKAM target catalog straight out of its shipped zip.
    Returns a list of star dicts (see _parse_line), deduplicated by star
    name with HR-numbered rows preferred over the hand-added "---" block
    (which lacks real B-V/duplicity data)."""
    with zipfile.ZipFile(zip_path or MKAM_CATALOG_ZIP) as zf:
        with zf.open(member) as fh:
            lines = io.TextIOWrapper(fh, encoding="ascii").readlines()
    by_name = {}
    for line in lines:
        star = _parse_line(line)
        if star is None:
            continue
        prev = by_name.get(star["name"])
        if prev is None or (prev["hr"] is None and star["hr"] is not None):
            by_name[star["name"]] = star
    return list(by_name.values())


def _jd_utc(when):
    """Naive datetimes are taken as UTC; aware ones are converted."""
    if when.tzinfo is not None:
        when = when.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return 2440587.5 + (when - dt.datetime(1970, 1, 1)).total_seconds() / 86400.0


def star_altaz(ra_deg, dec_deg, when_utc,
               lat_deg=KECK_LAT_DEG, lon_deg=KECK_LON_DEG):
    """(az_from_north_deg, el_deg) of a J2000 position at the summit.
    Scheduling-grade only (see module docstring): GMST hour angle on the
    catalog coordinates, no precession/nutation/refraction."""
    d = _jd_utc(when_utc) - 2451545.0
    t_c = d / 36525.0
    gmst = (280.46061837 + 360.98564736629 * d
            + 3.87933e-4 * t_c * t_c - t_c ** 3 / 3.871e7) % 360.0
    ha = math.radians((gmst + lon_deg - ra_deg) % 360.0)
    phi = math.radians(lat_deg)
    dec = math.radians(dec_deg)
    sin_el = (math.sin(phi) * math.sin(dec)
              + math.cos(phi) * math.cos(dec) * math.cos(ha))
    el = math.degrees(math.asin(max(-1.0, min(1.0, sin_el))))
    az = math.degrees(math.atan2(
        -math.cos(dec) * math.sin(ha),
        math.sin(dec) * math.cos(phi)
        - math.cos(dec) * math.sin(phi) * math.cos(ha)))
    return az % 360.0, el


def _is_single_enough(star, max_sep_arcsec, min_dmag):
    sep, dm = star["sep_arcsec"], star["dmag"]
    if sep is None or dm is None:
        return True                      # no companion noted
    return sep > max_sep_arcsec or dm >= min_dmag


def dimm_star_probabilities(times_utc, catalog=None, min_el_deg=45.0,
                            softness_deg=5.0, max_companion_sep_arcsec=10.0,
                            min_companion_dmag=4.0):
    """Probability each catalog star hosted the monitor over `times_utc`.

    Implements the scheduler model in the module docstring.  Returns a
    list sorted by descending probability of dicts:

      star       the catalog entry (see load_mkam_catalog)
      prob       time-averaged normalized zenith-proximity weight
      pick_frac  fraction of sample times this star is the outright
                 nearest-to-zenith eligible star
      zd_mean    its mean zenith distance (deg) over the times it was
                 eligible (None if never eligible)

    Times with no eligible star contribute nothing.  Empty input -> [].
    """
    if catalog is None:
        catalog = load_mkam_catalog()
    usable = [s for s in catalog if _is_single_enough(
        s, max_companion_sep_arcsec, min_companion_dmag)]
    if not usable or not times_utc:
        return []
    zd_max = 90.0 - min_el_deg
    prob = [0.0] * len(usable)
    picks = [0] * len(usable)
    zd_sum = [0.0] * len(usable)
    zd_n = [0] * len(usable)
    n_used = 0
    for when in times_utc:
        zds = []
        for i, s in enumerate(usable):
            _az, el = star_altaz(s["ra_deg"], s["dec_deg"], when)
            zd = 90.0 - el
            if zd <= zd_max:
                zds.append((zd, i))
                zd_sum[i] += zd
                zd_n[i] += 1
        if not zds:
            continue
        n_used += 1
        zd_best = min(zd for zd, _i in zds)
        picks[min(zds)[1]] += 1
        wts = [(math.exp(-(zd - zd_best) / softness_deg), i) for zd, i in zds]
        norm = sum(w for w, _i in wts)
        for w, i in wts:
            prob[i] += w / norm
    if n_used == 0:
        return []
    out = []
    for i, s in enumerate(usable):
        if zd_n[i] == 0:
            continue
        out.append({"star": s, "prob": prob[i] / n_used,
                    "pick_frac": picks[i] / n_used,
                    "zd_mean": zd_sum[i] / zd_n[i]})
    out.sort(key=lambda r: -r["prob"])
    return out


def top_monitor_orientations(times_utc, ref_time_utc, n=3, **kwargs):
    """The `n` most probable monitor pointings for the FA-geometry plot:
    dimm_star_probabilities over `times_utc`, with each candidate's
    (az, el) evaluated at `ref_time_utc`.  Returns dicts with keys
    name / pretty / vmag / prob / pick_frac / zd_mean / az / el."""
    ranked = dimm_star_probabilities(times_utc, **kwargs)
    out = []
    for r in ranked[:n]:
        s = r["star"]
        az, el = star_altaz(s["ra_deg"], s["dec_deg"], ref_time_utc)
        out.append({"name": s["name"], "pretty": s["pretty"],
                    "vmag": s["vmag"], "prob": r["prob"],
                    "pick_frac": r["pick_frac"], "zd_mean": r["zd_mean"],
                    "az": az, "el": el})
    return out
