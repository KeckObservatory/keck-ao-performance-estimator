"""Guide-star catalogue lookups for the field map: a small registry of the
Vizier catalogues Keck observers use to pick tip-tilt / natural guide stars
(GSC 2.4, 2MASS, UCAC4, PanSTARRS DR2, Gaia DR2), a pure parser that turns a
Vizier result table into plain star dicts, and a converter from star RA/Dec
to field-map (x, y) offsets. Band-to-band magnitude estimation (picking or
estimating a star's magnitude in a wanted band from whatever photometry it
carries) lives in photometry.py instead -- pure math with no dependency on
Vizier, reusable wherever a magnitude conversion is needed.

Only query_guide_stars() touches the network -- a thin astroquery.Vizier
wrapper mirrored by CatalogFetchWorker, and (like imaging._fetch_sky_jpeg) not
exercised by the offline suite. The registry, the table parser and the
geometry converter are all pure and are tested headless.

Deliberately Qt-free and heavy-import-lazy (astroquery/astropy imported inside
the functions that need them), like imaging.py.
"""
import numpy as np

# Each catalogue: its Vizier table id, the RA/Dec columns (all these tables are
# ICRS/J2000 in degrees), an optional identifier column, and the magnitude
# columns keyed by band label. Bands are matched to the tip-tilt sensor's
# wavelength by photometry.pick_mag(), so the registry just lists what each
# catalogue carries. Ordered most-commonly-used first (matches the observers'
# dropdown).
CATALOGS = {
    "GSC 2.4": dict(
        vizier_id="I/305/out", ra_col="RAJ2000", dec_col="DEJ2000",
        id_col="GSC2.3", mag_cols={"F": "Fmag", "j": "jmag", "V": "Vmag",
                                   "N": "Nmag"}),
    "2MASS": dict(
        vizier_id="II/246/out", ra_col="RAJ2000", dec_col="DEJ2000",
        id_col="_2MASS", mag_cols={"J": "Jmag", "H": "Hmag", "K": "Kmag"}),
    "UCAC4": dict(
        vizier_id="I/322A/out", ra_col="RAJ2000", dec_col="DEJ2000",
        id_col="UCAC4", mag_cols={"V": "Vmag", "R": "f.mag", "B": "Bmag",
                                  "J": "Jmag", "H": "Hmag", "K": "Kmag"}),
    "PanSTARRS DR2": dict(
        vizier_id="II/349/ps1", ra_col="RAJ2000", dec_col="DEJ2000",
        id_col="objID", mag_cols={"g": "gmag", "r": "rmag", "i": "imag",
                                  "z": "zmag", "y": "ymag"}),
    "Gaia DR2": dict(
        vizier_id="I/345/gaia2", ra_col="RA_ICRS", dec_col="DE_ICRS",
        id_col="Source", mag_cols={"G": "Gmag", "BP": "BPmag", "RP": "RPmag"}),
}


def _cell(row, col):
    """A numeric cell as float, or None if absent / masked / NaN."""
    if not col or col not in row.colnames:
        return None
    v = row[col]
    if v is np.ma.masked:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else v


def parse_catalog_table(table, spec):
    """Vizier result Table -> list of star dicts
    {id, ra(deg), dec(deg), mags:{band: value_or_None}}. Rows without a usable
    RA/Dec are dropped. Pure (no network); tested headless with a synthetic
    Table."""
    stars = []
    for i, row in enumerate(table):
        ra = _cell(row, spec["ra_col"])
        dec = _cell(row, spec["dec_col"])
        if ra is None or dec is None:
            continue
        mags = {band: _cell(row, col) for band, col in spec["mag_cols"].items()}
        idc = spec.get("id_col")
        if idc and idc in row.colnames and row[idc] is not np.ma.masked:
            sid = str(row[idc])
        else:
            sid = f"#{i + 1}"
        stars.append({"id": sid, "ra": ra, "dec": dec, "mags": mags})
    return stars


def stars_field_xy(stars, center_ra_deg, center_dec_deg):
    """Return the stars with plot-frame field offsets added: x = West+ (−East),
    y = North+, arcsec from the field centre. Pure geometry (astropy), tested
    headless."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    c = SkyCoord(center_ra_deg * u.deg, center_dec_deg * u.deg)
    if not stars:
        return []
    sc = SkyCoord([s["ra"] for s in stars] * u.deg,
                  [s["dec"] for s in stars] * u.deg)
    dlon, dlat = c.spherical_offsets_to(sc)
    out = []
    for s, dl, db in zip(stars, dlon, dlat):
        s2 = dict(s)
        s2["x"] = -float(dl.arcsec)     # plot x = West+ = −East
        s2["y"] = float(db.arcsec)
        out.append(s2)
    return out


def query_guide_stars(catalog_name, ra_deg, dec_deg, radius_arcsec,
                      row_limit=500):
    """Query a guide-star catalogue around (ra, dec) within radius_arcsec, as a
    list of star dicts (see parse_catalog_table). THE ONLY networked function
    here -- run it off the GUI thread (CatalogFetchWorker). Raises on
    query/parse error (caller reports it)."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier
    spec = CATALOGS[catalog_name]
    viz = Vizier(columns=["**"], row_limit=row_limit)
    center = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
    res = viz.query_region(center, radius=radius_arcsec * u.arcsec,
                           catalog=spec["vizier_id"])
    if not res:
        return []
    return parse_catalog_table(res[0], spec)
