"""Target name resolution via SIMBAD: turn a target name (e.g. "HD 141569")
into an ICRS RA/Dec, proper motion, and available magnitudes, for the Target
tab's Resolve button and the starlist picker's guide-star-magnitude fallback.

Qt-free and heavy-import-lazy (astroquery imported inside the function that
needs it), like catalogs.py/imaging.py. Only resolve_target_name() touches
the network -- not exercised by the offline suite.
"""

# SIMBAD "allfluxes" columns worth carrying: every label photometry.py's
# _BAND_UM understands (so estimate_sensing_mag/pick_mag can use them
# directly). SIMBAD's lowercase g/r/i/u/z are SDSS-like -- photometry knows
# g/r/i (PanSTARRS ~ SDSS); u/z have no _BAND_UM entry worth adding.
_FLUX_BANDS = ("U", "B", "V", "G", "R", "I", "J", "H", "K", "g", "r", "i")


def resolve_target_name(name):
    """Look up `name` in SIMBAD. Returns a dict:
      name    : SIMBAD's resolved canonical identifier (e.g. "HD 141569" ->
                may differ from what was typed, e.g. a common name resolving
                to a catalogue identifier).
      ra_deg  : ICRS RA, decimal degrees.
      dec_deg : ICRS Dec, decimal degrees.
      pmra    : proper motion in RA*cos(Dec) -- the standard SIMBAD/Gaia
                PMRA convention -- mas/yr, or None if SIMBAD has no measured
                proper motion for this object (common for e.g. Sgr A* itself,
                a radio source with no stellar proper motion).
      pmdec   : proper motion in Dec, mas/yr, or None.
      mags    : {band: value_or_None} in photometry.py's band labels
                (SIMBAD "allfluxes"; a column SIMBAD doesn't return at all
                is simply absent-as-None too). Feed straight into
                estimate_sensing_mag()/pick_mag().
    Raises ValueError if `name` does not resolve to anything in SIMBAD."""
    import numpy.ma as ma
    from astroquery.simbad import Simbad
    sim = Simbad()
    sim.add_votable_fields("pmra", "pmdec", "allfluxes")
    table = sim.query_object(name.strip())
    if table is None or len(table) == 0:
        raise ValueError(f"“{name}” did not resolve in SIMBAD")
    row = table[0]

    def _val(col):
        # tolerant of a column being absent entirely (synthetic test tables,
        # or a SIMBAD schema change), not just masked
        if col not in table.colnames:
            return None
        v = row[col]
        return None if ma.is_masked(v) else float(v)

    return dict(
        name=str(row["main_id"]),
        ra_deg=float(row["ra"]), dec_deg=float(row["dec"]),
        pmra=_val("pmra"), pmdec=_val("pmdec"),
        mags={b: _val(b) for b in _FLUX_BANDS},
    )
