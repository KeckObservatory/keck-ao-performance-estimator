#!/usr/bin/env python3
"""Build the star fields bundled for the sky-coverage tab
(src/keck_ao_estimator/data/sky_field_<tag>.npz, read by
keck_ao_estimator.sky_coverage.load_sky_field).

One real field per sight line: every Gaia DR2 star with G < 20.5 (VizieR
I/345/gaia2) and every 2MASS point source (VizieR II/246/out) inside a
0.5 deg cone, stored as tangent-plane offsets from the cone centre in
arcsec plus the magnitudes, float32. G stands in for the R magnitude the
STRAP quadcell and the NGS WFS sense in; H and K feed TRICK. The synthetic
field (no sight line) is written from sky_coverage.synthetic_sky_field().

    python sr_estimator/build_sky_fields.py                # query VizieR
    python sr_estimator/build_sky_fields.py --from-cache DIR

--from-cache converts <tag>.npz files already pulled by
keck_ao_experiments/skycoverage/measured_sky_coverage.py (same queries;
keys ra_2m/de_2m/H/K/ra_g/de_g/G/ra0/de0) instead of querying again. The
four bundled fields were built that way from the 2026-07-26 pull.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from keck_ao_estimator.sky_coverage import (  # noqa: E402
    SKY_FIELD_RADIUS_ARCSEC, SKY_FIELDS, _DATA_DIR, synthetic_sky_field)


def query(l_deg, b_deg):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier
    c = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame="galactic").icrs
    rad = SKY_FIELD_RADIUS_ARCSEC / 3600.0 * u.deg
    t = Vizier(columns=["RAJ2000", "DEJ2000", "Hmag", "Kmag"],
               catalog="II/246/out", row_limit=-1).query_region(
                   c, radius=rad)[0]
    g = Vizier(columns=["RA_ICRS", "DE_ICRS", "Gmag"],
               column_filters={"Gmag": "<20.5"},
               catalog="I/345/gaia2", row_limit=-1).query_region(
                   c, radius=rad)[0]
    return dict(ra_2m=np.asarray(t["RAJ2000"], float),
                de_2m=np.asarray(t["DEJ2000"], float),
                H=np.asarray(t["Hmag"], float), K=np.asarray(t["Kmag"], float),
                ra_g=np.asarray(g["RA_ICRS"], float),
                de_g=np.asarray(g["DE_ICRS"], float),
                G=np.asarray(g["Gmag"], float),
                ra0=np.array([c.ra.deg]), de0=np.array([c.dec.deg]))


def compact(cat):
    ra0, de0 = float(cat["ra0"][0]), float(cat["de0"][0])
    cosd = np.cos(np.radians(de0))
    out = {}
    for key, ra, de in (("g", "ra_g", "de_g"), ("ir", "ra_2m", "de_2m")):
        out[f"{key}_x"] = ((cat[ra] - ra0) * 3600.0 * cosd).astype("f4")
        out[f"{key}_y"] = ((cat[de] - de0) * 3600.0).astype("f4")
    for band in ("G", "H", "K"):
        out[band] = np.asarray(cat[band], "f4")
    out["radec0"] = np.array([ra0, de0])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--from-cache", metavar="DIR")
    a = ap.parse_args()
    for tag, (l_deg, b_deg, _label) in SKY_FIELDS.items():
        if l_deg is None:
            cat = out = synthetic_sky_field()
        elif a.from_cache:
            z = np.load(os.path.join(a.from_cache, f"{tag}.npz"))
            cat = {k: z[k] for k in z.files}
            out = compact(cat)
        else:
            cat = query(l_deg, b_deg)
            out = compact(cat)
        dest = os.path.join(_DATA_DIR, f"sky_field_{tag}.npz")
        np.savez_compressed(dest, **out)
        print(f"{dest}: {len(cat['G'])} optical, {len(cat['H'])} near-IR, "
              f"{os.path.getsize(dest) / 1024:.0f} kB")


if __name__ == "__main__":
    main()
