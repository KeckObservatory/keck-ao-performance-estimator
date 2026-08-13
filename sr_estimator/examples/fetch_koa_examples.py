#!/usr/bin/env python3
"""Fetch the example FITS frames from the Keck Observatory Archive (KOA).

Both frames are PUBLIC KOA data (past their proprietary periods; anonymous
access, no credentials), so the repo never redistributes anything an
observer could object to. The .gitignore rules keep any downloaded frame
out of git; the one committed frame (N2.20210821_42637.fits) is itself a
public KOA product, re-downloadable here for provenance.

Frames (pinned by KOAID so the fetch is reproducible):
  * NIRC2  N2.20210821.42637  -- M15 globular-cluster field, Kp, narrow
    camera, 2021-08-21 (18-month proprietary period, public since 2023;
    the committed field-map/Strehl demo frame used by
    regress/gui_phase22.py)
  * OSIRIS OI.20180527.31632  -- imager, Kp, HIP 61960 / SR 12, 2018-05-27
    (36-month proprietary period, public since 2021; a field-map /
    osiris.py demo input)

Downloads land in <outdir>/lev0/ (pykoa's layout).

Requires pykoa (pip install pykoa):  python fetch_koa_examples.py [outdir]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

FRAMES = [
    ("koa_nirc2", "N2.20210821.42637.fits"),
    ("koa_osiris", "OI.20180527.31632.fits"),
]


def fetch(outdir):
    from pykoa.koa import Koa   # import here: pykoa is not a dependency

    os.makedirs(outdir, exist_ok=True)
    for table, koaid in FRAMES:
        tbl = os.path.join(outdir, f"{koaid}.tbl")
        Koa.query_adql(
            f"select koaid, instrume, filehand, propint from {table} "
            f"where koaid = '{koaid}'",
            tbl, overwrite=True, format="ipac")
        Koa.download(tbl, "ipac", outdir)
        print(f"fetched {koaid} -> {outdir}/lev0/")


if __name__ == "__main__":
    fetch(sys.argv[1] if len(sys.argv) > 1 else HERE)
