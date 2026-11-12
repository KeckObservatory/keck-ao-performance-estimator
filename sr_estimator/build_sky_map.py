#!/usr/bin/env python3
"""Build the all-sky star counts bundled for the sky-coverage tab
(src/keck_ao_estimator/data/sky_allsky_gaia_dr3.npz, read by
keck_ao_estimator.sky_map.load_allsky).

Sky grid: N_FIB points on a Fibonacci sphere (equal area to ~1 %), kept at
dec >= DEC_MIN (ZA <= 60 deg at transit from Mauna Kea): 821 sight lines.
At each, Gaia DR3 counts in a cone, G < G_MAX, histogrammed in G (0.2 mag)
x BP-RP (0.25 mag). Cone radius 0.25 deg at |b| >= 20, 0.10 at 10-20, 0.05
nearer the plane (counts are high there and the queries slow; every cone
still holds tens of stars bright enough to matter).

IR magnitudes: TRICK senses H or K, which Gaia lacks. A sample of Gaia DR3
and 2MASS stars in N_SAMPLE small cones (stratified in |b|) is matched by
position at the 2MASS epoch (Gaia proper motions back to 2000.0), and
G - H and G - K are fitted as cubics in BP - RP with 3-sigma clipping. Per
|b| band the median residual and robust scatter are kept, and so is the
fraction of 2MASS H <= 15.5 stars with no Gaia counterpart (reddened stars
near the plane, mostly unresolved galaxies elsewhere); sky_map adds those
back as stars only TRICK can use.

Archives as found 2026-11-02 from Keck: VizieR / CDS unreachable; the ESA
Gaia archive cancels anonymous queries after 60 s and queues parallel ones;
IRSA (gaia_dr3_source, fp_psc) gives the same counts. So count cones go
one at a time to ESA and two at a time to IRSA, cached per cone; the grid
took 34 minutes.

    python sr_estimator/build_sky_map.py --cache DIR      # query, caching in DIR
    python sr_estimator/build_sky_map.py --cache DIR --no-fetch   # cache only

DIR/.gaia_cones/cone_<i>.npz and DIR/.ir_sample/s<k>.npz are the layouts of
keck_ao_experiments/plgs_sky_coverage_20261102 (gaia_cones.py, ir_mags.py),
from which the bundled file was built.
"""
import argparse
import os
import queue
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from keck_ao_estimator.sky_map import (  # noqa: E402
    ALLSKY_FILE, B_EDGES, C_CLIP, MISS_H_MAX)

N_FIB = 1000
DEC_MIN = -40.0
G_MAX = 19.8
G_STEP, C_STEP = 0.2, 0.25
SEED = 20261102
N_SAMPLE = 36
MATCH_ARCSEC = 1.0
EPOCH_GAIA, EPOCH_2MASS = 2016.0, 2000.0
ESA = "https://gea.esac.esa.int/tap-server/tap"
IRSA = "https://irsa.ipac.caltech.edu/TAP"
# service -> (TAP url, table, CONTAINS form, GROUP BY form, workers)
SERVICES = {"esa": (ESA, "gaiadr3.gaia_source", "1=CONTAINS({p}, {c})",
                    "alias", 1),
            "irsa": (IRSA, "gaia_dr3_source", "CONTAINS({p}, {c})=1",
                     "expr", 2)}


def fibonacci_grid(n=N_FIB):
    i = np.arange(n)
    dec = np.degrees(np.arcsin(1.0 - (2.0 * i + 1.0) / n))
    ra = np.degrees(i * np.pi * (3.0 - np.sqrt(5.0))) % 360.0
    return ra, dec


def grid():
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ra, dec = fibonacci_grid()
    idx = np.where(dec >= DEC_MIN)[0]
    gal = SkyCoord(ra=ra[idx] * u.deg, dec=dec[idx] * u.deg).galactic
    return idx, ra[idx], dec[idx], gal.l.deg, gal.b.deg


def cone_radius(b):
    b = abs(b)
    return 0.05 if b < 10 else (0.10 if b < 20 else 0.25)


# ---- counts per cone ---------------------------------------------------------
def adql(service, ra, dec, r):
    _, table, contains, group, _ = SERVICES[service]
    gb = f"FLOOR(phot_g_mean_mag*{1 / G_STEP:g})"
    cb = f"FLOOR(bp_rp*{1 / C_STEP:g})"
    where = contains.format(p="POINT('ICRS', ra, dec)",
                            c=f"CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {r})")
    by = "gb, cb" if group == "alias" else f"{gb}, {cb}"
    return (f"SELECT {gb} AS gb, {cb} AS cb, COUNT(*) AS n FROM {table} "
            f"WHERE {where} AND phot_g_mean_mag < {G_MAX} GROUP BY {by}")


def _worker(service, todo, cache, log):
    import pyvo
    tap = pyvo.dal.TAPService(SERVICES[service][0])
    while True:
        try:
            i, ra, dec, r, tries = todo.get_nowait()
        except queue.Empty:
            return
        try:
            t = tap.search(adql(service, ra, dec, r)).to_table()
            cb = np.ma.filled(np.ma.asarray(t["cb"], dtype=float), np.nan)
            np.savez(os.path.join(cache, f"cone_{i:04d}.npz"), ra=ra, dec=dec,
                     r=r, gb=np.asarray(t["gb"], float), cb=cb,
                     n=np.asarray(t["n"], np.int64), service=service)
            log("done")
        except Exception as e:                    # network / server
            if tries < 6:
                todo.put((i, ra, dec, r, tries + 1))
            log(f"{service} retry cone {i}: {e!s:.70}")
            time.sleep(15 * (tries + 1))
            tap = pyvo.dal.TAPService(SERVICES[service][0])


def fetch_counts(cache):
    os.makedirs(cache, exist_ok=True)
    idx, ra, dec, _, b = grid()
    todo = queue.Queue()
    for k in np.random.default_rng(SEED).permutation(len(idx)):
        i = int(idx[k])
        if not os.path.exists(os.path.join(cache, f"cone_{i:04d}.npz")):
            todo.put((i, float(ra[k]), float(dec[k]), cone_radius(b[k]), 0))
    total, lock, done = todo.qsize(), threading.Lock(), [0]

    def log(msg):
        with lock:
            if msg != "done":
                print("  " + msg, flush=True)
                return
            done[0] += 1
            if done[0] % 25 == 0 or done[0] == total:
                print(f"{done[0]}/{total} cones", flush=True)
    threads = [threading.Thread(target=_worker, args=(s, todo, cache, log))
               for s, spec in SERVICES.items() for _ in range(spec[4])]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


# ---- Gaia -> H, K relation ---------------------------------------------------
def sample_points():
    idx, ra, dec, _, b = grid()
    rng = np.random.default_rng(SEED)
    pick = []
    for lo, hi in zip(B_EDGES[:-1], B_EDGES[1:]):
        k = np.where((np.abs(b) >= lo) & (np.abs(b) < hi))[0]
        pick += list(rng.choice(k, N_SAMPLE // (len(B_EDGES) - 1),
                                replace=False))
    pick = np.array(sorted(pick))
    return ra[pick], dec[pick], b[pick]


def _sample_radius(b):
    return 0.03 if abs(b) < 10 else (0.06 if abs(b) < 20 else 0.12)


def fetch_sample(k, ra, dec, r, cache):
    f = os.path.join(cache, f"s{k:02d}.npz")
    if os.path.exists(f):
        return dict(np.load(f))
    import pyvo
    tap = pyvo.dal.TAPService(IRSA)
    circ = f"CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {r})"
    for tries in range(4):
        try:
            g = tap.search(
                "SELECT ra, dec, phot_g_mean_mag AS g, bp_rp, pmra, pmdec "
                "FROM gaia_dr3_source WHERE CONTAINS(POINT('ICRS', ra, dec), "
                f"{circ})=1 AND phot_g_mean_mag < {G_MAX}").to_table()
            t = tap.search(
                "SELECT ra, dec, h_m, k_m FROM fp_psc WHERE "
                f"CONTAINS(POINT('ICRS', ra, dec), {circ})=1 "
                "AND h_m < 17").to_table()
            break
        except Exception as e:                    # network / server
            print(f"  retry sample {k}: {e!s:.70}", flush=True)
            time.sleep(10 * (tries + 1))
    else:
        raise RuntimeError(f"sample cone {k} failed")
    fl = lambda c: np.ma.filled(np.ma.asarray(c, dtype=float), np.nan)
    d = dict(g_ra=fl(g["ra"]), g_dec=fl(g["dec"]), g=fl(g["g"]),
             c=fl(g["bp_rp"]), pmra=fl(g["pmra"]), pmdec=fl(g["pmdec"]),
             t_ra=fl(t["ra"]), t_dec=fl(t["dec"]), h=fl(t["h_m"]),
             k=fl(t["k_m"]))
    os.makedirs(cache, exist_ok=True)
    np.savez(f, **d)
    return d


def match(d):
    """Index of the Gaia match for each 2MASS star (-1 if none)."""
    from scipy.spatial import cKDTree
    dt = EPOCH_2MASS - EPOCH_GAIA
    dec0 = np.radians(np.nanmean(d["g_dec"]))
    gx = (d["g_ra"] * np.cos(dec0) * 3600
          + np.nan_to_num(d["pmra"]) * dt / 1000)
    gy = d["g_dec"] * 3600 + np.nan_to_num(d["pmdec"]) * dt / 1000
    tx, ty = d["t_ra"] * np.cos(dec0) * 3600, d["t_dec"] * 3600
    dist, j = cKDTree(np.c_[gx, gy]).query(np.c_[tx, ty])
    return np.where(dist <= MATCH_ARCSEC, j, -1)


def fit_colour(c, gx):
    """Cubic G - X in clipped BP - RP, iterative 3-sigma clipping."""
    ok = np.isfinite(c) & np.isfinite(gx)
    c, gx = np.clip(c[ok], *C_CLIP), gx[ok]
    use = np.ones(len(c), bool)
    for _ in range(10):
        p = np.polynomial.polynomial.polyfit(c[use], gx[use], 3)
        r = gx - np.polynomial.polynomial.polyval(c, p)
        s = 1.4826 * np.median(np.abs(r[use] - np.median(r[use])))
        new = np.abs(r) < 3 * s
        if (new == use).all():
            break
        use = new
    return p


def relation(cache, fetch=True):
    ra, dec, b = sample_points()
    rows = dict(c=[], gh=[], gk=[], ab=[])
    miss = np.zeros((len(B_EDGES) - 1, 2))
    for k in range(len(ra)):
        if not fetch and not os.path.exists(
                os.path.join(cache, f"s{k:02d}.npz")):
            raise SystemExit(f"sample cone {k} not cached in {cache}")
        d = fetch_sample(k, float(ra[k]), float(dec[k]),
                         _sample_radius(b[k]), cache)
        j = match(d)
        m = j >= 0
        rows["c"].append(d["c"][j[m]])
        rows["gh"].append(d["g"][j[m]] - d["h"][m])
        rows["gk"].append(d["g"][j[m]] - d["k"][m])
        rows["ab"].append(np.full(m.sum(), abs(b[k])))
        bi = np.searchsorted(B_EDGES, abs(b[k]), side="right") - 1
        bright = d["h"] <= MISS_H_MAX
        miss[bi] += (bright.sum(), (bright & ~m).sum())
    c, gh, gk, ab = (np.concatenate(rows[x]) for x in ("c", "gh", "gk", "ab"))
    out = dict(miss_frac=miss[:, 1] / miss[:, 0], n_matched=len(c))
    for key, gx in (("h", gh), ("k", gk)):
        p = fit_colour(c, gx)
        off, sig = [], []
        for lo, hi in zip(B_EDGES[:-1], B_EDGES[1:]):
            s = (ab >= lo) & (ab < hi) & np.isfinite(c) & np.isfinite(gx)
            r = gx[s] - np.polynomial.polynomial.polyval(
                np.clip(c[s], *C_CLIP), p)
            off.append(np.median(r))
            sig.append(1.4826 * np.median(np.abs(r - np.median(r))))
        out[f"g{key}_coeffs"] = p
        out[f"{key}_offset"], out[f"{key}_sigma"] = np.array(off), np.array(sig)
    return out


# ---- assemble ------------------------------------------------------------------
def assemble(count_cache, rel):
    idx, ra, dec, l, b = grid()
    keep, rows = [], dict(cone=[], gb=[], cb=[], n=[])
    radius = []
    for k, i in enumerate(idx):
        f = os.path.join(count_cache, f"cone_{int(i):04d}.npz")
        if not os.path.exists(f):
            continue
        with np.load(f) as z:
            j = len(keep)
            keep.append(k)
            radius.append(float(z["r"]))
            rows["cone"].append(np.full(len(z["n"]), j))
            rows["gb"].append(z["gb"])
            rows["cb"].append(np.where(np.isfinite(z["cb"]), z["cb"], -999))
            rows["n"].append(z["n"])
    if len(keep) < len(idx):
        raise SystemExit(f"only {len(keep)} of {len(idx)} cones cached")
    keep = np.array(keep)
    np.savez_compressed(
        ALLSKY_FILE,
        ra=ra.astype("f4"), dec=dec.astype("f4"), l=l.astype("f4"),
        b=b.astype("f4"), radius=np.array(radius, "f4"),
        cone=np.concatenate(rows["cone"]).astype("u2"),
        gb=np.concatenate(rows["gb"]).astype("i2"),
        cb=np.concatenate(rows["cb"]).astype("i2"),
        n=np.concatenate(rows["n"]).astype("i4"),
        g_step=G_STEP, c_step=C_STEP, g_max=G_MAX, dec_min=DEC_MIN,
        n_fib=N_FIB, **{k: np.asarray(v) for k, v in rel.items()})
    print(f"wrote {ALLSKY_FILE}: {len(keep)} sight lines, "
          f"{os.path.getsize(ALLSKY_FILE) / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", required=True,
                    help="directory holding .gaia_cones/ and .ir_sample/")
    ap.add_argument("--no-fetch", action="store_true",
                    help="use the cache only; never query")
    a = ap.parse_args()
    counts = os.path.join(a.cache, ".gaia_cones")
    sample = os.path.join(a.cache, ".ir_sample")
    if not a.no_fetch:
        fetch_counts(counts)
    rel = relation(sample, fetch=not a.no_fetch)
    print({k: np.round(v, 4) for k, v in rel.items()})
    assemble(counts, rel)


if __name__ == "__main__":
    main()
