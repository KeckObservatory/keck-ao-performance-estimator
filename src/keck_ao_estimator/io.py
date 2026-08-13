"""Loading and fetching the night's atmospheric data: DIMM/MASS/masspro file
parsing, and the MKWC (Mauna Kea Weather Center) archive fetch."""
import os
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np

from .constants import MKWC_DIMM_URL, MKWC_MASS_URL, MKWC_MASSPRO_URL


def parse_dt(parts):
    """First six whitespace tokens -> datetime (YYYY MM DD hh mm ss)."""
    Y, M, D, h, mi, s = (int(x) for x in parts[:6])
    return datetime(Y, M, D, h, mi, s)


def parse_secs(parts):
    """A monotonic second-of-month counter, fine for nearest-time matching
    within a single night (avoids any date-boundary subtraction headaches)."""
    Y, M, D, h, mi, s = (int(x) for x in parts[:6])
    return ((D * 24 + h) * 60 + mi) * 60 + s


def load_seeing_series(path):
    """Load a DIMM- or MASS-style file: returns (datetimes, seconds, seeing).
    Returns empty arrays if the file is missing or unreadable (caller decides
    whether that is fatal)."""
    dts, secs, vals = [], [], []
    if path is None:                    # e.g. MKWC fetch found no file
        return np.array([]), np.array([]), np.array([])
    try:
        fh = open(path)
    except (OSError, IOError, TypeError) as e:
        print(f"  WARNING: could not open '{path}' ({e}); treating as no data.")
        return np.array([]), np.array([]), np.array([])
    with fh:
        for line in fh:
            p = line.split()
            if len(p) < 7:                 # skip blank / header / short lines
                continue
            try:
                dts.append(parse_dt(p))
                secs.append(parse_secs(p))
                vals.append(float(p[6]))
            except ValueError:
                continue                   # skip un-parseable lines
    return np.array(dts), np.array(secs), np.array(vals)


def load_mass_profile(path):
    """Load the MASS profile file.
    Returns a list of (datetime, seconds, free_atm_seeing, cn2_bins) where
    cn2_bins is the list of 6 layer-integrated Cn2 values J_i [m^1/3] at the
    standard MASS heights (0.5, 1, 2, 4, 8, 16 km). The free-atm seeing drives
    the LGS/LTAO budget; the Cn2 bins drive the theta0 / d0 calculation AND
    the LTAO tomography layer-mismatch penalty (the bin altitudes coincide
    with the aloft part of the K1 reconstructor's static prior)."""
    out = []
    if path is None:                    # e.g. MKWC fetch found no file
        return out
    try:
        fh = open(path)
    except (OSError, IOError, TypeError) as e:
        print(f"  WARNING: could not open '{path}' ({e}); treating as no data.")
        return out
    with fh:
        for line in fh:
            p = line.split()
            if len(p) < 13:                # need 6 time + 6 Cn2 + 1 seeing
                continue
            try:
                dt = parse_dt(p)
                sc = parse_secs(p)
                cn2_bins = [float(x) for x in p[6:12]]   # 6 layer J_i [m^1/3]
                free_atm_see = float(p[12])
                out.append((dt, sc, free_atm_see, cn2_bins))
            except ValueError:
                continue
    #  MKWC masspro files sometimes list the same timestamp twice (usually a
    #  byte-identical repeat, occasionally a re-reported/corrected sample).
    #  Keep only the LAST occurrence of each timestamp -- a later line in the
    #  file supersedes an earlier one -- so each real MASS sample yields
    #  exactly one profile, and downstream one theta0/LGS point.
    n_raw = len(out)
    by_sec = {}
    for rec in out:                      # later records overwrite earlier ones
        by_sec[rec[1]] = rec
    out = sorted(by_sec.values(), key=lambda r: r[1])
    if len(out) != n_raw:
        print(f"  note: {n_raw - len(out)} duplicate-timestamp masspro "
              f"sample(s) dropped (kept last occurrence of each)")
    return out


def fetch_mkwc_files(ymd, cache_dir, refetch=False, trust_cache=False):
    """Download the DIMM/MASS/MASSPRO files for one night from the MKWC archive.

    Parameters
    ----------
    ymd        : 'YYYYMMDD' file date stamp (the morning/UT date of the night).
    cache_dir  : local directory to store the files in.
    refetch    : if True, always re-download (ignore any cache).
    trust_cache: if True, always use the cache when present (skip the staleness
                 check). Mutually informative with refetch; refetch wins.

    Cache-staleness rule
    --------------------
    MKWC keeps APPENDING to a night's files until that night is over, so a cache
    captured mid-night is only a partial stub. We therefore trust a cached file
    only once the night's UT date has fully passed (i.e. now is at/after the END
    of that UT day). Before that, we auto-refetch so a re-run after the night
    picks up the complete data instead of the stale stub. --refetch forces a
    re-download regardless; --no-refetch (trust_cache) forces using the cache.

    Returns (dimm_path, mass_path, masspro_path). Raises SystemExit on failure.
    """
    if not (len(ymd) == 8 and ymd.isdigit()):
        raise SystemExit(f"ERROR: --fetch-date '{ymd}' must be YYYYMMDD digits.")

    # Has the night's UT date fully passed? The file stamp ymd is the morning/UT
    # date; that UT day ends at 24:00 UT on ymd. Compare against 'now' in UTC.
    file_day = datetime.strptime(ymd, "%Y%m%d")
    ut_day_end = file_day + timedelta(days=1)          # 00:00 UT the next day
    # current UTC time (timezone-naive, to compare with the naive file_day)
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    night_complete = now_utc >= ut_day_end

    os.makedirs(cache_dir, exist_ok=True)
    targets = {
        "dimm":    (MKWC_DIMM_URL.format(ymd=ymd),
                    os.path.join(cache_dir, f"{ymd}.dimm.dat")),
        "mass":    (MKWC_MASS_URL.format(ymd=ymd),
                    os.path.join(cache_dir, f"{ymd}.mass.dat")),
        "masspro": (MKWC_MASSPRO_URL.format(ymd=ymd),
                    os.path.join(cache_dir, f"{ymd}.masspro.dat")),
    }

    # Decide whether a cached copy may be used this run.
    #   --refetch            -> never use cache
    #   --no-refetch         -> always use cache if present
    #   default              -> use cache only if the night is complete
    use_cache = (not refetch) and (trust_cache or night_complete)
    if (not refetch) and (not trust_cache) and (not night_complete):
        print(f"  Night {file_day.date()} (UT) is not over yet — auto-refetching "
              f"so partial cached data isn't reused. (use --no-refetch to override)")

    paths = {}
    for kind, (url, local) in targets.items():
        if os.path.exists(local) and use_cache:
            print(f"  [cache] {kind:8s} {local}")
            paths[kind] = local
            continue
        try:
            print(f"  [fetch] {kind:8s} {url}")
            # 30 s timeout so a hung server doesn't stall the run forever
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            # If the download fails but we have a cached copy, fall back to it
            # rather than dying -- a stale file beats no file, with a warning.
            if os.path.exists(local):
                print(f"  WARNING: download failed for {kind} ({e}); "
                      f"falling back to cached {local}")
                paths[kind] = local
                continue
            if kind == "dimm":
                # Without DIMM there is nothing to compute -- this one is fatal.
                raise SystemExit(
                    f"ERROR: could not download dimm file for {ymd}.\n"
                    f"       URL: {url}\n"
                    f"       Reason: {e}\n"
                    f"       (Check the date exists in the archive and that you "
                    f"have network access.)") from e
            # MASS / masspro missing is a normal condition, especially early on
            # a live night (MKWC posts DIMM first) or when the MASS was down:
            # warn and continue -- the estimator degrades gracefully to
            # DIMM-only (NGS) exactly as it does with missing local files.
            print(f"  WARNING: {kind} not available for {ymd} ({e}) — "
                  f"continuing without it (NGS-only if both MASS files are "
                  f"missing).")
            paths[kind] = None
            continue
        if not data.strip():
            if kind == "dimm":
                raise SystemExit(
                    f"ERROR: dimm file for {ymd} downloaded empty.\n"
                    f"       URL: {url}\n"
                    f"       (The archive may not have data for that date.)")
            print(f"  WARNING: {kind} file for {ymd} downloaded empty — "
                  f"continuing without it.")
            paths[kind] = None
            continue
        with open(local, "wb") as fh:
            fh.write(data)
        paths[kind] = local

    return paths["dimm"], paths["mass"], paths["masspro"]
