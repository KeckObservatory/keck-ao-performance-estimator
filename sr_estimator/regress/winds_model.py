#!/usr/bin/env python3
"""GFS winds module (winds.py): interpolation/collapse math against hand
values, tau0 formula, cache behavior, and -- the part Eduardo explicitly
required -- the failure modes: a dead or slow service must raise WindsError
promptly (single attempt, hard timeout, no retry loop), never hang, and
never mutate anything. Fully offline: urllib is monkeypatched throughout."""
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import keck_ao_estimator as engine
from keck_ao_estimator import winds


def _synth_hourly(levels):
    """One-hour 'hourly' dict from [(hPa, height_m, speed, dir_from), ...],
    replicated across the 04-16 UT night window."""
    hours = [f"2026-07-22T{h:02d}:00" for h in range(24)]
    out = {"time": hours}
    for p, h, s, d in levels:
        out[f"wind_speed_{p}hPa"] = [s] * 24
        out[f"wind_direction_{p}hPa"] = [d] * 24
        out[f"geopotential_height_{p}hPa"] = [h] * 24
    return out


# the archived 07Z GFS profile from the 2026-07-21 FA-event analysis
# (gfs_winds_20260722_06-08Z.csv) -- real, sonde-validated numbers
_EVENT_LADDER = [
    (350, 8612.59, 6.84, 227), (300, 9730.65, 8.33, 233),
    (250, 11005.72, 10.69, 274), (200, 12493.02, 15.3, 259),
    (150, 14295.52, 11.18, 260), (100, 16662.5, 9.1, 278),
    (70, 18775.96, 10.69, 86), (50, 20834.48, 12.62, 102),
    (40, 22222.22, 14.64, 78), (30, 24037.39, 22.0, 89),
    # synthetic low levels to cover the 0.5-2 km bins + ground
    (600, 4420.0, 4.0, 90), (500, 5900.0, 5.0, 100),
    (400, 7600.0, 6.0, 110),
]


def interpolation_and_collapse():
    tmp = tempfile.mkdtemp(prefix="winds_test_")
    calls = []

    class _Resp:
        def __init__(self, payload): self._p = payload
        def read(self): return json.dumps(self._p).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(url, timeout=None):
        calls.append(timeout)
        return _Resp({"hourly": _synth_hourly(_EVENT_LADDER)})

    orig = urllib.request.urlopen
    urllib.request.urlopen = fake_urlopen
    try:
        w = engine.night_winds("20260722", tmp, refetch=True)
    finally:
        urllib.request.urlopen = orig
        shutil.rmtree(tmp, ignore_errors=True)

    assert calls == [10], f"exactly ONE request with the 10 s timeout: {calls}"
    # hand-checks against the archived event profile:
    per_bin = dict(w["per_bin"])
    # 8 km bin = 12207 m ASL, between 250 hPa (11006 m, 10.69) and
    # 200 hPa (12493 m, 15.30) -- u/v interp lands between, nearer 200
    assert 10.7 < per_bin[8.0] < 15.3, per_bin[8.0]
    # 16 km bin = 20207 m ASL, between 70 hPa (18776 m, 10.69 from 86) and
    # 50 hPa (20834 m, 12.62 from 102) -- the stratospheric easterlies the
    # FA-event timing implicated
    assert 10.5 < per_bin[16.0] < 12.7, per_bin[16.0]
    # v_free: 5/3-moment weighting of the six bins with MK_FA_PROFILE_FRAC
    from keck_ao_estimator.fieldmap import MK_FA_PROFILE_FRAC
    v = np.array([per_bin[k] for k in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)])
    want = float((MK_FA_PROFILE_FRAC @ v ** (5 / 3)) ** (3 / 5))
    assert abs(w["v_free"] - want) < 0.06, (w["v_free"], want)
    # custom weights shift it: all weight on the 16 km bin -> that bin's v
    tmp2 = tempfile.mkdtemp(prefix="winds_test2_")
    urllib.request.urlopen = fake_urlopen
    try:
        w16 = engine.night_winds("20260722", tmp2, refetch=True,
                                 fa_weights=[0, 0, 0, 0, 0, 1.0])
    finally:
        urllib.request.urlopen = orig
        shutil.rmtree(tmp2, ignore_errors=True)
    assert abs(w16["v_free"] - per_bin[16.0]) < 0.06
    # ground: clamped to the lowest ladder level (600 hPa, 4.0 m/s)
    assert abs(w["v_ground"] - 4.0) < 0.1, w["v_ground"]
    assert w["n_hours"] == 13 and w["hours"] == "04-16 UT"
    print(f"  [ok] interpolation + 5/3-moment collapse match hand values "
          f"(v_free {w['v_free']}, 16-km bin {per_bin[16.0]} m/s -- the "
          f"FA-event's stratospheric easterlies)")


def error_paths():
    tmp = tempfile.mkdtemp(prefix="winds_err_")
    orig = urllib.request.urlopen
    calls = []

    def dead_service(url, timeout=None):
        calls.append(1)
        raise urllib.error.URLError("connection refused")

    urllib.request.urlopen = dead_service
    try:
        try:
            engine.night_winds("20260722", tmp, refetch=True)
            raise AssertionError("a dead service must raise WindsError")
        except engine.WindsError as e:
            assert "fetch failed" in str(e)
        # exactly one attempt PER HOST (forecast + historical fallback) --
        # bounded at 2, never a retry loop
        assert len(calls) == 2, f"one attempt per host, no retries: {calls}"

        # a timeout is just another URLError flavor -- same promptness rule
        calls.clear()

        def hung_service(url, timeout=None):
            calls.append(timeout)
            raise TimeoutError(f"timed out after {timeout}s")
        urllib.request.urlopen = hung_service
        try:
            engine.night_winds("20260722", tmp, refetch=True, timeout=3)
            raise AssertionError("a hung service must raise WindsError")
        except engine.WindsError:
            pass
        assert calls == [3, 3], f"worst case bounded at 2 x timeout: {calls}"

        # garbage JSON / missing hourly block -> WindsError, not KeyError
        class _Bad:
            def read(self): return b'{"nope": 1}'
            def __enter__(self): return self
            def __exit__(self, *a): return False
        urllib.request.urlopen = lambda url, timeout=None: _Bad()
        try:
            engine.night_winds("20260722", tmp, refetch=True)
            raise AssertionError("schema-less response must raise WindsError")
        except engine.WindsError as e:
            assert "no hourly" in str(e)

        # the REAL fallback case (hit live with 20260525): the forecast host
        # answers but with the date out of its window (all-null levels); the
        # historical host has the data -> success via host #2
        class _Resp:
            def __init__(self, payload): self._p = payload
            def read(self): return json.dumps(self._p).encode()
            def __enter__(self): return self
            def __exit__(self, *a): return False
        hosts_hit = []

        def split_service(url, timeout=None):
            hosts_hit.append("historical" in url)
            if "historical" in url:
                return _Resp({"hourly": _synth_hourly(_EVENT_LADDER)})
            nulled = _synth_hourly(_EVENT_LADDER)
            nulled = {k: (v if k == "time" else [None] * len(v))
                      for k, v in nulled.items()}
            return _Resp({"hourly": nulled})
        urllib.request.urlopen = split_service
        w = engine.night_winds("20260525", tmp, refetch=True)
        assert hosts_hit == [False, True], hosts_hit
        assert w["n_hours"] == 13
        print("  [ok] archived night: forecast host all-null -> historical "
              "host fallback succeeds (one attempt each)")
    finally:
        urllib.request.urlopen = orig
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] dead / hung / garbage service -> prompt WindsError, "
          "bounded attempts (one per host), no retries")


def cache_behavior():
    tmp = tempfile.mkdtemp(prefix="winds_cache_")
    orig = urllib.request.urlopen
    try:
        # seed a cache for a long-finished UT day, then make the network
        # unreachable: the cache must satisfy the call outright
        path = winds._cache_path(tmp, "20260722")
        os.makedirs(tmp, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(_synth_hourly(_EVENT_LADDER), fh)

        def boom(url, timeout=None):
            raise AssertionError("finished-day cache hit must not touch "
                                 "the network")
        urllib.request.urlopen = boom
        w = engine.night_winds("20260722", tmp)
        assert w["n_hours"] == 13
        print("  [ok] a finished UT day is served from cache, zero network")

        # an IN-PROGRESS UT day must refetch (same rule as the MKWC files)
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        with open(winds._cache_path(tmp, today), "w", encoding="utf-8") as fh:
            json.dump(_synth_hourly(_EVENT_LADDER), fh)
        hit = []
        class _Resp:
            def read(self): return json.dumps(
                {"hourly": _synth_hourly(_EVENT_LADDER)}).encode()
            def __enter__(self): return self
            def __exit__(self, *a): return False
        urllib.request.urlopen = lambda url, timeout=None: (
            hit.append(1), _Resp())[1]
        engine.night_winds(today, tmp)
        assert hit, "an in-progress UT day must refetch, not trust the cache"
        print("  [ok] an in-progress UT day refetches (MKWC staleness rule)")
    finally:
        urllib.request.urlopen = orig
        shutil.rmtree(tmp, ignore_errors=True)


def tau0_math():
    # pure-ground night at the reference winds: V_eff = v_ground exactly
    t_g = engine.tau0_seconds(0.66, 0.0, 8.0, 25.0)
    lam = 500e-9
    r0 = 0.98 * lam / np.radians(0.66 / 3600.0)
    assert abs(t_g - 0.314 * r0 / 8.0) < 1e-9
    # pure-FA night: V_eff = v_free exactly
    t_f = engine.tau0_seconds(0.66, 0.66, 8.0, 25.0)
    assert abs(t_f - 0.314 * r0 / 25.0) < 1e-9
    assert t_f < t_g, "faster wind must mean shorter tau0"
    # a mixed night sits between the two pure cases
    t_m = engine.tau0_seconds(0.66, 0.40, 8.0, 25.0)
    assert t_f < t_m < t_g
    # r0 = 16.6 cm (the bundled night's mean) at 8 m/s pure ground:
    # tau0 = 0.314*0.166/8 = 6.5 ms -- sane magnitude
    t = engine.tau0_seconds(0.62, 0.0, 8.0, 25.0)
    assert 0.005 < t < 0.008, t
    # degenerate inputs -> None, never a crash
    assert engine.tau0_seconds(None, 0.3, 8, 25) is None
    assert engine.tau0_seconds(0.0, 0.0, 8, 25) is None
    # FA claimed > total (mismatched instruments): clamped, not negative
    assert engine.tau0_seconds(0.4, 0.9, 8.0, 25.0) is not None
    print(f"  [ok] tau0: pure-ground/pure-FA limits exact, mixed in "
          f"between, sane ms magnitudes ({t_g*1e3:.1f} / {t_m*1e3:.1f} / "
          f"{t_f*1e3:.1f} ms), degenerate inputs -> None")


def main():
    interpolation_and_collapse()
    error_paths()
    cache_behavior()
    tau0_math()


if __name__ == "__main__":
    main()
    print("  [ok] GFS winds contract holds")
