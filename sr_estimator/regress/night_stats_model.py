#!/usr/bin/env python3
"""night_stats.time_selection_mask / masked_mean: the period-selection logic
behind the Data-tab summary-stats panel (window/night/time), headless.
Cross-checked directly against fieldmap.field_cn2_profile's OWN window/night/
time selection on a real prepared night, so the summary panel can never
silently disagree with the field map about what "observing window" means."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta
import keck_ao_estimator as engine
np = engine.np
DATA = os.path.join(HERE, "data")


def _args(*extra):
    from astropy.utils.iers import conf; conf.auto_max_age = None
    return engine.build_parser().parse_args([
        "--dimm", f"{DATA}/20260525_dimm.dat", "--mass", f"{DATA}/20260525_mass.dat",
        "--masspro", f"{DATA}/20260525_masspro.dat", "--telescope", "K2",
        "--out", os.path.join(HERE, "p25_out.png"), "--force", *extra])


def basic_masks():
    # small synthetic datetime array, no real prep needed for the pure mask logic
    t0 = datetime(2026, 7, 21, 20, 0, 0)
    times = [t0 + timedelta(minutes=15 * i) for i in range(8)]   # 20:00..21:45

    class FakePrep:
        show_target = True
        windows = [(t0 + timedelta(minutes=30), t0 + timedelta(minutes=75))]

        def in_any_window(self, t):
            w0, w1 = self.windows[0]
            return w0 <= t <= w1

    prep = FakePrep()

    sel_night = engine.time_selection_mask(times, "night", None, prep)
    assert sel_night.all(), "'night' must select everything"
    print("  [ok] 'night' selects the whole array")

    sel_win = engine.time_selection_mask(times, "window", None, prep)
    want = [prep.in_any_window(t) for t in times]
    assert list(sel_win) == want, (list(sel_win), want)
    assert 2 <= sel_win.sum() < len(times), "window must be a proper subset here"
    print(f"  [ok] 'window' matches prep.in_any_window exactly "
          f"({sel_win.sum()}/{len(times)} samples)")

    # no observing window configured -> falls back to whole night (matches
    # field_cn2_profile's own fallback, not an empty/all-False mask)
    prep_nowin = FakePrep(); prep_nowin.windows = []
    sel_fallback = engine.time_selection_mask(times, "window", None, prep_nowin)
    assert sel_fallback.all(), "no window configured must fall back to whole night"
    print("  [ok] 'window' with no configured window falls back to whole night")

    # show_target off -> also falls back to whole night even with windows set
    prep_off = FakePrep(); prep_off.show_target = False
    sel_off = engine.time_selection_mask(times, "window", None, prep_off)
    assert sel_off.all(), "show_target=False must fall back to whole night"
    print("  [ok] 'window' with show_target off falls back to whole night")

    # 'time': nearest single sample, exactly one True
    t_query = t0 + timedelta(minutes=40)         # nearest to times[3] (20:45)
    sel_time = engine.time_selection_mask(times, "time", t_query, prep)
    assert sel_time.sum() == 1 and sel_time[3], (list(sel_time),)
    print("  [ok] 'time' selects exactly the nearest single sample")

    # 'time' with t_hst=None falls back to whole night rather than crashing
    sel_time_none = engine.time_selection_mask(times, "time", None, prep)
    assert sel_time_none.all()
    print("  [ok] 'time' with no timestamp given falls back to whole night")

    # empty times array -> empty mask, not a crash
    sel_empty = engine.time_selection_mask([], "night", None, prep)
    assert len(sel_empty) == 0
    print("  [ok] an empty times array returns an empty (not crashing) mask")


def masked_mean_edges():
    vals = np.array([1.0, 2.0, np.nan, 4.0])
    mask = np.array([True, True, True, False])
    assert abs(engine.masked_mean(vals, mask) - 1.5) < 1e-9, \
        "non-finite values under the mask must be dropped before averaging"
    print("  [ok] masked_mean drops NaN under the mask (mean of [1,2] = 1.5)")

    assert engine.masked_mean(vals, np.array([False] * 4)) is None
    assert engine.masked_mean([], np.array([], dtype=bool)) is None
    assert engine.masked_mean(None, mask) is None
    print("  [ok] masked_mean returns None (not NaN/crash) for an empty "
          "selection, empty array, or None")


def against_real_prepared_night():
    """The real cross-check: on an actual prepared night, the summary
    panel's window/night masks applied to res.p_times must reproduce
    field_cn2_profile's own selection (same n, same mean seeing) -- the two
    features must never silently disagree about "observing window"."""
    args = _args("--target", "--window", "21:00-05:00")
    prep = engine.prepare_night(args)
    res = engine.compute_timeline(args, prep)
    if len(res.p_times) == 0:
        print("  [skip] no MASS profiles in the bundled test data")
        return

    for when in ("window", "night"):
        ref = engine.field_cn2_profile(args, prep, res, when=when)
        if ref is None:
            continue
        sel = engine.time_selection_mask(res.p_times, when, None, prep)
        assert int(sel.sum()) == ref["n"], (when, int(sel.sum()), ref["n"])
        got_mass = engine.masked_mean(res.col_mass, sel)
        assert abs(got_mass - ref["eps_fa_zenith"]) < 1e-9, \
            (when, got_mass, ref["eps_fa_zenith"])
        print(f"  [ok] '{when}': {sel.sum()} profiles, matches "
              f"field_cn2_profile's own n and mean MASS seeing exactly")

    # 'time': same nearest-index convention as field_snapshot
    t_hst = res.p_times[len(res.p_times) // 2]
    snap = engine.field_snapshot(args, prep, res, when="time", time_hst=t_hst)
    sel_t = engine.time_selection_mask(res.p_times, "time", t_hst, prep)
    assert sel_t.sum() == 1
    idx = int(np.nonzero(sel_t)[0][0])
    assert abs(res.col_theta0[idx] - snap["theta0_los"]) < 1e-9
    print("  [ok] 'time' selects the same sample field_snapshot() picks for "
          "the same query timestamp")


def main():
    basic_masks()
    masked_mean_edges()
    against_real_prepared_night()


if __name__ == "__main__":
    main()
    print("  [ok] night_stats period-selection contract holds")
