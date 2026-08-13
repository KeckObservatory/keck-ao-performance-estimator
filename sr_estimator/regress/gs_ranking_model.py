#!/usr/bin/env python3
"""Guide-star auto-ranking (engine only, headless): brighter-at-equal-offset
and closer-at-equal-magnitude both rank higher (falls out of the existing
anisoplanatism/tip-tilt physics, no special-casing in gs_ranking.py itself),
too-faint/no-magnitude stars are excluded with the right reason, and the
ranking's own field_metric_at call reproduces a directly-called equivalent
(the override plumbing added to fieldmap.py for this is wired correctly).
Also confirms the override plumbing left every EXISTING field_map_grid /
field_metric_at caller byte-identical (see the harness note at the end).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
import keck_ao_estimator as engine
np = engine.np
DATA = os.path.join(HERE, "data")


def _star(id_, x, y, **mags):
    return {"id": id_, "ra": 0.0, "dec": 0.0, "x": float(x), "y": float(y),
           "mags": {"R": None, "G": None, "BP": None, "RP": None, "F": None,
                    "V": None, "B": None, "r": None, "i": None, "J": None,
                    "H": None, "K": None, **mags}}


def main():
    from astropy.utils.iers import conf; conf.auto_max_age = None
    a = engine.build_parser().parse_args([
        "--dimm", f"{DATA}/20260525_dimm.dat", "--mass", f"{DATA}/20260525_mass.dat",
        "--masspro", f"{DATA}/20260525_masspro.dat", "--telescope", "K1",
        "--out", "/tmp/f.png", "--force"])
    p = engine.prepare_night(a)
    r = engine.compute_timeline(a, p)
    snap = engine.field_snapshot(a, p, r, "night")
    assert snap is not None
    laser = (5.0, -5.0)

    # --- single-LGS mode: TT-reference ranking -------------------------------
    bright_close = _star("bright_close", 3.0, 0.0, R=9.0)
    faint_close  = _star("faint_close",  3.0, 0.0, R=13.0)
    bright_far   = _star("bright_far",  15.0, 0.0, R=9.0)
    stars = [faint_close, bright_close, bright_far]
    ranked = engine.rank_guide_stars(a, p, snap, "single", stars, laser, "R",
                                     metric="strehl")
    order = [e["id"] for e in ranked if e["rank"] is not None]
    assert order[0] == "bright_close", order
    assert order.index("bright_close") < order.index("faint_close"), \
        "brighter at equal offset must rank higher"
    assert order.index("bright_close") < order.index("bright_far"), \
        "closer at equal magnitude must rank higher"
    assert [e["rank"] for e in ranked if e["rank"] is not None] == \
        list(range(1, len(order) + 1)), "ranks must be 1..N with no gaps"
    print(f"  [ok] single-LGS: brighter-at-equal-offset and closer-at-equal-mag "
          f"both rank first ({order})")

    # --- FWHM metric: lower is better (opposite sort direction) -------------
    ranked_fwhm = engine.rank_guide_stars(a, p, snap, "single", stars, laser,
                                          "R", metric="fwhm")
    order_fwhm = [e["id"] for e in ranked_fwhm if e["rank"] is not None]
    assert order_fwhm[0] == "bright_close", order_fwhm
    vals = [e["delivered_value"] for e in ranked_fwhm if e["rank"] is not None]
    assert vals == sorted(vals), "FWHM ranking must be ascending (smaller=better)"
    print(f"  [ok] FWHM metric ranks ascending (smaller=better): {vals}")

    # --- exclusion: no derivable magnitude, and too-faint --------------------
    no_mag = _star("no_mag", 2.0, 2.0)               # every band None
    too_faint = _star("too_faint", 2.0, 2.0, R=25.0)  # > SENSOR_FAINT_LIMIT["R"]
    stars2 = [bright_close, no_mag, too_faint]
    ranked2 = engine.rank_guide_stars(a, p, snap, "single", stars2, laser, "R",
                                      metric="strehl")
    excl = {e["id"]: e["excluded_reason"] for e in ranked2 if e["rank"] is None}
    assert "no_mag" in excl and "no derivable" in excl["no_mag"], excl
    assert "too_faint" in excl and "too faint" in excl["too_faint"], excl
    assert ranked2[-1]["rank"] is None and ranked2[-2]["rank"] is None, \
        "excluded stars must sort after every ranked one"
    assert ranked2[0]["id"] == "bright_close" and ranked2[0]["rank"] == 1
    print(f"  [ok] exclusion: {excl}")

    # --- consistency: ranking's own evaluation matches a direct field_metric_at
    # call with the equivalent override (the plumbing added to fieldmap.py is
    # wired correctly, not just internally self-consistent)
    direct = engine.field_metric_at(
        a, p, snap, "single", "strehl", (0.0, 0.0), (3.0, 0.0), laser,
        (0.0, 0.0), tt_mag_override=9.0)
    via_rank = next(e for e in ranked if e["id"] == "bright_close")["delivered_value"]
    assert abs(direct - via_rank) < 1e-12, (direct, via_rank)
    print(f"  [ok] ranking's field_metric_at call matches a direct override call "
          f"({direct:.4f})")

    # --- NGS mode: ranks the star AS the NGS/science reference itself -------
    ngs_close = _star("ngs_close", 2.0, 0.0, R=9.0)
    ngs_far = _star("ngs_far", 8.0, 0.0, R=9.0)
    ranked_ngs = engine.rank_guide_stars(a, p, snap, "ngs",
                                         [ngs_far, ngs_close], laser, "R",
                                         metric="strehl")
    order_ngs = [e["id"] for e in ranked_ngs]
    assert order_ngs[0] == "ngs_close", order_ngs
    direct_ngs = engine.field_metric_at(
        a, p, snap, "ngs", "strehl", (2.0, 0.0), (0.0, 0.0), laser,
        (0.0, 0.0), ngs_bright_override=9.0)
    via_rank_ngs = ranked_ngs[0]["delivered_value"]
    assert abs(direct_ngs - via_rank_ngs) < 1e-12, (direct_ngs, via_rank_ngs)
    print(f"  [ok] NGS mode ranks the star as the NGS reference itself "
          f"({order_ngs}, matches direct override call)")

    # --- LTAO mode: same wiring works (cn2_bins path) ------------------------
    ranked_ltao = engine.rank_guide_stars(a, p, snap, "ltao", stars, laser, "R",
                                          metric="strehl")
    assert [e["id"] for e in ranked_ltao if e["rank"] is not None][0] == \
        "bright_close"
    print("  [ok] LTAO mode: same ranking behavior (cn2_bins path)")

    # --- override plumbing did not disturb existing (no-override) callers ---
    # exact same field_metric_at call as before the override kwargs were
    # added, with no override arguments passed -- must reproduce the science-
    # direction value exactly like the existing field_map_grid contract does.
    sci = engine.lgs_strehl(
        snap["eps_tot_los"], snap["eps_fa_los"], "K1", "single", p.lam_nm,
        tt_mag=a.tt_mag, tt_offset=a.tt_offset, lgs_offset=7.0,
        legacy=a.legacy_budget, bw_factor=p._ltao_bw_fac,
        v_ground=a.wind_ground, v_free=a.wind_free)
    via_default = engine.field_metric_at(
        a, p, snap, "single", "strehl", (0.0, 0.0), (0.0, a.tt_offset),
        (7.0 / 2**0.5, -7.0 / 2**0.5), (0.0, 0.0))
    assert abs(sci - via_default) < 1e-9, (sci, via_default)
    print("  [ok] no-override field_metric_at call unchanged (byte-identity "
          "contract; also re-verify with harness.py check --local)")

    # --- optical-reddening safety (dusty-field IR->R) ------------------------
    # the extinction lower bound itself: normal IR colour -> nothing; a
    # strongly reddened colour -> a big A_R with a note; no colour -> nothing
    assert engine.optical_extinction_lower_bound({"J": 10.0, "K": 9.3}) == (0.0, None)
    a_r, note = engine.optical_extinction_lower_bound({"J": 12.4, "H": 11.7, "K": 9.4})
    assert a_r > 5 and note and "J-K" in note, (a_r, note)
    assert engine.optical_extinction_lower_bound({"K": 9.0}) == (0.0, None)
    # H-K path works when J is absent
    ahk, nhk = engine.optical_extinction_lower_bound({"H": 11.7, "K": 9.4})
    assert ahk > 3 and "H-K" in nhk, (ahk, nhk)
    print(f"  [ok] optical_extinction_lower_bound: normal->none, J-K=3.0 -> {note}")

    # a reddened IR-only star (J-K=3.0) close in: naive R from J looks usable,
    # but for STRAP(R) it is ranked on mag+A_R and EXCLUDED once that lower
    # bound crosses the R limit -- while remaining a fine TRICK(K) star.
    red = _star("reddened", 3.0, 0.0, J=12.4, H=11.7, K=9.4)     # J-K=3.0
    good = _star("good", 3.0, 0.0, R=12.0)
    r_R = engine.rank_guide_stars(a, p, snap, "single", [good, red], laser, "R",
                                  metric="strehl")
    by = {e["id"]: e for e in r_R}
    assert by["reddened"]["rank"] is None, "reddened star must not stay ranked for STRAP(R)"
    assert "IR-red" in by["reddened"]["excluded_reason"], by["reddened"]
    assert by["reddened"]["mag_effective"] > by["reddened"]["mag"] + 5, by["reddened"]
    assert by["good"]["rank"] == 1
    print(f"  [ok] STRAP(R): reddened IR star excluded ({by['reddened']['excluded_reason']})")
    # the SAME star for TRICK(K): direct K photometry -> usable, no penalty
    r_K = engine.rank_guide_stars(a, p, snap, "single", [good, red], laser, "K",
                                  metric="strehl")
    byk = {e["id"]: e for e in r_K}
    assert byk["reddened"]["rank"] is not None, "reddened star must be usable for TRICK(K)"
    assert byk["reddened"]["reddening_note"] is None and \
        byk["reddened"]["mag_effective"] == byk["reddened"]["mag"], byk["reddened"]
    print("  [ok] TRICK(K): the same reddened star is usable (direct K photometry)")

    # a MILDLY reddened star (within the limit) stays ranked but is flagged,
    # and ranks conservatively (mag_effective > mag)
    mild = _star("mild", 3.0, 0.0, J=11.0, H=10.6, K=9.5)       # J-K=1.5
    ref = _star("ref", 3.0, 0.0, R=11.0)
    r_m = engine.rank_guide_stars(a, p, snap, "single", [mild, ref], laser, "R",
                                  metric="strehl")
    bym = {e["id"]: e for e in r_m}
    assert bym["mild"]["rank"] is not None, "mild reddening must stay ranked (within limit)"
    assert bym["mild"]["reddening_note"] is not None, "mild reddening must be flagged"
    assert bym["mild"]["mag_effective"] > bym["mild"]["mag"], "must rank conservatively"
    print(f"  [ok] mild reddening: kept but flagged ({bym['mild']['reddening_note']}), "
          f"ranked on mag_eff {bym['mild']['mag_effective']:.1f} > mag "
          f"{bym['mild']['mag']:.1f}")


if __name__ == "__main__":
    main()
    print("  [ok] guide-star ranking physics contract holds")
