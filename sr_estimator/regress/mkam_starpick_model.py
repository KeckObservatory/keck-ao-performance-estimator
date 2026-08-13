#!/usr/bin/env python3
"""MKAM DIMM-star model (mkam_catalog.py), headless, hand-checked against
the REAL M79 2026-01-31 UT run: over 06:21-07:31 UT Aldebaran (Alp Tau,
dec +16.5 at latitude +19.8) transits 3.3 deg from zenith right before
the run and must dominate a stay-near-zenith scheduler, with Gam Ori and
Iot Aur as the plausible alternates -- the exact candidate set drawn on
the report's FA-geometry figure.  The alt/az is cross-checked against a
full astropy ICRS->AltAz transform (the module deliberately skips
precession/nutation/refraction, so agreement is ~0.4 deg, NOT arcsec).
Display-only pointing model: nothing here touches the estimate, so there
is no harness/byte-identity surface at all."""
import datetime as dt
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
import warnings
warnings.filterwarnings("ignore")
import keck_ao_estimator as engine

M79_RUN = [dt.datetime(2026, 1, 31, 6, 21) + dt.timedelta(minutes=5 * k)
           for k in range(15)]                       # 06:21-07:31 UT grid


def catalog_parses_and_dedupes():
    cat = engine.load_mkam_catalog()
    assert len(cat) == 131, len(cat)
    by = {s["name"]: s for s in cat}
    assert len(by) == len(cat), "names must be unique after dedupe"
    # the hand-added "---" header block duplicates HR stars: the HR row
    # (real V / duplicity data) must win -- Sirius keeps its 11" dm=10.3
    sirius = by["Alp CMa"]
    assert sirius["hr"] == 2491 and abs(sirius["vmag"] + 1.46) < 1e-9
    assert sirius["sep_arcsec"] == 11.0 and sirius["dmag"] == 10.3
    # Spica's header-block line carries a sign typo ("- 11 -9 41"): the
    # dec must come out -(11 + 9/60 + 41/3600), and the HR row wins
    spica = by["Alp Vir"]
    assert spica["hr"] == 5056
    assert abs(spica["dec_deg"] - -(11 + 9 / 60 + 41 / 3600)) < 0.02, spica
    # shifting name-field layouts both prettify
    assert by["Mu1 Sco"]["pretty"] == "μ¹ Sco"
    assert by["Gam Gem"]["pretty"] == "γ Gem"
    # duplicity fields ground-truthed against the file
    capella = by["Alp Aur"]
    assert capella["sep_arcsec"] == 0.0 and capella["dmag"] == 0.5
    print(f"  [ok] catalog: {len(cat)} unique stars; HR rows beat the "
          "hand-added block; Spica sign typo absorbed; names prettified")


def altaz_against_astropy():
    from astropy.utils.iers import conf as iers_conf
    iers_conf.auto_max_age = None
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time
    import math
    loc = EarthLocation(lat=19.8260 * u.deg, lon=-155.4747 * u.deg,
                        height=4145 * u.m)
    cat = {s["name"]: s for s in engine.load_mkam_catalog()}
    worst = 0.0
    for name in ("Gam Gem", "Alp Tau", "Bet Gem"):
        s = cat[name]
        for hh in (6.0, 7.5):
            when = dt.datetime(2026, 1, 31) + dt.timedelta(hours=hh)
            az, el = engine.star_altaz(s["ra_deg"], s["dec_deg"], when)
            ref = SkyCoord(ra=s["ra_deg"] * u.deg,
                           dec=s["dec_deg"] * u.deg).transform_to(
                AltAz(obstime=Time(when), location=loc))
            daz = (abs((az - ref.az.deg + 180.0) % 360.0 - 180.0)
                   * math.cos(math.radians(el)))
            worst = max(worst, daz, abs(el - ref.alt.deg))
    assert worst < 0.5, worst        # precession left out on purpose
    print(f"  [ok] star_altaz within {worst:.2f} deg of astropy "
          "(scheduling-grade: no precession/nutation by design)")


def m79_night_pick():
    ranked = engine.dimm_star_probabilities(M79_RUN)
    assert abs(sum(r["prob"] for r in ranked) - 1.0) < 1e-9
    assert all(ranked[i]["prob"] >= ranked[i + 1]["prob"]
               for i in range(len(ranked) - 1))
    assert all(r["zd_mean"] <= 45.0 + 1e-9 for r in ranked)
    top3 = [r["star"]["name"] for r in ranked[:3]]
    assert top3[0] == "Alp Tau", top3
    assert ranked[0]["prob"] > 0.3 and ranked[0]["pick_frac"] > 0.6
    assert set(top3) == {"Alp Tau", "Gam Ori", "Iot Aur"}, top3
    # stars with a comparable-brightness close companion can never host
    # the monitor, however close to zenith (Capella!)
    names = {r["star"]["name"] for r in ranked}
    for double in ("Alp Aur", "Zet Tau", "Eta Tau", "Alp Phe"):
        assert double not in names, double
    # the figure's mid-run snapshot: Aldebaran just past transit, WSW high
    cands = engine.top_monitor_orientations(
        M79_RUN, dt.datetime(2026, 1, 31, 6, 42), n=3)
    assert cands[0]["pretty"] == "α Tau"
    assert 235.0 < cands[0]["az"] < 250.0 and 81.0 < cands[0]["el"] < 85.0
    print("  [ok] M79 run: Aldebaran dominates "
          f"(P={ranked[0]['prob']:.2f}, pick={ranked[0]['pick_frac']:.2f}) "
          "with Gam Ori / Iot Aur alternates; doubles excluded; mid-run "
          f"snapshot az {cands[0]['az']:.0f} el {cands[0]['el']:.0f}")


def season_matters():
    # same UT clock window in July: Taurus is a day sky -- the pick MUST
    # change (this is the "time of year" input doing real work)
    july = [t.replace(month=7, day=15) for t in M79_RUN]
    ranked = engine.dimm_star_probabilities(july)
    assert ranked, "July evening must still offer near-zenith stars"
    top3 = [r["star"]["name"] for r in ranked[:3]]
    assert "Alp Tau" not in top3, top3
    assert top3[0] != "Alp Tau"
    print(f"  [ok] seasonality: July window picks {top3[0]} "
          "(Aldebaran correctly unavailable)")


def degenerate_inputs():
    assert engine.dimm_star_probabilities([]) == []
    assert engine.top_monitor_orientations(
        [], dt.datetime(2026, 1, 31, 6, 42)) == []
    # an absurd elevation floor leaves no eligible star -> [] not a crash
    assert engine.dimm_star_probabilities(M79_RUN, min_el_deg=89.9) == []
    print("  [ok] degenerate inputs (empty window, impossible elevation "
          "floor) -> [] never a crash")


def main():
    catalog_parses_and_dedupes()
    altaz_against_astropy()
    m79_night_pick()
    season_matters()
    degenerate_inputs()


if __name__ == "__main__":
    main()
    print("  [ok] MKAM star-pick contract holds")
