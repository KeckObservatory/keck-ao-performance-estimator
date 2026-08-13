#!/usr/bin/env python3
"""Proper-motion propagation (engine.apply_proper_motion) and SIMBAD target-
name resolution (target_resolve.resolve_target_name) physics/parsing contract,
headless. The live SIMBAD network query itself is untested here (monkeypatch
astroquery.simbad.Simbad.query_object with a synthetic table, matching the
gui_phase20.py precedent for Vizier/parse_catalog_table)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
import warnings
warnings.filterwarnings("ignore")
from datetime import date
import numpy as np
import keck_ao_estimator as engine


def proper_motion():
    ref = engine.parse_radec("15h49m57.7s", "-03d55m16s")

    # 0/0 PM must return the SAME object, unchanged -- the byte-identity
    # contract every call site (target.py's _effective_target_coords) relies
    # on for a target with no known proper motion
    same = engine.apply_proper_motion(ref, 0.0, 0.0, date(2026, 7, 20))
    assert same is ref, "0/0 PM must return the identical object, not a copy"
    same2 = engine.apply_proper_motion(ref, None, None, date(2026, 7, 20))
    assert same2 is ref, "None/None PM must also be a no-op"
    print("  [ok] 0/0 (and None/None) proper motion is an exact no-op")

    # direction/magnitude sanity: a purely +RA*cosDec motion moves the star
    # towards increasing RA (East); a purely +Dec motion moves it North
    east_only = engine.apply_proper_motion(ref, 100.0, 0.0, date(2026, 7, 20))
    assert east_only.ra.deg > ref.ra.deg, "positive PM RA*cosDec must increase RA"
    assert abs(east_only.dec.deg - ref.dec.deg) < 1e-6, \
        "PM RA*cosDec alone must not move Dec"
    north_only = engine.apply_proper_motion(ref, 0.0, 100.0, date(2026, 7, 20))
    assert north_only.dec.deg > ref.dec.deg, "positive PM Dec must increase Dec"
    print("  [ok] PM RA*cosDec / PM Dec move the target East / North as expected")

    # magnitude: PM(mas/yr) * years-since-J2000 = angular shift (small-angle),
    # matches astropy's own SkyCoord.apply_space_motion to <1%
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    import astropy.units as u
    import erfa
    obs = date(2026, 7, 20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", erfa.ErfaWarning)
        want = SkyCoord(ra=ref.ra, dec=ref.dec, frame="icrs",
                        pm_ra_cosdec=37.0 * u.mas / u.yr,
                        pm_dec=-12.0 * u.mas / u.yr,
                        obstime=Time("J2000")).apply_space_motion(
                            new_obstime=Time(obs.isoformat()))
    got = engine.apply_proper_motion(ref, 37.0, -12.0, obs)
    assert got.separation(want).arcsec < 1e-9, \
        "must reproduce a direct astropy apply_space_motion call exactly"
    years = (Time(obs.isoformat()) - Time("J2000")).to(u.yr).value
    expect_arcsec = np.hypot(37.0, 12.0) * years / 1000.0
    assert abs(got.separation(ref).arcsec - expect_arcsec) / expect_arcsec < 0.01, \
        (got.separation(ref).arcsec, expect_arcsec)
    print(f"  [ok] magnitude matches astropy exactly, and the small-angle "
          f"PM*years estimate to <1% ({got.separation(ref).arcsec:.3f}\" over "
          f"{years:.1f} yr)")

    # the ERFA "distance overridden" warning (expected -- no distance given,
    # pure angular propagation by design) must be suppressed, not surfaced
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        engine.apply_proper_motion(ref, 50.0, -20.0, date(2026, 7, 20))
    print("  [ok] the expected ERFA no-distance warning is suppressed")


def name_resolution():
    import astroquery.simbad
    from astropy.table import Table
    import numpy.ma as ma

    # IMPORTANT: patch the CLASS (type(Simbad)), not astroquery.simbad.Simbad
    # -- that module attribute is an INSTANCE of SimbadClass, and calling it
    # (Simbad(), as resolve_target_name does) mints a NEW instance via
    # BaseQuery.__call__, which never sees an instance-level patch. The
    # original version of this test patched the instance, so it silently ran
    # against the REAL network the whole time and only "passed" because the
    # fake values happened to be HD 141569's real ones (caught 2026-07-22).
    simbad_cls = type(astroquery.simbad.Simbad)

    def fake_query(name_col, ra, dec, pmra, pmdec, extra=None):
        def _query_object(self, name):
            if name == "bogus":
                return Table({"main_id": [], "ra": [], "dec": [],
                              "pmra": [], "pmdec": []})
            cols = {
                "main_id": [name_col], "ra": [ra], "dec": [dec],
                "pmra": ma.array([pmra], mask=[pmra is None]),
                "pmdec": ma.array([pmdec], mask=[pmdec is None]),
            }
            for k, v in (extra or {}).items():
                cols[k] = ma.array([v if v is not None else 0.0],
                                   mask=[v is None])
            return Table(cols)
        return _query_object

    # ...and query_object is not the only network call: resolve_target_name
    # first does add_votable_fields("pmra","pmdec","allfluxes"), which current
    # astroquery validates against SIMBAD's live TAP capabilities endpoint. So
    # this test still reached the network on every call -- silently when CDS
    # was up, as a ~14 s stall then a DALServiceError when it was down. The
    # fake tables above already carry the columns, so declaring them is a
    # no-op. (Same root cause as the gui_phase26 CI failure, 2026-07-28.)
    orig = simbad_cls.query_object
    orig_fields = simbad_cls.add_votable_fields
    simbad_cls.add_votable_fields = lambda self, *fields: None
    try:
        # deliberately NOT HD 141569's real values -- if the patch is ever
        # bypassed again (a live network query), these assertions fail loudly
        # instead of accidentally passing against reality
        simbad_cls.query_object = fake_query(
            "FAKE 1", 111.25, 22.5, 123.4, -234.5,
            extra={"V": 9.9, "K": 7.7, "R": None})
        r = engine.resolve_target_name("whatever")
        assert r["name"] == "FAKE 1"
        assert abs(r["ra_deg"] - 111.25) < 1e-9
        assert abs(r["dec_deg"] - 22.5) < 1e-9
        assert abs(r["pmra"] - 123.4) < 1e-9
        assert abs(r["pmdec"] - (-234.5)) < 1e-9
        print(f"  [ok] resolve_target_name parses a SIMBAD-shaped result "
              f"({r['name']}: {r['ra_deg']:.4f},{r['dec_deg']:.4f}, "
              f"PM {r['pmra']},{r['pmdec']} mas/yr) -- synthetic values that "
              f"CANNOT pass against the live network")

        # magnitudes (allfluxes): present columns parse, masked -> None,
        # absent columns (J here) -> None rather than KeyError
        assert r["mags"]["V"] == 9.9 and r["mags"]["K"] == 7.7
        assert r["mags"]["R"] is None, "masked flux must be None"
        assert r["mags"]["J"] is None, "absent flux column must be None"
        print("  [ok] mags: present/masked/absent allfluxes columns -> "
              "value/None/None")

        # masked PM (SIMBAD has no measured value, e.g. Sgr A* itself) -> None,
        # not a crash or a bogus 0.0
        simbad_cls.query_object = fake_query(
            "FAKE 2", 266.416817, -29.007825, None, None)
        r2 = engine.resolve_target_name("anything")
        assert r2["pmra"] is None and r2["pmdec"] is None, \
            "masked/no PM must resolve to None, not 0.0 or a crash"
        print("  [ok] a masked (no measured) proper motion resolves to None")

        # a name that doesn't resolve raises, rather than returning garbage
        simbad_cls.query_object = fake_query("x", 0.0, 0.0, None, None)
        try:
            engine.resolve_target_name("bogus")
            raise AssertionError("an empty SIMBAD result must raise ValueError")
        except ValueError as e:
            assert "did not resolve" in str(e)
        print("  [ok] a name with no SIMBAD match raises ValueError")
    finally:
        simbad_cls.query_object = orig
        simbad_cls.add_votable_fields = orig_fields


def main():
    proper_motion()
    name_resolution()


if __name__ == "__main__":
    main()
    print("  [ok] proper-motion + target-resolution contract holds")
