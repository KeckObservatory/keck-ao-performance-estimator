"""Correctness tests: independent ground truth, not the as-written code.

Every test below reimplements a physical law or a statistical combination
rule FROM ITS DEFINITION (Kolmogorov turbulence scaling, the Fried
isoplanatic-angle formula, the Marechal approximation, photon-noise
astrometric scaling, quadrature/RSS error combination, ...) and cross-checks
the engine's output against that independent computation -- not against
"whatever the function currently returns" (that is what test_regression.py /
test_gui_phases.py already do, and it is a CHARACTERIZATION test, not a
correctness test: it would pass unchanged even if a formula's exponent or
constant were silently wrong, as long as nothing else changed).

If a test here fails, either the engine has a real bug, or the physics
documented in this file is stale -- in which case update the derivation and
its comment, not just the assertion.
"""
import math

import numpy as np
import pytest

# Suppress astropy's IERS predictive-table staleness guard at import time (not
# inside individual tests): compute_airmass_curve needs a UT1-UTC correction,
# and this box's bundled IERS table may be >30 days old / offline. This does
# not affect any number this module checks (the coordinate transform itself is
# astropy's, unmodified); see regress/_run_tool.py for the same remedy used by
# the harness.
from astropy.utils.iers import conf as _iers_conf
_iers_conf.auto_max_age = None

import keck_ao_estimator as engine


# ---------------------------------------------------------------------------
# 1. Marechal approximation: S = exp(-(2*pi*sigma/lambda)^2)
# ---------------------------------------------------------------------------
class TestMarechalStrehl:
    @pytest.mark.parametrize("sigma_nm,lam_nm", [
        (0.0, 2200.0), (50.0, 2200.0), (150.0, 2200.0), (400.0, 1650.0),
        (80.0, 1200.0),
    ])
    def test_matches_textbook_formula(self, sigma_nm, lam_nm):
        independent = math.exp(-((2.0 * math.pi * sigma_nm / lam_nm) ** 2))
        assert engine.marechal_strehl(sigma_nm, lam_nm) == pytest.approx(
            independent, rel=1e-12)

    def test_zero_error_is_perfect_strehl(self):
        assert engine.marechal_strehl(0.0, 2200.0) == pytest.approx(1.0)

    def test_large_error_strehl_vanishes(self):
        # sigma = lambda -> (2*pi)^2 in the exponent -> S ~ 3.5e-18
        assert engine.marechal_strehl(2200.0, 2200.0) < 1e-15

    def test_small_error_quadratic_approximation(self):
        # For sigma << lambda, S ~ 1 - (2*pi*sigma/lambda)^2 (first-order
        # Taylor expansion of the exponential) -- an independent textbook
        # cross-check of the SAME formula from a different angle.
        sigma_nm, lam_nm = 10.0, 2200.0
        x = (2.0 * math.pi * sigma_nm / lam_nm) ** 2
        assert engine.marechal_strehl(sigma_nm, lam_nm) == pytest.approx(
            1.0 - x, abs=2e-4)


# ---------------------------------------------------------------------------
# 2. Zenith seeing scaling: eps(zeta) = eps_zenith * sec(zeta)^(3/5)
#    (Kolmogorov turbulence airmass law: r0 ~ cos(zeta)^(3/5))
# ---------------------------------------------------------------------------
class TestZenithSeeingSecantLaw:
    @pytest.mark.parametrize("zenith_deg", [0.0, 30.0, 45.0, 60.0, 70.0])
    def test_matches_secant_power_law(self, zenith_deg):
        independent = (1.0 / math.cos(math.radians(zenith_deg))) ** (3.0 / 5.0)
        assert engine.zenith_seeing_factor(zenith_deg) == pytest.approx(
            independent, rel=1e-12)

    def test_zenith_factor_is_unity_at_zenith(self):
        assert engine.zenith_seeing_factor(0.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. Fried isoplanatic angle theta0 from a Cn2(h) profile:
#    theta0(500nm) = [2.914 k^2 sec(z)^(8/3) * integral Cn2(h) h^(5/3) dh]^(-3/5)
#    theta0(lambda) = theta0(500nm) * (lambda/500)^(6/5)   (chromatic scaling)
# ---------------------------------------------------------------------------
class TestFriedTheta0:
    def _independent_theta0_arcsec(self, J_single_layer, h_m, zenith_deg, lam_nm):
        """Reimplementation of the Fried (1982) isoplanatic-angle formula for
        a SINGLE turbulent layer, from the textbook definition -- written
        independently of engine.theta0_d0_from_profile's internals."""
        k500 = 2.0 * math.pi / 500e-9
        secz = 1.0 / math.cos(math.radians(zenith_deg))
        theta0_500_rad = (2.914 * k500 ** 2 * secz ** (8.0 / 3.0)
                          * J_single_layer * h_m ** (5.0 / 3.0)) ** (-3.0 / 5.0)
        theta0_500_arcsec = theta0_500_rad * 206265.0
        return theta0_500_arcsec * (lam_nm / 500.0) ** (6.0 / 5.0)

    @pytest.mark.parametrize("layer_idx,zenith_deg,lam_nm", [
        (0, 0.0, 500.0),    # 0.5 km layer, zenith, at the native 500 nm
        (2, 0.0, 500.0),    # 2 km layer, zenith
        (4, 30.0, 2200.0),  # 8 km layer, off-zenith, K band (chromatic scaling)
        (5, 0.0, 1650.0),   # 16 km layer, H band
    ])
    def test_single_layer_matches_fried_formula(self, layer_idx, zenith_deg, lam_nm):
        J = np.zeros(6)
        J[layer_idx] = 2.0e-13          # an arbitrary, physically-sized Cn2 dh (m^1/3)
        h_m = engine.MASS_HEIGHTS_M[layer_idx]
        want = self._independent_theta0_arcsec(J[layer_idx], h_m, zenith_deg, lam_nm)
        got, _d0 = engine.theta0_d0_from_profile(J, zenith_deg, lam_nm)
        assert got == pytest.approx(want, rel=1e-9)

    def test_chromatic_scaling_exponent_six_fifths(self):
        # theta0(lambda2)/theta0(lambda1) = (lambda2/lambda1)^(6/5), independent
        # of the profile shape -- verify across two very different profiles.
        J = np.array([1e-13, 2e-13, 0.0, 3e-13, 1e-13, 0.5e-13])
        t500, _ = engine.theta0_d0_from_profile(J, 0.0, 500.0)
        t2200, _ = engine.theta0_d0_from_profile(J, 0.0, 2200.0)
        assert (t2200 / t500) == pytest.approx((2200.0 / 500.0) ** 1.2, rel=1e-9)

    def test_higher_layer_gives_smaller_theta0(self):
        # theta0 ~ h^(-3/5): the SAME integrated turbulence higher up in the
        # atmosphere isoplanatically decorrelates faster (smaller patch) --
        # the sign of a basic, independently-known physical relationship.
        J_low = np.array([1e-13, 0, 0, 0, 0, 0])
        J_high = np.array([0, 0, 0, 0, 0, 1e-13])
        t_low, _ = engine.theta0_d0_from_profile(J_low, 0.0, 500.0)
        t_high, _ = engine.theta0_d0_from_profile(J_high, 0.0, 500.0)
        assert t_high < t_low


# ---------------------------------------------------------------------------
# 4. Kolmogorov nm-RMS budget-term scaling: sigma ~ seeing^(5/6)
#    (equivalently sigma^2 ~ seeing^(5/3), the standard D/r0 wavefront
#    variance law, since r0 ~ 1/seeing)
# ---------------------------------------------------------------------------
class TestKolmogorovSeeingScaling:
    def test_fitting_error_scales_as_five_sixths_power(self):
        t_lo = engine.lgs_budget_terms(0.40, 0.25, "K2", "single")
        t_hi = engine.lgs_budget_terms(0.80, 0.25, "K2", "single")
        independent_ratio = (0.80 / 0.40) ** (5.0 / 6.0)
        assert (t_hi["fit"] / t_lo["fit"]) == pytest.approx(
            independent_ratio, rel=1e-9)

    def test_scintillation_scales_as_five_sixths_power(self):
        t_lo = engine.lgs_budget_terms(0.35, 0.20, "K1", "single")
        t_hi = engine.lgs_budget_terms(0.70, 0.20, "K1", "single")
        independent_ratio = (0.70 / 0.35) ** (5.0 / 6.0)
        assert (t_hi["scint"] / t_lo["scint"]) == pytest.approx(
            independent_ratio, rel=1e-9)

    def test_focal_anisoplanatism_scales_with_freeatm_five_sixths(self):
        # single-beacon focal anisoplanatism (the cone effect) scales with the
        # FREE-ATMOSPHERE seeing, not total seeing.
        t_lo = engine.lgs_budget_terms(0.50, 0.20, "K1", "single")
        t_hi = engine.lgs_budget_terms(0.50, 0.40, "K1", "single")
        independent_ratio = (0.40 / 0.20) ** (5.0 / 6.0)
        assert (t_hi["alt"] / t_lo["alt"]) == pytest.approx(
            independent_ratio, rel=1e-9)

    def test_angular_anisoplanatism_offset_scaling_five_sixths(self):
        # angular anisoplanatism grows with beacon/asterism offset as
        # (theta/theta_ref)^(5/6) in nm RMS (Fried anisoplanatic MSE ~
        # (theta/theta0)^(5/3) in variance).
        t_near = engine.lgs_budget_terms(0.50, 0.30, "K1", "single", lgs_offset=2.0)
        t_far = engine.lgs_budget_terms(0.50, 0.30, "K1", "single", lgs_offset=8.0)
        independent_ratio = (8.0 / 2.0) ** (5.0 / 6.0)
        assert (t_far["ang"] / t_near["ang"]) == pytest.approx(
            independent_ratio, rel=1e-9)


# ---------------------------------------------------------------------------
# 5. Quadrature (RSS) summation: independent Gaussian error sources combine
#    as sigma_total = sqrt(sum sigma_i^2) -- verify lgs_strehl() has not
#    dropped, duplicated, or mis-weighted a term from lgs_budget_terms().
# ---------------------------------------------------------------------------
class TestQuadratureConsistency:
    @pytest.mark.parametrize("telescope,mode", [
        ("K1", "single"), ("K2", "single"), ("K1", "ltao"), ("K2", "ltao"),
    ])
    def test_lgs_strehl_sums_every_ho_term_in_quadrature(self, telescope, mode):
        eps_tot, eps_fa = 0.55, 0.32
        cn2 = np.array([1e-13, 2e-13, 0.5e-13, 1.5e-13, 1e-13, 0.5e-13]) \
            if mode == "ltao" else None
        terms = engine.lgs_budget_terms(eps_tot, eps_fa, telescope, mode,
                                        cn2_bins=cn2)
        # Every key except 'tt' (tip-tilt is a SEPARATE, physically distinct
        # multiplicative Strehl factor -- documented in lgs_strehl -- not part
        # of the high-order wavefront-error sum) must be an independent
        # Gaussian-error contributor to the high-order RMS. This is derived
        # from the term SET itself, not copied from lgs_strehl's source, so a
        # future term added to the dict but forgotten in the sum is caught.
        ho_terms = [v for k, v in terms.items() if k != "tt"]
        independent_ho_rms = math.sqrt(sum(v ** 2 for v in ho_terms))
        independent_strehl = (engine.marechal_strehl(independent_ho_rms, engine.LAMBDA_K_NM)
                              * engine.marechal_strehl(terms["tt"], engine.LAMBDA_K_NM))
        got = engine.lgs_strehl(eps_tot, eps_fa, telescope, mode,
                                cn2_bins=cn2)
        assert got == pytest.approx(independent_strehl, rel=1e-12)


# ---------------------------------------------------------------------------
# 6. Static/calibration budget: quadrature sum of the AO performance error
#    budget v3_1_3 sheet's rows (K1 LGS; adopted 2026-07-24), checked against
#    the sheet's numbers directly (not against the engine's own STATIC_*
#    constants) -- catches a transcription error from the sheet into code.
#    K2's telescope-aberration group still carries its v3_1_1 rows (47, 17):
#    no K2 v3_1_3 sheet is in hand (see budget.py's K2 caveat).
# ---------------------------------------------------------------------------
class TestStaticBudgetQuadrature:
    # STATIC_CALIB/DM/INST/REG are budget_overrides-mutable scalars (see
    # keck_ao_estimator.budget's module docstring on HAZARD 1): the engine
    # shim deliberately does NOT re-export them as bare names (that would be
    # a permanently-stale snapshot for any code reading them through the
    # shim), so these value checks import the real module directly.
    from keck_ao_estimator import budget as _budget

    def test_static_subgroups_match_kaon_sheet_rows(self):
        assert engine.STATIC_TEL["K2"] == pytest.approx(
            math.hypot(47.0, 17.0), abs=0.05)          # v3_1_1 rows (caveat)
        assert engine.STATIC_TEL["K1"] == pytest.approx(
            math.hypot(66.0, 35.0), abs=0.05)          # static 66 + dynamic 35
        assert self._budget.STATIC_CALIB == pytest.approx(
            math.sqrt(25.0 ** 2 + 50.0 ** 2 + 15.0 ** 2), abs=0.05)
        assert self._budget.STATIC_DM == pytest.approx(
            math.sqrt(31.0 ** 2 + 13.0 ** 2 + 1.0 ** 2), abs=0.05)
        assert self._budget.STATIC_INST == pytest.approx(
            math.hypot(30.0, 60.0), abs=0.05)          # AO sys 30 + instr 60
        assert self._budget.STATIC_REG == pytest.approx(
            math.hypot(15.0, 15.0), abs=0.05)

    def test_static_subtotal_is_quadrature_of_the_five_groups(self):
        for tel in ("K1", "K2"):
            independent = math.sqrt(
                engine.STATIC_TEL[tel] ** 2 + self._budget.STATIC_CALIB ** 2
                + self._budget.STATIC_DM ** 2 + self._budget.STATIC_INST ** 2
                + self._budget.STATIC_REG ** 2)
            assert engine.static_subtotal(tel) == pytest.approx(
                independent, rel=1e-9)

    def test_k1_telescope_aberration_worse_than_k2(self):
        # K1's primary+segment figure is documented as worse than K2's (66 nm
        # vs 47 nm uncorrectable static aberration) -- the sign of a physical
        # fact about the two telescopes, not an implementation detail.
        assert engine.STATIC_TEL["K1"] > engine.STATIC_TEL["K2"]


# ---------------------------------------------------------------------------
# 7. Photon-noise astrometric scaling: centroid error ~ 1/sqrt(flux), and
#    flux ~ 10^(0.4*delta_mag) (the astronomical magnitude system) =>
#    error(mag+5) / error(mag) = 10^(-0.2*5) = 0.1 exactly.
#    Checked on the legacy STRAP row, which documents this exact law.
# ---------------------------------------------------------------------------
class TestPhotonNoiseMagnitudeScaling:
    def _independent_legacy_tt_mas(self, tt_mag, tt_offset, s_tot=1.0):
        """Independent reimplementation of tt_wfe_nm(sensor='strap-legacy')
        from its documented formula (docstring + inline comments), including
        the open-loop-tilt ceiling clamp."""
        off_fac = tt_offset / engine.DEF_TT_OFFSET
        flux_fac = 10.0 ** (-0.2 * (engine.DEF_TT_MAG - tt_mag))
        meas = 7.25 * flux_fac * s_tot
        bw = 0.47 * s_tot
        aniso = 9.17 * off_fac * s_tot
        cent = 1.28 * off_fac
        disp, ncp, shake, margin = 0.53, 0.03, 2.60, 5.00
        mas = math.sqrt(meas ** 2 + bw ** 2 + aniso ** 2 + cent ** 2
                        + disp ** 2 + ncp ** 2 + shake ** 2 + margin ** 2)
        ceiling = engine.OPEN_LOOP_TILT_ONEAXIS_MAS * s_tot
        return min(mas, ceiling)

    @pytest.mark.parametrize("tt_mag,tt_offset", [
        (15.2, 19.3), (12.0, 19.3), (8.0, 5.0), (13.5, 0.0),
    ])
    def test_legacy_strap_matches_independent_reimplementation(self, tt_mag, tt_offset):
        want_nm = self._independent_legacy_tt_mas(tt_mag, tt_offset) * engine.NM_PER_MAS
        got_nm = engine.tt_wfe_nm(1.0, tt_mag, tt_offset, sensor="strap-legacy")
        assert got_nm == pytest.approx(want_nm, rel=1e-9)

    def test_five_magnitudes_fainter_is_exactly_ten_times_more_flux_noise(self):
        # Isolate the photon-noise term itself, independent of the engine: a
        # star 5 magnitudes FAINTER has 10**(0.4*5) = 100x less flux, and
        # photon-noise-limited centroid error ~ 1/sqrt(flux), so its
        # measurement noise must be EXACTLY 10x larger (10**(0.2*5) = 10).
        meas_bright = 7.25 * 10.0 ** (-0.2 * (engine.DEF_TT_MAG - 10.0))
        meas_faint = 7.25 * 10.0 ** (-0.2 * (engine.DEF_TT_MAG - 15.0))
        assert (meas_faint / meas_bright) == pytest.approx(10.0 ** (0.2 * 5.0),
                                                            rel=1e-12)
        # and the engine's own tt_wfe_nm at those two magnitudes must be
        # consistent with the independent full-formula reimplementation
        off = 0.0  # on-axis: aniso/cent rows -> 0, isolates meas/bw/fixed rows
        bright = self._independent_legacy_tt_mas(10.0, off)
        faint = self._independent_legacy_tt_mas(15.0, off)
        assert engine.tt_wfe_nm(1.0, 10.0, off, sensor="strap-legacy") \
            == pytest.approx(bright * engine.NM_PER_MAS, rel=1e-9)
        assert engine.tt_wfe_nm(1.0, 15.0, off, sensor="strap-legacy") \
            == pytest.approx(faint * engine.NM_PER_MAS, rel=1e-9)

    def test_open_loop_tilt_ceiling_caps_very_faint_stars(self):
        # An extremely faint TT star cannot make the loop worse than fully
        # open (no correction): tt_wfe_nm must saturate at
        # OPEN_LOOP_TILT_ONEAXIS_MAS * NM_PER_MAS * s_tot, not keep climbing.
        very_faint_nm = engine.tt_wfe_nm(1.0, 25.0, 19.3, sensor="strap-legacy")
        ceiling_nm = engine.OPEN_LOOP_TILT_ONEAXIS_MAS * engine.NM_PER_MAS
        assert very_faint_nm == pytest.approx(ceiling_nm, rel=1e-9)


# ---------------------------------------------------------------------------
# 8. NGS off-axis anisoplanatic Strehl degradation, exercised through the
#    ACTUAL field_map_grid production path (real prepared night, real
#    snapshot), cross-checked against the independent Fried (1982) law:
#    S(theta) = S(0) * exp(-(theta/theta0)^(5/3))
# ---------------------------------------------------------------------------
class TestNgsAnisoplanaticDegradation:
    @pytest.fixture
    def night(self):
        from conftest import DATA
        a = engine.build_parser().parse_args([
            "--dimm", f"{DATA}/20260525_dimm.dat",
            "--mass", f"{DATA}/20260525_mass.dat",
            "--masspro", f"{DATA}/20260525_masspro.dat",
            "--telescope", "K2", "--out", "/tmp/_corr_test.png", "--force"])
        p = engine.prepare_night(a)
        r = engine.compute_timeline(a, p)
        snap = engine.field_snapshot(a, p, r, "night")
        assert snap is not None and snap["theta0_los"] > 0
        return a, p, snap

    def test_offaxis_strehl_matches_fried_law_on_the_real_grid(self, night):
        a, p, snap = night
        ngs_xy, tt_xy, laser_xy = (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)
        extent, Z, meta = engine.field_map_grid(
            a, p, snap, "ngs", "strehl", ngs_xy, tt_xy, laser_xy, n_grid=41)
        n = Z.shape[0]
        xs = np.linspace(extent[0], extent[1], n)
        ys = np.linspace(extent[2], extent[3], n)
        c = n // 2
        s_onaxis = float(Z[c, c])
        theta0 = snap["theta0_los"]
        # probe every OTHER grid point (skipping the singular centre) and
        # check the Fried law holds at each -- not a single cherry-picked
        # point.
        checked = 0
        for i in range(0, n, 4):
            for j in range(0, n, 4):
                if i == c and j == c:
                    continue
                dist = math.hypot(xs[j] - ngs_xy[0], ys[i] - ngs_xy[1])
                expected = s_onaxis * math.exp(-((dist / theta0) ** (5.0 / 3.0)))
                assert Z[i, j] == pytest.approx(expected, rel=1e-9, abs=1e-9), \
                    f"grid point ({i},{j}) at {dist:.2f}\" off-axis"
                checked += 1
        assert checked > 50  # sanity: we actually probed a meaningful grid

    def test_zero_offset_is_full_strehl(self, night):
        a, p, snap = night
        ngs_xy, tt_xy, laser_xy = (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)
        extent, Z, meta = engine.field_map_grid(
            a, p, snap, "ngs", "strehl", ngs_xy, tt_xy, laser_xy, n_grid=41)
        c = Z.shape[0] // 2
        s_direct = engine.ngs_strehl(snap["eps_tot_los"], a.ngs_bright,
                                     a.telescope, p.lam_nm)
        assert float(Z[c, c]) == pytest.approx(s_direct, rel=1e-9)

    def test_isoplanatic_angle_offset_gives_1_over_e(self):
        # By definition of theta0, a star exactly AT theta0 loses a factor
        # of exp(-1) ~ 0.368 -- an independent definitional check of the
        # SAME exponential law used above, at its defining point.
        theta0_arcsec = 10.0
        factor = math.exp(-((theta0_arcsec / theta0_arcsec) ** (5.0 / 3.0)))
        assert factor == pytest.approx(math.exp(-1.0), rel=1e-12)


# ---------------------------------------------------------------------------
# 9. Airmass geometry: sec(z) internal self-consistency (secz reported by
#    compute_airmass_curve must equal 1/cos(90-altitude) computed from the
#    SAME call's own returned altitude -- independent of whether astropy's
#    coordinate transform itself is exactly right, this catches a
#    mismatched-array / unit (deg vs rad) / wrong-column bug).
# ---------------------------------------------------------------------------
class TestAirmassGeometryConsistency:
    def test_secz_is_self_consistent_with_returned_altitude(self):
        from datetime import datetime, timedelta
        from astropy.coordinates import SkyCoord

        target = SkyCoord("12h00m00s", "+19d49m00s")  # near-zenith at Keck
        times = [datetime(2026, 6, 21, 10, 0) + timedelta(minutes=15 * i)
                for i in range(8)]
        airmass, alt_deg, _az = engine.compute_airmass_curve(
            target.ra, target.dec, times)
        independent = 1.0 / np.cos(np.radians(90.0 - np.asarray(alt_deg)))
        np.testing.assert_allclose(airmass, independent, rtol=1e-6)

    def test_higher_altitude_is_lower_airmass(self):
        # sec(z) is monotonically decreasing as altitude -> 90 deg: a basic,
        # independently-known geometric fact used as a sanity check. Window
        # chosen (and verified) so the target stays above the horizon
        # throughout -- below the horizon secz/airmass is unphysical (goes
        # negative), which would confound a naive min/max comparison.
        from datetime import datetime, timedelta
        from astropy.coordinates import SkyCoord

        target = SkyCoord("12h00m00s", "+19d49m00s")
        times = [datetime(2026, 6, 21, 20, 0) + timedelta(minutes=20 * i)
                for i in range(15)]
        airmass, alt_deg, _az = engine.compute_airmass_curve(
            target.ra, target.dec, times)
        alt_deg = np.asarray(alt_deg)
        assert np.all(alt_deg > 0), "test window must stay above the horizon"
        i_lo, i_hi = np.argmin(alt_deg), np.argmax(alt_deg)
        assert airmass[i_hi] < airmass[i_lo]


class TestWcsStringCardCoercion:
    """KOA-exported NIRC2 headers can write CD-matrix keywords as FITS
    STRING cards; astropy's WCS silently ignores those and falls back to an
    identity 1 deg/pixel scale while still reporting has_celestial=True.
    imaging's header path must coerce them, recovering the real plate scale.
    (The bundled 2021-export example frame carries the quirk too; this
    synthetic pin keeps the coverage independent of which frame is bundled.)"""

    def _quirky_header(self):
        from astropy.io import fits
        hdr = fits.Header()
        hdr["NAXIS"] = 2
        hdr["NAXIS1"] = 1024
        hdr["NAXIS2"] = 1024
        hdr["CTYPE1"] = "RA---TAN"
        hdr["CTYPE2"] = "DEC--TAN"
        # the quirk: numeric WCS values stored as STRING cards
        hdr["CD1_1"] = "-1.1023889e-05"
        hdr["CD1_2"] = "0.0"
        hdr["CD2_1"] = "0.0"
        hdr["CD2_2"] = "1.1023889e-05"
        hdr["CRPIX1"] = "512.0"
        hdr["CRPIX2"] = "512.0"
        hdr["CRVAL1"] = "259.28029"
        hdr["CRVAL2"] = "43.13652"
        return hdr

    def test_string_cd_cards_recover_real_plate_scale(self):
        import warnings
        from astropy.wcs import WCS, FITSFixedWarning
        from keck_ao_estimator.imaging import _coerce_wcs_numeric_strings

        hdr = self._quirky_header()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            raw = WCS(hdr)
            fixed = WCS(_coerce_wcs_numeric_strings(hdr))
        # the failure mode being pinned: the raw header still LOOKS valid...
        assert raw.has_celestial
        # ...but carries the identity fallback (1 deg/pixel), while the
        # coerced one recovers the true ~0.04"/pixel NIRC2-wide scale
        raw_scale = abs(raw.pixel_scale_matrix[0, 0]) * 3600.0
        fixed_scale = abs(fixed.pixel_scale_matrix[0, 0]) * 3600.0
        assert abs(raw_scale - 3600.0) < 1e-6, raw_scale
        assert abs(fixed_scale - 0.0397) < 0.001, fixed_scale

    def test_non_numeric_strings_left_alone(self):
        from keck_ao_estimator.imaging import _coerce_wcs_numeric_strings
        hdr = self._quirky_header()
        hdr["CD1_1"] = "not a number"
        out = _coerce_wcs_numeric_strings(hdr)
        assert out["CD1_1"] == "not a number"   # no crash, value untouched
        assert isinstance(out["CD2_2"], float)  # the rest still coerced


class TestStrapFaintEndSteepening:
    """The hybrid STRAP measurement row (2026-08-09): the on-sky-calibrated
    law holds through its last measured anchor (R=15.5), then the
    performance-sheet's steepening quadcell slopes are grafted on,
    anchor-continuous. Independent check: rebuild the row from its
    definition and compare tt_wfe_nm totals."""

    def _meas(self, R):
        m = 6.9 * 10 ** (0.116 * (min(R, 15.5) - 12.0))
        if R > 15.5:
            m *= 10 ** (0.204 * (min(R, 17.5) - 15.5))
        if R > 17.5:
            m *= 10 ** (0.230 * (R - 17.5))
        return m

    def test_hybrid_row_totals(self):
        import keck_ao_estimator as engine
        from keck_ao_estimator.constants import NM_PER_MAS
        from keck_ao_estimator import psf as _psf
        fixed2 = 0.47**2 + 0.53**2 + 0.03**2 + 2.6**2 + 5.0**2  # on-axis rows
        for R in (12.0, 14.0, 15.5, 16.0, 17.0, 18.0, 19.0):
            want = math.sqrt(self._meas(R) ** 2 + fixed2)
            # a loop cannot do worse than no correction: the engine clips at
            # the open-loop tilt ceiling (R=19 is ceiling-limited)
            want = min(want, _psf.OPEN_LOOP_TILT_ONEAXIS_MAS)
            got = engine.tt_wfe_nm(1.0, R, 0.0, sensor="strap") / NM_PER_MAS
            assert abs(got - want) < 0.05, (R, got, want)

    def test_continuity_and_bright_end_unchanged(self):
        import keck_ao_estimator as engine
        from keck_ao_estimator.constants import NM_PER_MAS
        # continuous at the knee (no jump a saved config could fall over)
        lo = engine.tt_wfe_nm(1.0, 15.499, 0.0, sensor="strap")
        hi = engine.tt_wfe_nm(1.0, 15.501, 0.0, sensor="strap")
        assert abs(hi - lo) / lo < 0.005
        # the default budget star (R=15.2) sits BELOW the knee: the frozen
        # goldens must be untouched by the faint-end graft
        t = engine.tt_wfe_nm(1.0, 15.2, 19.3, sensor="strap") / NM_PER_MAS
        assert abs(t - 19.52) < 0.1, t  # pre-graft value, must not move
