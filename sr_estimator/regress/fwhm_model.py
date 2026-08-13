#!/usr/bin/env python3
"""Physics contract for the FWHM model (engine-level, no GUI).

Guards the 2026-07-10 fix: tip-tilt is IMAGE MOTION and must convolve (broaden)
the PSF core, not merely lower its peak. Before the fix the core was a pure Airy
and the FWHM sat frozen at 1.029 lambda/D regardless of conditions.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); os.chdir(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src"))
import inspect
import numpy as np
import keck_ao_estimator as e

LAM = e.LAMBDA_K_NM
V2K = (LAM / 500.0) ** -0.2
DL = 1.029 * (LAM * 1e-9) / e.TEL_DIAMETER_M * 206265e3


def close(a, b, tol):
    return abs(a - b) <= tol


def main():
    # --- limits ------------------------------------------------------------
    f = e.psf_fwhm_mas(1.0, 0.6, LAM, tt_nm=0.0)
    assert close(f, DL, 0.05), f"S=1,no tilt -> {f} (want 1.029 lam/D = {DL})"
    halo = 0.6 * (LAM / 500.0) ** -0.2 * 1000.0
    f0 = e.psf_fwhm_mas(0.0, 0.6, LAM, tt_nm=0.0)
    assert close(f0, halo, 1.0), f"S=0 -> {f0} (want seeing disk {halo})"
    assert np.isnan(e.psf_fwhm_mas(0.5, float("nan"), LAM))
    print(f"  [ok] limits: S=1 -> {f:.2f} mas = 1.029 lam/D ; S=0 -> {f0:.0f} mas")

    # --- tilt must BROADEN the core, monotonically -------------------------
    prev = None
    for tt_mas in (0, 5, 10, 20, 30):
        v = e.psf_fwhm_mas(1.0, 0.6, LAM, tt_nm=tt_mas * e.NM_PER_MAS)
        if prev is not None:
            assert v > prev + 1e-6, f"tilt {tt_mas} mas did not broaden the core"
        prev = v
    assert prev > 1.7 * DL, "30 mas jitter should nearly double the core"
    print(f"  [ok] tilt broadens core monotonically (30 mas -> {prev:.1f} mas)")

    # exact convolution sits just BELOW the naive quadrature (Airy(x)Gauss
    # is not Gauss(x)Gauss) -- a cheap guard against silently swapping models
    q = np.hypot(DL, 2.3548 * 20.0)
    v = e.psf_fwhm_mas(1.0, 0.6, LAM, tt_nm=20.0 * e.NM_PER_MAS)
    assert v < q and v > 0.9 * q, f"exact {v:.1f} vs quadrature {q:.1f}"
    print(f"  [ok] exact convolution {v:.1f} mas < quadrature approx {q:.1f} mas")

    # --- Eduardo's on-sky anchor: NGS ~52 mas at K-band seeing 0.6" --------
    eps500 = 0.60 / V2K
    s_tot = (eps500 / e.REF_TOTAL) ** (5.0 / 6.0)
    S = e.ngs_strehl(eps500, 8.0, "K2", LAM)
    tt = e.ngs_tt_nm(s_tot, 8.0, 0.0)
    anchor = e.psf_fwhm_mas(S, eps500, LAM, tt)
    assert close(anchor, 52.0, 0.2), f"anchor {anchor:.2f} != 52 mas"
    print(f"  [ok] on-sky anchor: NGS FWHM {anchor:.2f} mas @ eps_K=0.6\"")

    # NGS must degrade with seeing (the whole point of the tilt-servo row)
    fw = []
    for ek in (0.15, 0.30, 0.45, 0.60, 0.75, 0.90):
        e5 = ek / V2K
        st = (e5 / e.REF_TOTAL) ** (5.0 / 6.0)
        fw.append(e.psf_fwhm_mas(e.ngs_strehl(e5, 8.0, "K2", LAM), e5, LAM,
                                 e.ngs_tt_nm(st, 8.0, 0.0)))
    assert all(b > a for a, b in zip(fw[1:], fw[2:])), "NGS FWHM must grow"
    assert fw[-1] > fw[0] + 4.0, f"NGS barely degrades: {fw[0]:.1f}->{fw[-1]:.1f}"
    assert fw[-1] < 1.4 * fw[0], "NGS should degrade SLOWLY"
    print(f"  [ok] NGS degrades slowly with seeing: {fw[0]:.1f} -> {fw[-1]:.1f} mas")

    # --- the tilt-servo row must NEVER enter the Strehl budget -------------
    for fn in (e.tt_wfe_nm, e.lgs_budget_terms, e.lgs_strehl, e.ngs_strehl):
        src = inspect.getsource(fn)
        assert "TILT_SERVO" not in src and "tilt_servo" not in src, \
            f"NGS tilt-servo row leaked into {fn.__name__} -> would change Strehl"
    # the frozen 2004 sheet stays reproducible via the legacy sensor (and the
    # NGS path pins to it); the refined-STRAP default is the 2026-07
    # recalibration to the paired on-sky STRAP/TRICK data
    assert close(e.tt_wfe_nm(1.0, sensor="strap-legacy"), 163.6, 1.5), \
        e.tt_wfe_nm(1.0, sensor="strap-legacy")
    assert close(e.tt_wfe_nm(1.0), 244.4, 2.0), e.tt_wfe_nm(1.0)
    print("  [ok] tilt-servo row confined to the FWHM path; legacy TT anchor "
          "reproducible; refined-STRAP default in place")

    # --- simulated Gaussian-fit metric + the 20260701 on-sky validation ----
    # a Gaussian LSQ fit to a pure Airy reads slightly NARROW (no shoulder to
    # absorb): ~0.94 dl
    g1 = e.fwhm_gaussfit_mas(1.0, 0.6, LAM, 0.0)
    assert 0.90 * DL < g1 < 1.00 * DL, f"gaussfit(Airy) {g1} vs dl {DL}"
    # S=0: fit of the Moffat seeing disk reads within a few % of its width
    halo = 0.6 * (LAM / 500.0) ** -0.2 * 1000.0
    g0 = e.fwhm_gaussfit_mas(0.0, 0.6, LAM, 0.0)
    assert close(g0, halo, 0.05 * halo), f"gaussfit(S=0) {g0} vs halo {halo}"
    assert np.isnan(e.fwhm_gaussfit_mas(0.5, float("nan"), LAM))
    print(f"  [ok] gaussfit limits: Airy -> {g1:.1f} ({g1/DL:.2f} dl); "
          f"S=0 -> {g0:.0f} mas")

    # 20260701 00:35-01:36, K1, X~1.8 (32 OSIRIS frames), window medians:
    # AO Strehl tool (half-max) 62.8; OSIRIS QL (Gaussian fit) 76.7.
    S_w, tt_w, eps_w = 0.151, 19.8 * e.NM_PER_MAS, 0.89
    fit_w = e.FITTING_ERR["K1"] * (eps_w / e.REF_TOTAL) ** (5.0 / 6.0)
    hm = e.psf_fwhm_mas(S_w, eps_w, LAM, tt_w, fit_nm=fit_w, n_act=20)
    gf = e.fwhm_gaussfit_mas(S_w, eps_w, LAM, tt_w, fit_w, 20)
    assert abs(hm - 62.8) / 62.8 < 0.08, f"half-max {hm:.1f} vs tool A 62.8"
    assert abs(gf - 76.7) / 76.7 < 0.10, f"gaussfit {gf:.1f} vs OSIRIS 76.7"
    assert gf > hm + 3.0, "fit must read above half-max in poor conditions"
    print(f"  [ok] 20260701 validation: half-max {hm:.1f} (tool A 62.8, "
          f"{(hm/62.8-1)*100:+.0f}%); gaussfit {gf:.1f} (OSIRIS 76.7, "
          f"{(gf/76.7-1)*100:+.0f}%)")

    # --- third convention: free-background Gaussian fit (fwhm_gaussfit_sky_mas)
    # models the live interactive "AO Strehl tool" (MPFITPEAK default: Gaussian
    # + free constant), not OSIRISSTREHL_QL2 (no-background, that's
    # fwhm_gaussfit_mas). box_mas is a real caller parameter on BOTH fit modes
    # now (the real tool's box is hand-drawn, not fixed) -- default 300 kept
    # for back-compat with the two asserts above.
    gs1 = e.fwhm_gaussfit_sky_mas(1.0, 0.6, LAM, 0.0)
    assert 0.85 * DL < gs1 < 1.00 * DL, f"gaussfit-sky(Airy) {gs1} vs dl {DL}"
    assert np.isnan(e.fwhm_gaussfit_sky_mas(0.5, float("nan"), LAM))
    # a free background can only narrow (or match) the fit relative to the
    # no-background convention on the SAME PSF -- it has one more degree of
    # freedom to absorb flat halo/sky flux instead of forcing it into sigma
    gf1 = e.fwhm_gaussfit_mas(1.0, 0.6, LAM, 0.0)
    assert gs1 <= gf1 + 0.5, f"gaussfit-sky {gs1} should not read wider than gaussfit {gf1}"
    g0s = e.fwhm_gaussfit_sky_mas(0.0, 0.6, LAM, 0.0)
    g0f = e.fwhm_gaussfit_mas(0.0, 0.6, LAM, 0.0)
    assert g0s < g0f, f"free background should narrow the S=0 seeing-disk fit ({g0s} vs {g0f})"
    print(f"  [ok] gaussfit-sky limits: Airy -> {gs1:.1f} ({gs1/DL:.2f} dl, "
          f"<= no-bg {gf1:.1f}); S=0 -> {g0s:.0f} mas (< no-bg {g0f:.0f})")

    # box_mas is now a real, adjustable parameter on both fit modes (the real
    # AO Strehl tool's box is hand-drawn; OSIRISSTREHL_QL2's own auto box is
    # ~30.7 mas at K -- see psf.py's "REAL MEASUREMENT TOOLS" comment). A
    # tighter box must not diverge/blow up, and should track the half-max
    # value more closely than the wide-box default (less halo/shoulder in view).
    hm_w = e.psf_fwhm_mas(0.151, 0.89, LAM, 19.8 * e.NM_PER_MAS,
                          fit_nm=e.FITTING_ERR["K1"] * (0.89 / e.REF_TOTAL) ** (5.0 / 6.0),
                          n_act=20)
    fit_w = e.FITTING_ERR["K1"] * (0.89 / e.REF_TOTAL) ** (5.0 / 6.0)
    gf_tight = e.fwhm_gaussfit_mas(0.151, 0.89, LAM, 19.8 * e.NM_PER_MAS, fit_w, 20,
                                   box_mas=31.0)
    gf_wide = e.fwhm_gaussfit_mas(0.151, 0.89, LAM, 19.8 * e.NM_PER_MAS, fit_w, 20,
                                  box_mas=300.0)
    assert np.isfinite(gf_tight) and gf_tight > 0
    assert abs(gf_tight - hm_w) < abs(gf_wide - hm_w), \
        f"tight box ({gf_tight:.1f}) should track half-max ({hm_w:.1f}) " \
        f"more closely than the wide-box default ({gf_wide:.1f})"
    print(f"  [ok] box_mas is explorable: tight (31mas) -> {gf_tight:.1f} mas "
          f"(closer to half-max {hm_w:.1f} than wide-box {gf_wide:.1f})")

    # the NGS anchor must survive the 3-component PSF
    e5 = 0.60 / V2K
    st = (e5 / e.REF_TOTAL) ** (5.0 / 6.0)
    hm3 = e.psf_fwhm_mas(e.ngs_strehl(e5, 8.0, "K2", LAM), e5, LAM,
                         e.ngs_tt_nm(st, 8.0, 0.0),
                         fit_nm=e.FITTING_ERR["K2"] * st,
                         n_act=e.DM_ACTUATORS_ACROSS["K2"])
    assert close(hm3, 52.0, 0.4), f"3-comp NGS anchor drifted: {hm3:.2f}"
    print(f"  [ok] NGS anchor holds with 3-comp PSF: {hm3:.2f} mas")

    # --- LGS/LTAO tilt comes from the budget, and now tracks seeing --------
    lo = e.psf_fwhm_mas(0.30, 0.40, LAM, tt_nm=11.0 * e.NM_PER_MAS)
    hi = e.psf_fwhm_mas(0.10, 1.20, LAM, tt_nm=28.0 * e.NM_PER_MAS)
    assert hi > lo + 15.0, f"LGS FWHM should respond strongly ({lo:.1f}->{hi:.1f})"
    print(f"  [ok] LGS/LTAO FWHM responds to conditions ({lo:.1f} -> {hi:.1f} mas)")


if __name__ == "__main__":
    main()
    print("  [ok] FWHM physics contract holds")
