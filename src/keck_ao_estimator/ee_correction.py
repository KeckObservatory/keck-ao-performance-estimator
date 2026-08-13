"""Growth-curve (encircled-energy) aperture correction for Strehl.

Small-aperture SR measurements are inflated: the aperture misses
halo flux that the DL reference does not lose at the same rate (and
a close-in annulus sits ON the halo, double-counting). Matched-
settings tests (2026-07-25, the summit-vs-GUI side-by-side on the
GC field) showed the two tools agree exactly and the entire
historical cross-tool discrepancy is this convention: r = 0.3"
photometry read x1.6-3.3 above the 1" convention on the same stars,
FWHM identical.

Model: PSF ~ SR*DL + (1-SR)*halo, so EE(r) ~ SR*EE_DL(r) +
(1-SR)*EE_halo(r), giving a ONE-PARAMETER correction family in
h = EE_halo(r)/EE_DL(r):

    SR_true = SR_small * h / (1 - SR_small * (1 - h))

h is calibrated per field from stars measured in BOTH conventions
(small auto aperture AND the full radius) where the full-radius
measurement is clean. Parametrizing by each star's own SR -- not by
field position -- absorbs anisoplanatic halo variation to first
order: validated on 358 two-convention pairs (M79 2026-01-31),
mean bias +0.171 -> -0.005 corrected, h drifting only 0.42 -> 0.51
from on-axis to 30"+ off-axis (~0.01-0.02 leakage into corrected
block means). Calibrate h PER FIELD; never transfer between fields.
"""
import numpy as np

__all__ = ["ee_correct", "ee_expected_small", "ee_calibrate_h",
           "ee_calibrate_h_or_fallback", "ee_fallback_band",
           "EE_H_FALLBACK", "EE_H_OBSERVED"]

EE_H_MIN_PAIRS = 5          # fewest usable pairs to fit h
EE_H_GRID = np.linspace(0.05, 1.0, 96)

# --- fallback h for fields that cannot support their own fit --------
# KAON review comment #17. Eduardo: "could we use a fallback h value
# for fields with insufficient stars to calculate it? With a warning
# of course."
#
# The value is the MEDIAN of the campaign's per-field ledger, not the
# mean -- five fields is too few for a mean to be robust to one odd
# field, and the median is the value that minimises the worst-case
# error over the observed set:
#
#     0131 M79 0.41 | 0304 M3 0.53 (232 pairs) | 0528 0.61 (84)
#     Arches   0.40 (20 pairs)     | GC-0530 0.50 (21)
#
# THIS IS A LAST RESORT. The per-field rule exists because h varies by
# half its own value across fields; a fallback trades a refusal for a
# quantified bias. Never use it when a field CAN be calibrated, and
# never present a fallback-corrected SR without the warning string.
EE_H_FALLBACK = 0.50
EE_H_OBSERVED = (0.40, 0.61)
EE_H_LEDGER = {"0131 M79": 0.41, "0304 M3": 0.53, "0528": 0.61,
               "Arches": 0.40, "GC-0530": 0.50}


def ee_correct(sr_small, h):
    """True (full-aperture-convention) SR from a small-aperture SR."""
    sr_small = np.asarray(sr_small, dtype=float)
    return sr_small * h / (1.0 - sr_small * (1.0 - h))


def ee_expected_small(sr_true, h):
    """Model inverse: the small-aperture SR a given true SR yields."""
    sr_true = np.asarray(sr_true, dtype=float)
    return sr_true / (sr_true + (1.0 - sr_true) * h)


def ee_calibrate_h(pairs):
    """Fit h from (sr_small, sr_full) calibration pairs.

    pairs: iterable of (small-aperture SR, full-radius SR) for the
    SAME star, full-radius side clean (uncrowded). Returns (h, rms).
    Raises ValueError with a plain-language reason when the field
    cannot support a fit (too few clean pairs)."""
    pairs = [(float(a), float(b)) for a, b in pairs
             if 0.0 < a < 1.0 and 0.0 < b < 1.0]
    if len(pairs) < EE_H_MIN_PAIRS:
        raise ValueError(
            f"EE calibration needs >= {EE_H_MIN_PAIRS} clean "
            f"two-aperture pairs; got {len(pairs)}")
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    best_h, best_rms = 1.0, np.inf
    for h in EE_H_GRID:
        rms = float(np.sqrt(np.mean((ee_correct(a, h) - b) ** 2)))
        if rms < best_rms:
            best_h, best_rms = float(h), rms
    return best_h, best_rms


def ee_fallback_band(sr_small, h=EE_H_FALLBACK, observed=EE_H_OBSERVED):
    """SR interval implied by h's field-to-field spread, at sr_small.

    The point of the fallback is that its error is KNOWN, so quote it.
    Returns (sr_lo, sr_nominal, sr_hi) from the observed h range."""
    lo, hi = observed
    return (float(ee_correct(sr_small, lo)),
            float(ee_correct(sr_small, h)),
            float(ee_correct(sr_small, hi)))


def ee_calibrate_h_or_fallback(pairs, sr_typical=None):
    """Like `ee_calibrate_h`, but falls back instead of refusing.

    Returns (h, rms, warning). `rms` is NaN and `warning` is a
    non-empty string whenever the fallback was used; `warning` is ""
    when a real per-field fit succeeded, so callers can simply do
    `if warning: log(warning)`.

    The default path is untouched: `ee_calibrate_h` still raises, so
    nothing that does not opt in changes behaviour.
    """
    try:
        h, rms = ee_calibrate_h(pairs)
        return h, rms, ""
    except ValueError as exc:
        n = sum(1 for a, b in pairs
                if 0.0 < float(a) < 1.0 and 0.0 < float(b) < 1.0)
        lo, hi = EE_H_OBSERVED
        msg = (f"EE FALLBACK: using campaign-median h = {EE_H_FALLBACK:.2f} "
               f"because this field has {n} clean two-aperture pair(s), "
               f"below the {EE_H_MIN_PAIRS} needed to fit its own "
               f"({exc}). Observed per-field h spans {lo:.2f}-{hi:.2f} "
               f"across {len(EE_H_LEDGER)} campaign fields, so the "
               f"corrected SR carries that spread as a SYSTEMATIC, not "
               f"a statistical, uncertainty.")
        if sr_typical is not None:
            a, b, c = ee_fallback_band(sr_typical)
            msg += (f" At SR_small = {float(sr_typical):.3f} that is "
                    f"SR_true = {b:.3f} (+{c - b:.3f} / -{b - a:.3f}).")
        msg += " Calibrate per field wherever possible."
        return EE_H_FALLBACK, float("nan"), msg
