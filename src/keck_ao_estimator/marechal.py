"""The Marechal approximation -- the single formula every Strehl estimate in
this tool ultimately reduces to."""
import numpy as np

from .constants import LAMBDA_K_NM


def marechal_strehl(sigma_nm, lam_nm=LAMBDA_K_NM):
    """K-band Strehl from an RMS wavefront error (nm) via the Marechal approx."""
    return np.exp(-(2.0 * np.pi * sigma_nm / lam_nm) ** 2)


TILT_LAWS = ("marechal", "hardy")


def tilt_strehl(sigma_nm, lam_nm=LAMBDA_K_NM, law="marechal"):
    """Strehl factor of the residual TIP-TILT alone, from its equivalent
    wavefront error sigma_nm (the budget's one-axis mas x NM_PER_MAS).

    law="marechal" -- exp(-x^2), x = 2 pi sigma / lam: what every Strehl
    in this tool uses, and the form the tip-tilt sensor rows were
    calibrated through.
    law="hardy" -- 1 / (1 + x^2): the long-exposure form for a Gaussian
    image jitter (KAON 1303 eq. 49, closely Hardy 1998 eq. 4.62), which
    KAON 1303 and KAON 1318 use for their sky-coverage curves. Equal to Marechal to
    first order in x^2, with a much heavier tail at large jitter.
    """
    if law not in TILT_LAWS:
        raise ValueError(f"tilt law must be one of {TILT_LAWS}, not {law!r}")
    if law == "hardy":
        return 1.0 / (1.0 + (2.0 * np.pi * sigma_nm / lam_nm) ** 2)
    return marechal_strehl(sigma_nm, lam_nm)
