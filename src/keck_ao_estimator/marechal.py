"""The Marechal approximation -- the single formula every Strehl estimate in
this tool ultimately reduces to."""
import numpy as np

from .constants import LAMBDA_K_NM


def marechal_strehl(sigma_nm, lam_nm=LAMBDA_K_NM):
    """K-band Strehl from an RMS wavefront error (nm) via the Marechal approx."""
    return np.exp(-(2.0 * np.pi * sigma_nm / lam_nm) ** 2)
