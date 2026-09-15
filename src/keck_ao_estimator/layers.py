"""Turbulence by layer on the tomographic reconstructor's altitude grid.

The K1 LTAO reconstructor carries a STATIC prior on 7 layers -- the ground
(0 km) plus the six MASS altitudes 0.5/1/2/4/8/16 km -- with fixed
turbulence fractions (budget.RECON_PRIOR_FRAC, KAON 1542 sect. 3.4 (2)). This
module is the small, pure conversion layer between a per-layer strength
vector on that grid and the two seeing numbers the rest of the estimator
runs on (total = all 7 layers, free-atm = the 6 aloft layers):

  * layer strengths are integrated Cn2*dh per layer, J_i [m^1/3], at 500 nm,
    zenith -- the same quantity the MASS bins carry (atmosphere.py);
  * a layer's "seeing-equivalent" is the seeing that layer alone would give,
    eps_i = seeing(J_i); strengths add linearly in J, so seeing combines as
    eps = (sum eps_i^(5/3))^(3/5), never as a plain sum.

The prior carries fractions only (no absolute r0), so "set the turbulence to
the reconstructor" needs a scale: recon_prior_layers() distributes the
integrated turbulence of a chosen TOTAL seeing across the 7 layers in the
prior's proportions; the free-atm seeing then follows from the prior's aloft
share (sum of fractions 1..6 = 0.5443 -> eps_fa = eps_tot * 0.5443^(3/5)).
"""
import numpy as np

from .constants import MASS_HEIGHTS_M
from .atmosphere import integrated_cn2_to_seeing, seeing_to_integrated_cn2
from .budget import RECON_PRIOR_FRAC

# reconstructor altitude grid: ground + the six MASS bins (metres above summit)
RECON_HEIGHTS_M = np.concatenate([[0.0], MASS_HEIGHTS_M])
N_RECON_LAYERS = RECON_HEIGHTS_M.size            # 7


def _as_layers(J):
    J = np.asarray(J, dtype=float).ravel()
    if J.size != N_RECON_LAYERS:
        raise ValueError(f"expected {N_RECON_LAYERS} layer strengths "
                         f"(ground + 6 MASS bins), got {J.size}")
    if not np.all(np.isfinite(J)) or np.any(J < 0):
        raise ValueError("layer strengths must be finite and >= 0")
    return J


def layer_seeing(J_i):
    """Seeing-equivalent [arcsec, 500 nm, zenith] of ONE layer's integrated
    Cn2*dh J_i [m^1/3]: the seeing that layer alone would produce."""
    return integrated_cn2_to_seeing(J_i)


def layer_from_seeing(eps_i):
    """Integrated Cn2*dh [m^1/3] of a layer whose seeing-equivalent is eps_i
    [arcsec]; eps_i <= 0 gives an empty layer (the inverse relation is
    singular at zero seeing)."""
    eps_i = float(eps_i)
    return seeing_to_integrated_cn2(eps_i) if eps_i > 0.0 else 0.0


def layers_seeing(J):
    """Total / free-atm / ground seeing [arcsec, 500 nm, zenith] implied by
    the 7 layer strengths J [m^1/3] (index 0 = ground, 1..6 = the MASS
    altitudes). Returns a dict with eps_tot, eps_fa, eps_ground, the aloft
    6-bin profile cn2_bins (what the budget's layer-mismatch and theta0
    take) and the fraction of the integrated turbulence sitting in each
    layer (all zero when every layer is empty)."""
    J = _as_layers(J)
    tot = float(J.sum())
    return dict(
        eps_tot=integrated_cn2_to_seeing(tot),
        eps_fa=integrated_cn2_to_seeing(J[1:].sum()),
        eps_ground=integrated_cn2_to_seeing(J[0]),
        cn2_bins=J[1:].copy(),
        frac=(J / tot if tot > 0.0 else np.zeros_like(J)),
    )


def fractions_to_layers(fractions, eps_tot_zenith):
    """Layer strengths [m^1/3] from turbulence FRACTIONS on the 7-layer grid
    (the reconstructor's own units: the share of the integrated turbulence
    in each layer) and the total seeing `eps_tot_zenith` [arcsec, 500 nm,
    zenith] that sets the absolute scale. The fractions are normalized to
    sum to 1 (a set summing to 0.93 is taken as proportions, so the total
    seeing is always eps_tot); all-zero fractions give empty layers."""
    f = np.asarray(fractions, dtype=float).ravel()
    if f.size != N_RECON_LAYERS:
        raise ValueError(f"expected {N_RECON_LAYERS} fractions, got {f.size}")
    if not np.all(np.isfinite(f)) or np.any(f < 0):
        raise ValueError("fractions must be finite and >= 0")
    eps_tot_zenith = float(eps_tot_zenith)
    tot = float(f.sum())
    if eps_tot_zenith <= 0.0 or tot <= 0.0:
        return np.zeros(N_RECON_LAYERS)
    return (f / tot) * seeing_to_integrated_cn2(eps_tot_zenith)


def recon_prior_layers(eps_tot_zenith):
    """The reconstructor's static prior as ABSOLUTE layer strengths [m^1/3]:
    the integrated turbulence of `eps_tot_zenith` [arcsec, 500 nm, zenith]
    distributed over the 7 layers in the prior's fractions. By construction
    layers_seeing() of the result reproduces eps_tot exactly and the aloft
    part has zero layer mismatch (budget.layer_mismatch == 0)."""
    return fractions_to_layers(RECON_PRIOR_FRAC, eps_tot_zenith)


def layers_from_seeing_pair(eps_tot_zenith, eps_fa_zenith, aloft_shape=None):
    """Build a 7-layer vector from the estimator's usual (total, free-atm)
    seeing pair: ground = total - free-atm (in J), aloft = free-atm spread
    over the 6 MASS altitudes in `aloft_shape` (fractions; default the
    reconstructor prior's aloft part). Free-atm is clamped to <= total, as
    everywhere else in the estimator."""
    eps_tot_zenith = float(eps_tot_zenith)
    eps_fa_zenith = min(float(eps_fa_zenith), eps_tot_zenith)
    J_tot = seeing_to_integrated_cn2(eps_tot_zenith) if eps_tot_zenith > 0 else 0.0
    J_fa = seeing_to_integrated_cn2(eps_fa_zenith) if eps_fa_zenith > 0 else 0.0
    shape = (RECON_PRIOR_FRAC[1:] / RECON_PRIOR_FRAC[1:].sum()
             if aloft_shape is None else np.asarray(aloft_shape, float))
    shape = shape / shape.sum()
    return np.concatenate([[max(J_tot - J_fa, 0.0)], J_fa * shape])
