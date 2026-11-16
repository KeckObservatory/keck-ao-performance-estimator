"""Remove the OSIRIS imager's row and column offsets from a bench frame.

Port of osiris_destripe.pro (keckAOIDL k1, asurendran 2026-11-13, PR #966).
The H2RG's row and column offsets change from frame to frame, so a
background or dark frame does not remove them. In the annulus they bias the
sky, and the Strehl ratio (SR) with it: SR > 1 on bench frames. The K1 tool's
find_strehl.pro applies this to bench (daytime) frames only, after the
background subtraction and fix_image. The measurement here does the same
(osiris_reduce with destripe=True).

The algorithm, as IDL does it:

1. Find the star: the maximum of the image smoothed by a 5 px box (edge
   pixels replicated, IDL SMOOTH /EDGE_TRUNCATE).
2. Subtract each column's median, taken over the rows more than `halfbox`
   px from the star.
3. Subtract each row's median of the column-corrected image, taken over
   the columns more than `halfbox` px from the star.

IDL's MEDIAN (no /EVEN) returns the upper of the two middle values for an
even count; `_idl_median` does the same. The K1 tool runs this on the full
2048 x 2048 frame.
"""
import numpy as np
from scipy.ndimage import uniform_filter

OSIRIS_DESTRIPE_HALFBOX_PX = 250        # 2.5" at 9.95 mas/px


def _idl_median(a, axis):
    """IDL MEDIAN along `axis`: the middle value, the upper middle one for an even count."""
    n = a.shape[axis]
    return np.take(np.partition(a, n // 2, axis=axis), n // 2, axis=axis)


def destripe_star_position(image):
    """(x0, y0) of the star as osiris_destripe.pro finds it: the brightest pixel of the 5 px box-smoothed image."""
    sm = uniform_filter(np.asarray(image, dtype=float), size=5, mode="nearest")
    y0, x0 = np.unravel_index(int(np.argmax(sm)), sm.shape)
    return int(x0), int(y0)


def osiris_destripe(image, x0=None, y0=None, halfbox=OSIRIS_DESTRIPE_HALFBOX_PX):
    """`image` with the column and then the row offsets removed (osiris_destripe.pro).

    :param image: 2-D frame, background-subtracted or not.
    :param x0: star column; with `y0`, found by destripe_star_position when either is None.
    :param y0: star row.
    :param int halfbox: half size (px) of the box around the star left out of the medians.
    :returns: float array, same shape.
    """
    out = np.array(image, dtype=float)
    if x0 is None or y0 is None:
        x0, y0 = destripe_star_position(out)
    ny, nx = out.shape
    rows = np.abs(np.arange(ny) - y0) > halfbox
    cols = np.abs(np.arange(nx) - x0) > halfbox
    if rows.any():
        out -= _idl_median(out[rows, :], axis=0)[None, :]
    if cols.any():
        out -= _idl_median(out[:, cols], axis=1)[:, None]
    return out
