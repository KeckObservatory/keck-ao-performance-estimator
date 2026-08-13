"""Matplotlib figure builders: the main Strehl timeline, the error-budget
terms figure, and the FWHM timeline/overlay. Each render_* function builds
and returns a Figure without saving it -- the CLI saves it, the GUI draws it
straight onto its canvas. All are deterministic (regression-harness verified
byte-identical / pixel-close to the pre-extraction output).
"""
from datetime import timedelta

import numpy as np
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle

from . import budget
from .budget import DEF_LGS_OFFSET, STATIC_TEL, static_subtotal
from .constants import LAMBDA_K_NM, POINTING_LIMITS, TEL_DIAMETER_M
from .geometry import compute_airmass_curve, pointing_state


class _SplitSwatch:
    """Proxy artist for a legend entry drawn as a two-color split rectangle
    (left half / right half), used to show a budget slot that changes color
    by mode -- e.g. focal aniso (red, single-beacon) becoming tomography
    (teal, LTAO)."""
    def __init__(self, left_color, right_color):
        self.left_color = left_color
        self.right_color = right_color


class _SplitSwatchHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        left = Rectangle([xdescent, ydescent], width / 2.0, height,
                         facecolor=orig_handle.left_color, edgecolor="none",
                         transform=trans)
        right = Rectangle([xdescent + width / 2.0, ydescent], width / 2.0,
                          height, facecolor=orig_handle.right_color,
                          edgecolor="none", transform=trans)
        return [left, right]


def gapline(ts, vs, secs, tol):
    """Insert a NaN break wherever consecutive samples are further apart
    than the match tolerance, so a connecting line never bridges a real
    MASS outage. Shared by render_main_figure and render_terms_figure.
    (render_main_figure also defines an identical local copy.)"""
    out_t, out_v = [], []
    for k in range(len(ts)):
        if k > 0 and secs[k] - secs[k - 1] > tol:
            out_t.append(ts[k]); out_v.append(np.nan)
        out_t.append(ts[k]); out_v.append(vs[k])
    return out_t, out_v


def render_main_figure(args, prep, res, window_label_margin=False):
    """Build and return the main timeline Figure (does NOT save it). The
    CLI saves exactly this Figure; the GUI draws it straight onto its
    canvas. Deterministic -- the regression harness verifies the saved PNG
    is byte-identical to the pre-extraction output.

    window_label_margin: place the observing-window label ABOVE the panels
    (in the margin) instead of inside the data. Default False keeps the exact
    frozen CLI layout; the GUI passes True so the label never covers points."""
    _ltao_bw_fac = prep._ltao_bw_fac
    baseline_zen_factor = prep.baseline_zen_factor
    dimm_dt = prep.dimm_dt
    dimm_sec = prep.dimm_sec
    dimm_see = prep.dimm_see
    fixed_zen_factor = prep.fixed_zen_factor
    in_any_window = prep.in_any_window
    lam_label = prep.lam_label
    lam_nm = prep.lam_nm
    mass_dt = prep.mass_dt
    mass_sec = prep.mass_sec
    mass_see = prep.mass_see
    night_date = prep.night_date
    out_path = prep.out_path
    profiles = prep.profiles
    show_target = prep.show_target
    tomography_on = prep.tomography_on
    ut_stamp = prep.ut_stamp
    windows = prep.windows
    zen_factor_by_time = prep.zen_factor_by_time
    col_airmass = res.col_airmass
    col_ang1 = res.col_ang1
    col_d0 = res.col_d0
    col_dimm = res.col_dimm
    col_mass = res.col_mass
    col_mm = res.col_mm
    col_ngs_th0 = res.col_ngs_th0
    col_r0_cm = res.col_r0_cm
    col_terms = res.col_terms
    col_theta0 = res.col_theta0
    col_tt10 = res.col_tt10
    col_zf = res.col_zf
    n_fb = res.n_fb
    ngs_bright = res.ngs_bright
    ngs_faint = res.ngs_faint
    ngs_fb = res.ngs_fb
    p_airmass = res.p_airmass
    p_dimm_in = res.p_dimm_in
    p_secs = res.p_secs
    p_times = res.p_times
    p_zf = res.p_zf
    sr_ltao = res.sr_ltao
    sr_single = res.sr_single
    th0_assumed = res.th0_assumed
    times = res.times
    # =========================================================================
    #  PLOT
    # =========================================================================
    # colors
    C_NGS, C_LGS, C_LTAO = "#6A3D9A", "#1B3A6B", "#138086"
    C_DIMM, C_MASS       = "#C0392B", "#8B1A4A"
    C_GREEN, C_RED       = "#2E8B57", "#C0392B"
    C_GC                 = "#6A3D9A"

    # choose panels. The LGS middle section depends on telescope + tomography:
    #   * tomography ON  -> show the LTAO panel (with a faint single-beacon trace
    #                       overlaid for reference). This is the K1 default and
    #                       keeps K1 at three panels: NGS, LTAO(+faint SLGS),
    #                       seeing -- no separate single-beacon panel.
    #   * tomography OFF -> show the single-beacon LGS panel only (K2 default).
    # When tomography is ON we do NOT also draw a standalone single-beacon panel;
    # the faint reference trace inside the LTAO panel serves that role.
    panels = ["ngs"]
    if tomography_on:
        panels.append("ltao")
    else:
        panels.append("single")
    panels.append("seeing")
    panels.append("profile")           # theta0 (left) + d0 (right)
    n = len(panels)

    fig = Figure(figsize=(13, 3.1 * n + 0.5))
    axes = fig.subplots(n, 1, sharex=True,
                        gridspec_kw={"hspace": 0.10})
    axd = dict(zip(panels, axes))

    lgs_label = "and LTAO" if tomography_on else "single-beacon LGS"
    title_modes = "NGS, " + lgs_label
    axes[0].set_title(
        f"{title_modes} Strehl ({lam_label}) vs. seeing — "
        f"{night_date:%Y-%m-%d} ({args.telescope})",
        fontweight="bold", fontsize=13, pad=24)

    # ---- zenith-angle banner (always shown so the ZA convention is on-plot) --
    if show_target:
        if args.zenith_angle:
            _Xb = 1.0 / np.cos(np.radians(min(abs(args.zenith_angle), 85.0)))
            za_banner = (f"Zenith angle: {args.target_name} line-of-sight inside "
                         f"observing window(s); ZA = {args.zenith_angle:g}° "
                         f"(airmass {_Xb:.2f}) elsewhere")
        else:
            za_banner = (f"Zenith angle: {args.target_name} line-of-sight inside "
                         f"observing window(s); ZA = 0° (zenith) elsewhere")
    elif args.zenith_angle:
        _X = 1.0 / np.cos(np.radians(min(abs(args.zenith_angle), 85.0)))
        za_banner = (f"Zenith angle: ZA = {args.zenith_angle:g}° "
                     f"(airmass {_X:.2f}, seeing ×{fixed_zen_factor:.2f}) "
                     f"applied all night")
    else:
        za_banner = "Zenith angle: ZA = 0° (zenith) — no line-of-sight projection"
    # place in the padding gap just under the title, centered
    axes[0].text(0.5, 1.005, za_banner, transform=axes[0].transAxes,
                 ha="center", va="bottom", fontsize=9.5, style="italic",
                 color="#444444")

    # ---- gap-aware line helper ----------------------------------------------
    #  Connecting lines on the PROFILE timebase must not bridge real MASS
    #  outages: insert a NaN break wherever consecutive profile samples are
    #  further apart than the match tolerance (same rule as the MASS trace in
    #  the seeing panel), so a sparse MASS night reads as sparse in every
    #  panel. Markers are drawn separately, so isolated samples stay visible.
    def gapline(ts, vs, secs, tol):
        out_t, out_v = [], []
        for k in range(len(ts)):
            if k > 0 and secs[k] - secs[k - 1] > tol:
                out_t.append(ts[k]); out_v.append(np.nan)
            out_t.append(ts[k]); out_v.append(vs[k])
        return out_t, out_v

    # ---- Panel: NGS ---------------------------------------------------------
    ax = axd["ngs"]
    ax.plot(times, ngs_bright, "-", color=C_NGS, lw=1.0, alpha=0.35, zorder=2)
    if ngs_fb.any():
        # off-axis run mixing real and assumed theta0: circles = MASS theta0,
        # triangles = assumed fallback, so the provenance is visible per point
        real = ~ngs_fb
        ax.scatter(np.asarray(times)[real], ngs_bright[real],
                   c=ngs_bright[real], cmap="RdYlGn", s=26, marker="o",
                   edgecolor="white", linewidth=0.4, vmin=0, vmax=0.75,
                   zorder=3, label="θ₀ from MASS")
        ax.scatter(np.asarray(times)[ngs_fb], ngs_bright[ngs_fb],
                   c=ngs_bright[ngs_fb], cmap="RdYlGn", s=34, marker="^",
                   edgecolor="0.35", linewidth=0.6, vmin=0, vmax=0.75,
                   zorder=3, label=f"θ₀ assumed ({args.assumed_theta0:g}″ K, zenith)")
    else:
        ax.scatter(times, ngs_bright, c=ngs_bright, cmap="RdYlGn", s=26,
                   edgecolor="white", linewidth=0.4, vmin=0, vmax=0.75, zorder=3)
    ax.plot(times, ngs_faint, "--", color=C_NGS, lw=1.3, alpha=0.7,
            label=f"NGS R={args.ngs_faint:g} (mid)")
    ax.axhline(ngs_bright.mean(), color=C_NGS, ls=":", lw=1.0, alpha=0.6,
               label=f"R={args.ngs_bright:g} mean = {ngs_bright.mean():.2f}")
    ax.set_ylabel(f"NGS Strehl\n({lam_label.split('(')[0].strip()})", fontsize=12)
    _ngs_hi = np.nanmax(np.concatenate([ngs_bright, ngs_faint])) \
              if np.isfinite(ngs_bright).any() else 0.0
    ax.set_ylim(0, max(0.8, min(1.0, np.ceil(_ngs_hi * 10 + 0.5) / 10)))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", ncol=2, fontsize=7.5, framealpha=0.6, labelspacing=0.3, handlelength=1.4, borderpad=0.3, handletextpad=0.4)
    ngs_note = (f"natural guide star (R={args.ngs_bright:g} points, "
                f"R={args.ngs_faint:g} dashed) — follows TOTAL seeing")
    if args.telescope == "K1":
        ngs_note += "  [K1: RTC+OCAM fit, −0.05 quadcell, steepened seeing]"
    if abs(lam_nm - LAMBDA_K_NM) >= 1.0:
        ngs_note += f"  [{lam_label.split('(')[0].strip()}: extrapolated from K-band fit]"
    if float(args.ngs_offset or 0.0) > 0.0:
        if n_fb > 0:
            ngs_note += (f"  [NGS {args.ngs_offset:g}″ off-axis: ×exp(−(θ/θ₀)^(5/3)); "
                         f"triangles = assumed θ₀ ({n_fb}/{len(times)} samples, no MASS)]")
        elif th0_assumed is None:
            ngs_note += (f"  [NGS {args.ngs_offset:g}″ off-axis: ×exp(−(θ/θ₀)^(5/3)), "
                         f"θ₀ from MASS profile — fallback disabled, gaps where no MASS]")
        else:
            ngs_note += (f"  [NGS {args.ngs_offset:g}″ off-axis: ×exp(−(θ/θ₀)^(5/3)), "
                         f"θ₀ from MASS profile]")
    ax.text(0.012, 0.93, ngs_note,
            transform=ax.transAxes, fontsize=9, color=C_NGS, va="top")

    # ---- Panel: single-beacon LGS (only when tomography is OFF) -------------
    if "single" in axd:
        ax = axd["single"]
        g_t, g_v = gapline(p_times, sr_single, p_secs, args.match_tol)
        ax.plot(g_t, g_v, "-", color=C_LGS, lw=1.0, alpha=0.35, zorder=2)
        ax.scatter(p_times, sr_single, c=sr_single, cmap="RdYlGn", s=26,
                   edgecolor="white", linewidth=0.4, vmin=0, vmax=0.55, zorder=3)
        if np.isfinite(sr_single).any():
            _m = np.nanmean(sr_single)
            ax.axhline(_m, color=C_LGS, ls=":", lw=1.0, alpha=0.6,
                       label=f"mean = {_m:.2f}")
        ax.set_ylabel(f"single-beacon LGS\n{lam_label.split('(')[0].strip()} Strehl", fontsize=12)
        _sr_hi = (np.nanmax(np.concatenate([sr_single, sr_ltao]))
                  if (len(sr_single) and np.isfinite(sr_single).any()) else 0.0)
        ax.set_ylim(0, max(0.6, min(1.0, np.ceil(_sr_hi * 10 + 0.5) / 10)))
        ax.grid(alpha=0.3)
        # when a target is shown, the airmass legend takes the upper-right of this
        # panel, so move this panel's own legend to lower-left to avoid overlap
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="lower left" if show_target else "upper right", fontsize=7.5, framealpha=0.6, labelspacing=0.3, handlelength=1.4, borderpad=0.3, handletextpad=0.4)
        note = "single laser — follows FREE-ATMOSPHERE seeing"
        if args.telescope == "K1":
            note += "  [K1 DM saturates in poor seeing → more seeing-sensitive]"
        note += "\nLGS shown only where MASS data available"
        ax.text(0.012, 0.93, note,
                transform=ax.transAxes, fontsize=9, color=C_LGS, va="top")

    # ---- Panel: LTAO with faint single-beacon overlay (tomography ON) -------
    if "ltao" in axd:
        ax = axd["ltao"]
        g_t, g_v = gapline(p_times, sr_ltao, p_secs, args.match_tol)
        ax.plot(g_t, g_v, "-", color=C_LTAO, lw=1.0, alpha=0.35, zorder=2)
        ax.scatter(p_times, sr_ltao, c=sr_ltao, cmap="RdYlGn", s=26,
                   edgecolor="white", linewidth=0.4, vmin=0, vmax=0.55, zorder=3)
        if np.isfinite(sr_ltao).any():
            _m = np.nanmean(sr_ltao)
            ax.axhline(_m, color=C_LTAO, ls=":", lw=1.0, alpha=0.7,
                       label=f"LTAO mean = {_m:.2f}")
        # faint single-beacon trace overlaid for reference
        g_t, g_v = gapline(p_times, sr_single, p_secs, args.match_tol)
        ax.plot(g_t, g_v, "-", color=C_LGS, lw=0.9, alpha=0.30,
                zorder=1, label="single-beacon LGS (faint, ref)")
        ax.set_ylabel(f"LTAO (4-beacon)\n{lam_label.split('(')[0].strip()} Strehl", fontsize=12)
        _sr_hi = (np.nanmax(np.concatenate([sr_single, sr_ltao]))
                  if (len(sr_single) and np.isfinite(sr_single).any()) else 0.0)
        ax.set_ylim(0, max(0.6, min(1.0, np.ceil(_sr_hi * 10 + 0.5) / 10)))
        ax.grid(alpha=0.3)
        # airmass legend takes upper-right when a target is shown; avoid overlap
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="lower left" if show_target else "upper right", fontsize=7.5, framealpha=0.6, labelspacing=0.3, handlelength=1.4, borderpad=0.3, handletextpad=0.4)
        note = "tomography — high layer measured & corrected"
        if args.telescope == "K1":
            note += "  [K1 DM saturates in poor seeing → more seeing-sensitive]"
        note += "\nLGS/LTAO shown only where MASS data available"
        ax.text(0.012, 0.93, note,
                transform=ax.transAxes, fontsize=9, color=C_LTAO, va="top")

    # ---- Panel: seeing ------------------------------------------------------
    ax = axd["seeing"]
    dd = sorted(zip(dimm_dt, dimm_see))
    ax.plot([x[0] for x in dd], [x[1] for x in dd], "-", color=C_DIMM, lw=1.4,
            alpha=0.9, label="DIMM (total seeing)")
    if len(mass_dt):
        #  MASS is plotted gap-aware: consecutive samples are only connected
        #  (and the area under them filled) when they are closer together than
        #  the DIMM<->MASS match tolerance. Interpolating a line straight
        #  across a big gap -- or past the last sample -- would imply
        #  continuous MASS coverage the instrument didn't deliver, and would
        #  visually contradict the LGS/LTAO/theta0 panels, which honestly show
        #  isolated points when MASS sampling is sparse. Every real sample
        #  also gets a marker so lone samples (no neighbor within tolerance)
        #  remain visible instead of vanishing with no line to carry them.
        md = sorted(zip(mass_dt, mass_sec, mass_see), key=lambda x: x[1])
        m_t   = [x[0] for x in md]
        m_s   = [x[1] for x in md]
        m_v   = [x[2] for x in md]
        # insert NaN breaks where the gap between samples exceeds the tolerance
        plot_t, plot_v = [m_t[0]], [m_v[0]]
        for i in range(1, len(m_t)):
            if m_s[i] - m_s[i - 1] > args.match_tol:
                plot_t.append(m_t[i]); plot_v.append(np.nan)   # break line+fill
                plot_t.append(m_t[i]); plot_v.append(m_v[i])
            else:
                plot_t.append(m_t[i]); plot_v.append(m_v[i])
        ax.plot(plot_t, plot_v, "-", color=C_MASS, lw=1.4,
                alpha=0.8, label="MASS (free-atmosphere seeing)")
        ax.plot(m_t, m_v, "o", color=C_MASS, ms=3.0, alpha=0.8, zorder=3)
        ax.fill_between(plot_t, plot_v, color=C_MASS, alpha=0.10,
                        where=[np.isfinite(v) for v in plot_v])
    ax.set_ylabel("seeing @ 0.5 µm\n(arcsec)", fontsize=12)
    ax.set_ylim(0, 2.0); ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=7.5, framealpha=0.6, labelspacing=0.3, handlelength=1.4, borderpad=0.3, handletextpad=0.4)

    # r0 (Fried parameter at 500 nm) plotted as its own curve on the RIGHT axis,
    # computed from the DIMM total seeing: r0 = 0.98 * lambda / epsilon.
    # Normal (non-inverted) axis: larger r0 = better = higher up.
    C_R0 = "#2E8B57"
    dd_t = [x[0] for x in dd]
    dd_r0 = [0.98 * 500e-9 / (x[1] / 206265.0) * 100.0 for x in dd]   # cm
    axr = ax.twinx()
    axr.plot(dd_t, dd_r0, "-", color=C_R0, lw=1.2, alpha=0.8,
             label="r₀ @ 500 nm (from DIMM)")
    axr.set_ylabel("r₀ @ 500 nm  (cm)", fontsize=12, color=C_R0)
    axr.tick_params(axis="y", labelcolor=C_R0)
    axr.set_ylim(0, max(45, np.nanmax(dd_r0) * 1.1))
    axr.legend(loc="upper right", fontsize=7.5, framealpha=0.6, labelspacing=0.3,
               handlelength=1.4, borderpad=0.3, handletextpad=0.4)

    # ---- Panel: theta0 ------------------------------------------------------
    ax = axd["profile"]
    C_TH0 = "#1B3A6B"   # theta0 in the standard data blue (matches other panels)
    g_t, g_v = gapline(p_times, col_theta0, p_secs, args.match_tol)
    ax.plot(g_t, g_v, "-", color=C_TH0, lw=1.0, alpha=0.4, zorder=2)
    ax.scatter(p_times, col_theta0, s=18, color=C_TH0, edgecolor="white",
               linewidth=0.3, zorder=3, label="θ₀")
    ax.set_ylabel(f"θ₀  (arcsec, {lam_label.split('(')[0].strip()})",
                  fontsize=12)
    th_max = np.nanmax(col_theta0) if np.isfinite(col_theta0).any() else 4.0
    ax.set_ylim(0, th_max * 1.15)
    ax.grid(alpha=0.3)
    _los = "line-of-sight" if (show_target or args.zenith_angle) else "zenith"
    ax.text(0.012, 0.90,
            f"θ₀ at {lam_label.split('(')[0].strip()} (scales λ^1.2) "
            f"· from MASS Cₙ² profile · {_los}  "
            f"(d₀ in CSV)",
            transform=ax.transAxes, fontsize=8.5, color="#555555", va="top")
    ax.set_xlabel(f"HST (night of {night_date:%Y-%m-%d})", fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))

    # Clamp the x-axis to the actual data span (with a small pad) so that an
    # observing window or airmass point lying outside the data can never stretch
    # the axis into empty space. Windows outside the span simply clip.
    span_dts = [times.min(), times.max(), dimm_dt.min(), dimm_dt.max()]
    if len(mass_dt):
        span_dts += [mass_dt.min(), mass_dt.max()]
    data_lo, data_hi = min(span_dts), max(span_dts)
    pad = timedelta(minutes=20)
    for ax_ in axes:
        ax_.set_xlim(data_lo - pad, data_hi + pad)

    # ---- target overlay: airmass curve + observing-window boxes (opt-in) ----
    #  Entirely controlled by --target. When off: no airmass, no window boxes,
    #  no target annotation at all.
    if show_target:
        # the LGS panel that exists this run (ltao if tomography on, else single)
        lgs_key = "ltao" if "ltao" in axd else "single"

        # --- airmass overlay on the LGS panel's right axis ---
        host = axd[lgs_key]
        t_start = min(times.min(), dimm_dt.min())
        t_end   = max(times.max(), dimm_dt.max())
        n_grid  = int((t_end - t_start).total_seconds() // 240) + 1   # every 4 min
        grid    = [t_start + timedelta(minutes=4 * i) for i in range(n_grid)]
        airmass, elev, az = compute_airmass_curve(args.ra, args.dec, grid)

        # classify each grid point by this telescope's pointing limits
        states = [pointing_state(e, a, args.telescope)
                  for e, a in zip(elev, az)]
        # airmass arrays split by accessibility state:
        #   open      -> solid line (unvignetted)
        #   vignetted -> dashed/faded line (0-18 deg, observable but degraded)
        #   blocked   -> not plotted
        am_open = np.array([am if s == "open" else np.nan
                            for am, s in zip(airmass, states)])
        am_vig  = np.array([am if s == "vignetted" else np.nan
                            for am, s in zip(airmass, states)])

        host_r = host.twinx()
        lim = POINTING_LIMITS[args.telescope]
        host_r.plot(grid, am_open, color="#444444", lw=1.7, alpha=0.85, zorder=5,
                    label=f"{args.target_name} airmass (unvignetted)")
        if np.isfinite(am_vig).any():
            host_r.plot(grid, am_vig, color="#888888", lw=1.4, ls="--",
                        alpha=0.7, zorder=5,
                        label="vignetted (0–18°)")
        host_r.set_ylabel(f"{args.target_name} airmass", fontsize=12, color="#444444")
        host_r.tick_params(axis="y", labelcolor="#444444")
        # Lowest airmass (best, = transit) at the TOP, via inverted limits.
        finite_all = np.concatenate([am_open[np.isfinite(am_open)],
                                     am_vig[np.isfinite(am_vig)]]) \
                     if (np.isfinite(am_open).any() or np.isfinite(am_vig).any()) \
                     else np.array([])
        if args.airmass_center and finite_all.size:
            amin = np.nanmin(am_open) if np.isfinite(am_open).any() else finite_all.min()
            host_r.set_ylim(amin + args.airmass_pad, amin - args.airmass_pad)
        elif finite_all.size:
            amin = finite_all.min(); amax = finite_all.max()
            host_r.set_ylim(amax + 0.05, amin - 0.05)
        # (airmass legend intentionally omitted: solid = unvignetted, dashed =
        #  vignetted 0-18 deg; the right-axis label already identifies the curve)

        # console note on accessibility
        n_open = sum(1 for s in states if s == "open")
        n_vig  = sum(1 for s in states if s == "vignetted")
        n_blk  = sum(1 for s in states if s == "blocked")
        print(f"  Target accessibility ({args.telescope}): "
              f"{n_open} unvignetted, {n_vig} vignetted, {n_blk} blocked "
              f"(of {len(states)} grid pts)")

        # --- observing-window boxes (shaded = on-target ZA-projected) --------
        for (w0, w1) in windows:
            for ax in axes:
                ax.axvspan(w0, w1, facecolor=C_GC, alpha=0.07, zorder=0)
                ax.axvspan(w0, w1, facecolor="none", edgecolor=C_GC, lw=1.5,
                           ls="--", zorder=6)
            outside_lbl = (f"ZA {args.zenith_angle:g}°" if args.zenith_angle
                           else "ZA 0° (zenith)")
            _lbl = (f"{args.target_name} observations "
                    f"({w0:%H:%M}–{w1:%H:%M} HST)\n"
                    f"shaded = on-target ZA-projected · unshaded = {outside_lbl}")
            if window_label_margin:
                # GUI: hang the label just ABOVE the LTAO panel (into the
                # normally-empty NGS-bottom gap) so it never sits on data. x
                # tracks the window centre; y is in axes fraction (blended).
                lax = axd[lgs_key]
                lax.annotate(
                    _lbl, xy=(w0 + (w1 - w0) / 2, 1.015),
                    xycoords=lax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    color=C_GC, annotation_clip=False,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GC,
                              lw=1.2, alpha=0.95), zorder=7)
            else:
                axd[lgs_key].annotate(
                    _lbl, xy=(w0 + (w1 - w0) / 2, 0.575), ha="center", va="top",
                    fontsize=9, fontweight="bold", color=C_GC,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GC,
                              lw=1.2, alpha=0.95), zorder=7)

    return fig


def render_terms_figure(args, prep, res):
    """Build and return the error-budget-terms Figure, or None when there
    are no MASS profiles or --no-terms-plot is set. Does NOT save. Same
    determinism guarantee as render_main_figure."""
    _ltao_bw_fac = prep._ltao_bw_fac
    baseline_zen_factor = prep.baseline_zen_factor
    dimm_dt = prep.dimm_dt
    dimm_sec = prep.dimm_sec
    dimm_see = prep.dimm_see
    fixed_zen_factor = prep.fixed_zen_factor
    in_any_window = prep.in_any_window
    lam_label = prep.lam_label
    lam_nm = prep.lam_nm
    mass_dt = prep.mass_dt
    mass_sec = prep.mass_sec
    mass_see = prep.mass_see
    night_date = prep.night_date
    out_path = prep.out_path
    profiles = prep.profiles
    show_target = prep.show_target
    tomography_on = prep.tomography_on
    ut_stamp = prep.ut_stamp
    windows = prep.windows
    zen_factor_by_time = prep.zen_factor_by_time
    col_airmass = res.col_airmass
    col_ang1 = res.col_ang1
    col_d0 = res.col_d0
    col_dimm = res.col_dimm
    col_mass = res.col_mass
    col_mm = res.col_mm
    col_ngs_th0 = res.col_ngs_th0
    col_r0_cm = res.col_r0_cm
    col_terms = res.col_terms
    col_theta0 = res.col_theta0
    col_tt10 = res.col_tt10
    col_zf = res.col_zf
    n_fb = res.n_fb
    ngs_bright = res.ngs_bright
    ngs_faint = res.ngs_faint
    ngs_fb = res.ngs_fb
    p_airmass = res.p_airmass
    p_dimm_in = res.p_dimm_in
    p_secs = res.p_secs
    p_times = res.p_times
    p_zf = res.p_zf
    sr_ltao = res.sr_ltao
    sr_single = res.sr_single
    th0_assumed = res.th0_assumed
    times = res.times
    if len(p_times) and not args.no_terms_plot:
        span_dts = [times.min(), times.max(), dimm_dt.min(), dimm_dt.max()]
        if len(mass_dt):
            span_dts += [mass_dt.min(), mass_dt.max()]
        data_lo, data_hi = min(span_dts), max(span_dts)
        pad = timedelta(minutes=20)
        fig2 = Figure(figsize=(15, 12))
        axg = fig2.subplots(4, 2, sharex=True)
        fig2.suptitle(
            f"LGS/LTAO error-budget terms (nm RMS) — {night_date:%Y-%m-%d} "
            f"({args.telescope}, {'legacy' if args.legacy_budget else 'refined'} "
            f"budget)\nfixed terms (rows 1-3 omit them; bottom row includes "
            f"them): HO measurement {budget.HOMEAS:g} · Na focus {budget.NAFOC:g} · "
            f"static {static_subtotal(args.telescope):.0f} ({args.telescope}) · "
            f"margin {budget.MARGIN:g} nm",
            fontsize=12, fontweight="bold")
        T = col_terms   # columns: fit scint ang bw_s bw_l cone_s alt_l tt
        # angular-aniso panel: label the offset actually charged, and show what
        # an on-axis (0") laser would cost -- identically zero under the
        # refined law, so that reference is a note rather than a flat line
        _th = (DEF_LGS_OFFSET[args.telescope] if args.lgs_offset is None
               else args.lgs_offset)
        #  the 1" reference stands in for realistic imperfect laser centering:
        #  a perfectly on-axis (0") beacon would zero this term, but perfect
        #  centering may not be achievable in practice
        if args.legacy_budget:
            ang_series = [("angular aniso — legacy 2″ allocation",
                           T[:, 2], "#7A2E5D", "-")]
            ang_note = None
        elif _th > 0:
            ang_series = [(f"angular aniso — {_th:g}″ laser offset (current)",
                           T[:, 2], "#7A2E5D", "-"),
                          ("angular aniso — 1″ offset (centering ref)",
                           col_ang1, "#7A2E5D", "--")]
            ang_note = None
        else:
            ang_series = [("angular aniso — on-axis laser (0″, current)",
                           T[:, 2], "#7A2E5D", "-"),
                          ("angular aniso — 1″ offset (centering ref)",
                           col_ang1, "#7A2E5D", "--")]
            ang_note = ("current term is 0 at 0″; dashed = cost of 1″ "
                        "imperfect centering")
        tt_label  = (f"tip-tilt — R={args.tt_mag:g} @ {args.tt_offset:g}″ "
                     f"(current)")
        tt_ref_lb = "tip-tilt — R=10 on-axis (reference)"
        #  when tomography is off (K2 default) the LTAO-specific series are
        #  meaningless -- drop them from the bandwidth and altitude panels so
        #  the figure shows only what was actually computed
        if tomography_on:
            bw_series = [("bandwidth — single", T[:, 3], "#1B6CA8", "-"),
                         ("bandwidth — LTAO (half-rate)", T[:, 4], "#1B6CA8", "--")]
            alt_series = [("focal aniso — single",       T[:, 5], "#B02020", "-"),
                          ("tomography+mismatch — LTAO", T[:, 6], "#138086", "-")]
        else:
            bw_series  = [("bandwidth — single", T[:, 3], "#1B6CA8", "-")]
            alt_series = [("focal aniso — single", T[:, 5], "#B02020", "-")]
        panels = [
            (axg[0, 0], [("fitting",            T[:, 0], "#B05A00", "-")], None),
            (axg[0, 1], [("scintillation",      T[:, 1], "#4C7A34", "-")], None),
            (axg[1, 0], ang_series, ang_note),
            (axg[1, 1], [(tt_label,             T[:, 7], "#5B4B8A", "-"),
                         (tt_ref_lb,            col_tt10, "#5B4B8A", "--")], None),
            (axg[2, 0], bw_series, None),
            (axg[2, 1], alt_series, None),
        ]
        for ax2, series, note in panels:
            for label, vals, color, ls in series:
                g_t, g_v = gapline(p_times, vals, p_secs, args.match_tol)
                ax2.plot(g_t, g_v, ls, color=color, lw=1.5, label=label)
                ax2.plot(p_times, vals, "o", color=color, ms=2.4, alpha=0.7)
            ax2.legend(fontsize=8, loc="upper right", framealpha=0.85)
            ax2.grid(alpha=0.3)
            # headroom above the data so the upper-right legend never covers
            # it: ~13% of the axis height per legend row plus a base margin
            _mx = max((np.nanmax(v) for _, v, _, _ in series
                       if np.isfinite(np.asarray(v, float)).any()), default=1.0)
            ax2.set_ylim(0, _mx * (1.18 + 0.13 * len(series)))
            if note:
                ax2.text(0.015, 0.04, note, transform=ax2.transAxes,
                         fontsize=8, color="#555555", va="bottom")
        # ---- row 4: fractional contribution of every term to the total ----
        #  Percent of total wavefront-error VARIANCE (variances are what add:
        #  S_total = exp(-(2pi/lam)^2 (sum sigma_i^2))), including the fixed
        #  terms and tip-tilt, for each laser mode.
        _tel = args.telescope
        FIX = [("HO meas", budget.HOMEAS, "#999999"),
               ("Na focus", budget.NAFOC, "#BBBBBB"),
               (f"tel aberr ({_tel})", STATIC_TEL[_tel], "#C3C3C3"),
               ("WFS calib", budget.STATIC_CALIB, "#CBCBCB"),
               ("DM static", budget.STATIC_DM, "#D1D1D1"),
               ("AO+instr", budget.STATIC_INST, "#D7D7D7"),
               ("registration", budget.STATIC_REG, "#DDDDDD"),
               ("margin", budget.MARGIN, "#E3E3E3")]
        single_stack = [
            ("fitting", T[:, 0], "#B05A00"),
            ("focal aniso (→ tomo in LTAO)", T[:, 5], "#B02020"),
            ("ang aniso", T[:, 2], "#7A2E5D"), ("bandwidth", T[:, 3], "#1B6CA8"),
            ("scint", T[:, 1], "#4C7A34"), ("tip-tilt", T[:, 7], "#5B4B8A")]
        ltao_stack = [
            ("fitting", T[:, 0], "#B05A00"),
            ("tomo+mismatch (replaces focal)", T[:, 6], "#138086"),
            ("ang aniso", T[:, 2], "#7A2E5D"), ("bandwidth", T[:, 4], "#1B6CA8"),
            ("scint", T[:, 1], "#4C7A34"), ("tip-tilt", T[:, 7], "#5B4B8A")]

        def _draw_stack(ax2, mode_label, varying):
            layers = varying + [(n, np.full(len(p_times), v), c)
                                for n, v, c in FIX]
            var = np.array([np.asarray(v, float) ** 2 for _, v, _ in layers])
            frac = 100.0 * var / var.sum(axis=0)
            cum = np.vstack([np.zeros(len(p_times)), np.cumsum(frac, axis=0)])
            for k, (name, _v, color) in enumerate(layers):
                _t, lo = gapline(p_times, cum[k],     p_secs, args.match_tol)
                _t, hi = gapline(p_times, cum[k + 1], p_secs, args.match_tol)
                ax2.fill_between(_t, lo, hi, color=color, alpha=0.85,
                                 linewidth=0, label=name)
            ax2.set_ylim(0, 100)
            ax2.set_title(f"{mode_label} — % of total WFE variance",
                          fontsize=9, pad=2)
            ax2.grid(alpha=0.25)

        if tomography_on:
            _draw_stack(axg[3, 0], "single-beacon LGS", single_stack)
            _draw_stack(axg[3, 1], "LTAO", ltao_stack)
            legend_ax = axg[3, 0]
        else:
            #  tomography off (K2 default): no LTAO panel -- span single-beacon
            #  across both columns and remove the empty right axis
            gs = axg[3, 0].get_gridspec()
            axg[3, 0].remove(); axg[3, 1].remove()
            ax_full = fig2.add_subplot(gs[3, :])
            _draw_stack(ax_full, "single-beacon LGS", single_stack)
            ax_full.set_xlabel(f"HST (night of {night_date:%Y-%m-%d})")
            ax_full.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax_full.set_xlim(data_lo - pad, data_hi + pad)
            legend_ax = ax_full
        #  the stacked areas fill the axes by construction, so the legend
        #  lives entirely OUTSIDE, to the left of the panel (one column).
        #  Build it explicitly so the focal-aniso/tomography slot -- which is
        #  RED in the single-beacon stack and TEAL in the LTAO stack -- is
        #  shown as a split red/teal swatch, making the mode-dependent color
        #  swap visible, not just stated in the label text.
        FIT, FOC, TOMO = "#B05A00", "#B02020", "#138086"
        ANG, BW, SCI, TT = "#7A2E5D", "#1B6CA8", "#4C7A34", "#5B4B8A"
        leg_handles, leg_labels = [], []
        leg_handles.append(Rectangle((0, 0), 1, 1, fc=FIT)); leg_labels.append("fitting")
        if tomography_on:
            leg_handles.append(_SplitSwatch(FOC, TOMO))
            leg_labels.append("focal aniso / tomo+mismatch")
        else:
            leg_handles.append(Rectangle((0, 0), 1, 1, fc=FOC))
            leg_labels.append("focal aniso (→ tomo in LTAO)")
        for c, lb in [(ANG, "ang aniso"), (BW, "bandwidth"),
                      (SCI, "scint"), (TT, "tip-tilt")]:
            leg_handles.append(Rectangle((0, 0), 1, 1, fc=c)); leg_labels.append(lb)
        for lb, _v, c in FIX:
            leg_handles.append(Rectangle((0, 0), 1, 1, fc=c)); leg_labels.append(lb)
        legend_ax.legend(leg_handles, leg_labels,
                         handler_map={_SplitSwatch: _SplitSwatchHandler()},
                         fontsize=7.5, loc="center right",
                         bbox_to_anchor=(-0.09, 0.5), ncol=1,
                         framealpha=0.95, labelspacing=0.4, borderaxespad=0)
        legend_ax.set_ylabel("% of total variance")

        for ax2 in axg[:3, 0]:
            ax2.set_ylabel("WFE (nm RMS)")
        # bottom-row x-labels: when tomography is on both percent panels remain;
        # when off, the full-width panel (ax_full) was already labeled above
        if tomography_on:
            for ax2 in axg[3, :]:
                ax2.set_xlabel(f"HST (night of {night_date:%Y-%m-%d})")
                ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        # apply span + window shading to whichever axes still exist
        live_axes = [a for a in fig2.axes]
        for ax2 in live_axes:
            ax2.set_xlim(data_lo - pad, data_hi + pad)
            # mirror the main figure's window shading so the terms panels carry
            # the same target/line-of-sight context: inside these spans the
            # terms are projected onto the target's airmass, not zenith
            if show_target:
                for (w0, w1) in windows:
                    ax2.axvspan(w0, w1, color="#6A3D9A", alpha=0.06, zorder=0)
        if show_target:
            # one shared note (figure bottom-left) stating the projection
            _proj = (f"shaded = {args.target_name} observing window "
                     f"(terms projected onto target airmass); "
                     f"unshaded = ZA {args.zenith_angle:g}°"
                     if args.zenith_angle else
                     f"shaded = {args.target_name} observing window "
                     f"(terms projected onto target airmass); unshaded = zenith")
            fig2.text(0.055, 0.005, _proj, fontsize=8, color="#555555",
                      ha="left", va="bottom")
        fig2.tight_layout(rect=[0.055, 0.02 if show_target else 0, 1, 0.93])
        return fig2
    return None


FWHM_COLLAPSE_MULT = 3.0   # FWHM > this x lambda/D  =>  the AO core is gone
FWHM_MIN_SPAN_FRAC = 0.15  # never show a y-range narrower than this x lambda/D


def _fwhm_axis_limits(arrays, dl_mas):
    """Linear y-limits (mas) for a FWHM panel, plus how many samples fall off
    the top.

    The delivered FWHM is strongly BIMODAL: it sits within a few mas of the
    diffraction limit while the core survives, then jumps to the seeing disk
    (~1000 mas) the moment the core collapses -- with essentially nothing in
    between. Autoscaling to those spikes flattens the whole night into a line,
    so the axis is bounded by the NON-COLLAPSED population (FWHM <=
    FWHM_COLLAPSE_MULT x lambda/D). The collapsed samples are then labelled on
    the plot rather than drawn, and the axis actually resolves the structure
    that matters. A minimum span keeps an ultra-flat night from magnifying
    sub-mas noise. Returns (lo, hi, n_clipped, n_total, max_value)."""
    cat = np.concatenate([np.asarray(a, dtype=float) for a in arrays
                          if a is not None and len(a)])
    cat = cat[np.isfinite(cat)]
    if cat.size == 0:
        return 0.0, max(2.0 * dl_mas, 1.0), 0, 0, np.nan
    good = cat[cat <= FWHM_COLLAPSE_MULT * dl_mas]
    if good.size == 0:                       # everything collapsed
        good = cat
    lo = min(dl_mas, float(good.min())) * 0.98
    hi = float(good.max()) * 1.06
    min_span = FWHM_MIN_SPAN_FRAC * dl_mas
    if hi - lo < min_span:
        hi = lo + min_span
    return lo, hi, int((cat > hi).sum()), int(cat.size), float(np.nanmax(cat))


def _annotate_fwhm_axis(ax, lo, hi, n_clip, n_tot, vmax):
    """Apply the linear limits and, when samples were clipped, say so on-plot
    so a truncated axis can never be mistaken for a well-behaved night."""
    ax.set_ylim(lo, hi)
    if n_clip:
        ax.annotate(f"{n_clip}/{n_tot} samples above axis (max {vmax:.0f} mas)"
                    f" — core lost, FWHM → seeing disk",
                    xy=(0.995, 0.965), xycoords="axes fraction",
                    ha="right", va="top", fontsize=8, color="#C0392B",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="#C0392B", lw=0.8, alpha=0.9), zorder=8)


def render_fwhm_figure(args, prep, res, show=None):
    """Build the FWHM timeline Figure for --report fwhm (used INSTEAD of the
    Strehl figure in that mode): NGS panel on the DIMM timebase, LGS/LTAO panel
    on the profile timebase, linear mas, with the diffraction limit drawn as a
    reference. `show` selects which convention(s) to plot -- 'halfmax' (the
    core+halo model), 'gaussfit' (the no-background quick-look Gaussian-fit
    simulation), 'gaussfit-sky' (the free-background quick-look Gaussian-fit
    simulation), 'both' (halfmax+gaussfit, kept for back-compat), or 'all' --
    defaulting to args.fwhm_curves; plotting all three at once is busy, so a
    single convention is the norm."""
    tomography_on = prep.tomography_on
    lam_nm, lam_label = prep.lam_nm, prep.lam_label
    night_date = prep.night_date
    show = show or getattr(args, "fwhm_curves", "srtool")
    hm = show in ("halfmax", "both", "all")
    gf = show in ("gaussfit", "both", "all")
    gs = show in ("gaussfit-sky", "all")
    st = show in ("srtool", "all")

    fig = Figure(figsize=(13, 6.7))
    axes = fig.subplots(2, 1, sharex=True, gridspec_kw={"hspace": 0.12})
    ax_ngs, ax_lgs = axes

    dl_mas = 1.029 * (lam_nm * 1e-9) / TEL_DIAMETER_M * 206265.0e3
    C_NGS, C_LGS, C_LTAO, C_SEE = "#6A3D9A", "#1B3A6B", "#138086", "#C0392B"

    fig.suptitle(
        f"AO PSF FWHM ({lam_label}) vs. time — {night_date:%Y-%m-%d} "
        f"({args.telescope}, {'LTAO' if tomography_on else 'single-beacon'})",
        fontweight="bold", fontsize=13)
    # the seeing disk is 5-25x the delivered FWHM: stated once here rather than
    # plotted, so it cannot compress the delivered curves off the axis
    _see = (res.col_dimm * res.col_zf * (lam_nm / 500.0) ** (-0.2)) * 1000.0
    fig.text(0.5, 0.935,
             f"seeing disk (no AO, not plotted): median "
             f"{np.nanmedian(_see):.0f} mas   range "
             f"{np.nanmin(_see):.0f}–{np.nanmax(_see):.0f} mas",
             ha="center", va="top", fontsize=9, style="italic", color=C_SEE)

    # NGS panel (DIMM timebase): dots = core+halo model, dashed = Gaussian-fit
    # simulation; which are drawn is set by `show`.
    _lim = []
    if hm:
        ax_ngs.plot(res.times, res.fwhm_ngs_bright, "o", color=C_NGS, ms=3.5,
                    label=f"NGS R={args.ngs_bright:g} — core+halo model")
        ax_ngs.plot(res.times, res.fwhm_ngs_faint, "o", color=C_NGS, ms=2.5,
                    alpha=0.35, label=f"NGS R={args.ngs_faint:g} — core+halo model")
        _lim += [res.fwhm_ngs_bright, res.fwhm_ngs_faint]
    if gf:
        ax_ngs.plot(res.times, res.fwhm_gauss_ngs_bright, "--", color=C_NGS,
                    lw=1.0, alpha=0.85,
                    label=f"NGS R={args.ngs_bright:g} — Gaussian-fit sim. (quick-look)")
        _lim += [res.fwhm_gauss_ngs_bright]
    if gs:
        ax_ngs.plot(res.times, res.fwhm_sky_ngs_bright, ":", color=C_NGS,
                    lw=1.2, alpha=0.85,
                    label=f"NGS R={args.ngs_bright:g} — Gaussian-fit sim. (+background)")
        _lim += [res.fwhm_sky_ngs_bright]
    if st:
        ax_ngs.plot(res.times, res.fwhm_tool_ngs_bright, "-.", color=C_NGS,
                    lw=1.2, alpha=0.9,
                    label=f"NGS R={args.ngs_bright:g} — as the SR tool reads it")
        _lim += [res.fwhm_tool_ngs_bright]
    ax_ngs.set_ylabel("NGS FWHM (mas)")
    _annotate_fwhm_axis(ax_ngs, *_fwhm_axis_limits(_lim, dl_mas))

    # LGS / LTAO panel (profile timebase); same convention as the NGS panel.
    if len(res.p_times):
        _lim = []
        if hm:
            g_t, g_v = gapline(res.p_times, res.fwhm_single, res.p_secs,
                               args.match_tol)
            ax_lgs.plot(g_t, g_v, "-", color=C_LGS, lw=1.0, alpha=0.5)
            ax_lgs.plot(res.p_times, res.fwhm_single, "o", color=C_LGS, ms=3.5,
                        label="single LGS — core+halo model")
            _lim += [res.fwhm_single]
        if gf:
            g_t, g_v = gapline(res.p_times, res.fwhm_gauss_single, res.p_secs,
                               args.match_tol)
            ax_lgs.plot(g_t, g_v, "--", color=C_LGS, lw=1.1, alpha=0.85,
                        label="single LGS — Gaussian-fit sim. (quick-look)")
            _lim += [res.fwhm_gauss_single]
        if gs:
            g_t, g_v = gapline(res.p_times, res.fwhm_sky_single, res.p_secs,
                               args.match_tol)
            ax_lgs.plot(g_t, g_v, ":", color=C_LGS, lw=1.3, alpha=0.85,
                        label="single LGS — Gaussian-fit sim. (+background)")
            _lim += [res.fwhm_sky_single]
        if st:
            g_t, g_v = gapline(res.p_times, res.fwhm_tool_single, res.p_secs,
                               args.match_tol)
            ax_lgs.plot(g_t, g_v, "-.", color=C_LGS, lw=1.3, alpha=0.9,
                        label="single LGS — as the SR tool reads it")
            _lim += [res.fwhm_tool_single]
        if tomography_on:
            if hm:
                g_t, g_v = gapline(res.p_times, res.fwhm_ltao, res.p_secs,
                                   args.match_tol)
                ax_lgs.plot(g_t, g_v, "-", color=C_LTAO, lw=1.0, alpha=0.5)
                ax_lgs.plot(res.p_times, res.fwhm_ltao, "o", color=C_LTAO,
                            ms=3.5, label="LTAO — core+halo model")
                _lim += [res.fwhm_ltao]
            if gf:
                g_t, g_v = gapline(res.p_times, res.fwhm_gauss_ltao, res.p_secs,
                                   args.match_tol)
                ax_lgs.plot(g_t, g_v, "--", color=C_LTAO, lw=1.1, alpha=0.85,
                            label="LTAO — Gaussian-fit sim.")
                _lim += [res.fwhm_gauss_ltao]
            if gs:
                g_t, g_v = gapline(res.p_times, res.fwhm_sky_ltao, res.p_secs,
                                   args.match_tol)
                ax_lgs.plot(g_t, g_v, ":", color=C_LTAO, lw=1.3, alpha=0.85,
                            label="LTAO — Gaussian-fit sim. (+background)")
                _lim += [res.fwhm_sky_ltao]
            if st:
                g_t, g_v = gapline(res.p_times, res.fwhm_tool_ltao, res.p_secs,
                                   args.match_tol)
                ax_lgs.plot(g_t, g_v, "-.", color=C_LTAO, lw=1.3, alpha=0.9,
                            label="LTAO — as the SR tool reads it")
                _lim += [res.fwhm_tool_ltao]
        _annotate_fwhm_axis(ax_lgs, *_fwhm_axis_limits(_lim, dl_mas))
    else:
        ax_lgs.text(0.5, 0.5, "no MASS profiles — LGS/LTAO FWHM unavailable",
                    transform=ax_lgs.transAxes, ha="center", va="center",
                    color="#666")
        ax_lgs.set_ylim(dl_mas * 0.96, dl_mas * 2.0)
    ax_lgs.set_ylabel("LGS/LTAO FWHM (mas)")

    for ax in axes:
        ax.axhline(dl_mas, color="0.35", lw=0.9, ls="--", zorder=1)
        # label BELOW the line: the delivered FWHM always sits above it, so a
        # va="bottom" label would land on the data
        ax.annotate(f"diffraction limit {dl_mas:.0f} mas", xy=(0.995, dl_mas),
                    xycoords=("axes fraction", "data"),
                    xytext=(0, -3), textcoords="offset points",
                    ha="right", va="top", fontsize=8, color="0.35")
        ax.grid(True, which="major", alpha=0.25)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        if prep.show_target:
            for (w0, w1) in prep.windows:
                ax.axvspan(w0, w1, color="#6A3D9A", alpha=0.06, zorder=0)
    ax_lgs.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_lgs.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax_lgs.set_xlabel(f"HST (night of {night_date:%Y-%m-%d})")

    _box_mas = getattr(args, "fwhm_box_mas", 300.0)
    _nshown = int(hm) + int(gf) + int(gs) + int(st)
    fig.subplots_adjust(bottom=0.10 + 0.018 * max(_nshown - 1, 0))
    _row = 0.006 + 0.022 * (_nshown - 1)
    if hm:
        fig.text(0.5, _row,
                 f"{'dots — ' if _nshown > 1 else ''}half-max of the "
                 f"3-component PSF: Airy core (D={TEL_DIAMETER_M:g} m) ⊛ "
                 f"tip-tilt jitter + corrected-band shoulder (Moffat, "
                 f"FWHM=θc) + seeing wings; no confirmed real-tool analog "
                 f"(see psf_fwhm_mas)",
                 ha="center", va="bottom", fontsize=8, color="#777777")
        _row -= 0.022
    if gf:
        fig.text(0.5, _row,
                 f"{'dashed — ' if _nshown > 1 else ''}Gaussian-fit sim., NO "
                 f"background (free amplitude, no sky, box={_box_mas:g} "
                 f"mas): models the OSIRIS quicklook tool's rarely-used "
                 f"Strehl button (OSIRISSTREHL_QL2.pro), at a box size "
                 f"chosen for you",
                 ha="center", va="bottom", fontsize=8, color="#777777")
        _row -= 0.022
    if gs:
        fig.text(0.5, _row,
                 f"{'dotted — ' if _nshown > 1 else ''}Gaussian-fit sim., "
                 f"FREE background (box={_box_mas:g} mas): models the "
                 f"OSIRIS quicklook tool's hand-drawn-box fit feature -- a "
                 f"separate, independent tool from the AO Strehl tool",
                 ha="center", va="bottom", fontsize=8, color="#777777")
        _row -= 0.022
    if st:
        fig.text(0.5, _row,
                 f"{'dash-dot — ' if _nshown > 1 else ''}what THIS tool's own "
                 f"Measured-SR tab reads: the same PSF rendered on NIRC2 "
                 f"pixels, annulus sky subtracted, through its find_fwhm.pro "
                 f"port (fwhm_srtool_mas) -- the curve to compare against a "
                 f"MEASURED FWHM",
                 ha="center", va="bottom", fontsize=8, color="#777777")
    return fig


def overlay_fwhm_on_main(fig, args, prep, res, show=None):
    """--report both: add a FWHM curve on a right-hand axis of the NGS panel and
    the LGS/LTAO panel of the (unchanged) Strehl figure. `show` selects the
    convention(s) as in render_fwhm_figure (default args.fwhm_curves). The twin
    axes are appended AFTER the frozen panels, so the default outputs are
    untouched."""
    show = show or getattr(args, "fwhm_curves", "srtool")
    hm = show in ("halfmax", "both", "all")
    gf = show in ("gaussfit", "both", "all")
    gs = show in ("gaussfit-sky", "all")
    st = show in ("srtool", "all")
    C_FW = "#B26A00"
    ax_ngs, ax_mid = fig.axes[0], fig.axes[1]
    dl_mas = 1.029 * (prep.lam_nm * 1e-9) / TEL_DIAMETER_M * 206265.0e3

    axr = ax_ngs.twinx()
    _lim = []
    if hm:
        axr.plot(res.times, res.fwhm_ngs_bright, "--", color=C_FW, lw=1.1,
                 label="FWHM — core+halo model")
        _lim += [res.fwhm_ngs_bright]
    if gf:
        axr.plot(res.times, res.fwhm_gauss_ngs_bright, ":", color=C_FW, lw=1.1,
                 label="FWHM — Gaussian-fit sim.")
        _lim += [res.fwhm_gauss_ngs_bright]
    if gs:
        axr.plot(res.times, res.fwhm_sky_ngs_bright, "-.", color=C_FW, lw=1.1,
                 label="FWHM — Gaussian-fit sim. (+background)")
        _lim += [res.fwhm_sky_ngs_bright]
    if st:
        axr.plot(res.times, res.fwhm_tool_ngs_bright, "-", color=C_FW, lw=1.1,
                 label="FWHM — as the SR tool reads it")
        _lim += [res.fwhm_tool_ngs_bright]
    axr.set_ylabel("FWHM (mas)", color=C_FW, fontsize=9)
    axr.tick_params(axis="y", labelcolor=C_FW, labelsize=8)
    axr.legend(loc="upper right", fontsize=7, framealpha=0.85)
    _annotate_fwhm_axis(axr, *_fwhm_axis_limits(_lim, dl_mas))

    axr2 = ax_mid.twinx()
    fw_mid = res.fwhm_ltao if prep.tomography_on else res.fwhm_single
    fwg_mid = (res.fwhm_gauss_ltao if prep.tomography_on
               else res.fwhm_gauss_single)
    fws_mid = (res.fwhm_sky_ltao if prep.tomography_on
               else res.fwhm_sky_single)
    fwt_mid = (res.fwhm_tool_ltao if prep.tomography_on
               else res.fwhm_tool_single)
    if len(res.p_times):
        _lim = []
        if hm:
            g_t, g_v = gapline(res.p_times, fw_mid, res.p_secs, args.match_tol)
            axr2.plot(g_t, g_v, "--", color=C_FW, lw=1.1,
                      label="FWHM — core+halo model")
            _lim += [fw_mid]
        if gf:
            g_t, g_v = gapline(res.p_times, fwg_mid, res.p_secs, args.match_tol)
            axr2.plot(g_t, g_v, ":", color=C_FW, lw=1.1,
                      label="FWHM — Gaussian-fit sim.")
            _lim += [fwg_mid]
        if gs:
            g_t, g_v = gapline(res.p_times, fws_mid, res.p_secs, args.match_tol)
            axr2.plot(g_t, g_v, "-.", color=C_FW, lw=1.1,
                      label="FWHM — Gaussian-fit sim. (+background)")
            _lim += [fws_mid]
        if st:
            g_t, g_v = gapline(res.p_times, fwt_mid, res.p_secs, args.match_tol)
            axr2.plot(g_t, g_v, "-", color=C_FW, lw=1.1,
                      label="FWHM — as the SR tool reads it")
            _lim += [fwt_mid]
        axr2.legend(loc="upper right", fontsize=7, framealpha=0.85)
        _annotate_fwhm_axis(axr2, *_fwhm_axis_limits(_lim, dl_mas))
    else:
        axr2.set_ylim(dl_mas * 0.96, dl_mas * 2.0)
    axr2.set_ylabel("FWHM (mas)", color=C_FW, fontsize=9)
    axr2.tick_params(axis="y", labelcolor=C_FW, labelsize=8)
    return fig


# --- UTC display post-processing (GUI-only; the CLI never calls these, so
# --- the harness-frozen default outputs stay HST) -----------------------------
import re as _re

_HHMM_RE = _re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")


def shift_hst_text(text, hours=10.0):
    """Shift every HH:MM clock time inside `text` by `hours` and swap the
    'HST' tag for 'UTC'. ONLY for strings this codebase generated, where
    every clock time is an HST wall time (plot annotations, field-snapshot
    descriptions) -- it is a display conversion, not a datetime parser."""

    def _sub(m):
        h, mi = int(m.group(1)), int(m.group(2))
        h = int((h + hours) % 24)
        return f"{h:02d}:{mi:02d}"

    return _HHMM_RE.sub(_sub, text).replace("HST", "UTC")


def apply_utc_display(fig, hours=10.0):
    """Relabel a RENDERED figure's HST time axes and annotations to UTC:
    tick labels shift +10 h, 'HST (night of ...)' x-labels become
    'UTC (night of ... HST)', and any axes text carrying clock times tagged
    HST is rewritten via shift_hst_text. Purely cosmetic post-processing --
    the plotted data/datetimes are untouched, so window shading, cursors
    and pixel content stay exactly where the engine put them."""
    from matplotlib.ticker import FuncFormatter

    off = timedelta(hours=hours)

    def _fmt(x, _pos):
        return (mdates.num2date(x) + off).strftime("%H:%M")

    for ax in fig.axes:
        xl = ax.get_xlabel() or ""
        if xl.startswith("HST (") and xl.endswith(")"):
            ax.set_xlabel(f"UTC ({xl[5:-1]} HST)", fontsize=12)
            ax.xaxis.set_major_formatter(FuncFormatter(_fmt))
        for t in ax.texts:
            s = t.get_text()
            if "HST" in s:
                t.set_text(shift_hst_text(s, hours))
    for t in fig.texts:
        s = t.get_text()
        if "HST" in s:
            t.set_text(shift_hst_text(s, hours))


def render_predicted_terms_figure(args, snap, bw_factor, lam_nm, lam_label):
    """Error-budget terms for a PREDICTED (synthetic) scenario -- the
    Error-terms tab's rendering when the Prediction tab drives the field
    map. Unlike render_terms_figure (a night's per-sample time series),
    this is a single-snapshot term breakdown: grouped horizontal bars of
    every budget term in nm RMS for single-beacon LGS and LTAO, computed
    by the same lgs_budget_terms() call the timeline and field map use,
    at the snapshot's line-of-sight conditions. `snap` is a
    synthetic_field_snapshot() dict; `bw_factor` the LTAO bandwidth
    factor (prep._ltao_bw_fac or its no-run surrogate). Honors active
    budget_overrides like every other renderer."""
    from .marechal import marechal_strehl

    et, ef = snap["eps_tot_los"], snap["eps_fa_los"]
    kw = dict(tt_mag=args.tt_mag, tt_offset=args.tt_offset,
              lgs_offset=args.lgs_offset, legacy=args.legacy_budget,
              v_ground=args.wind_ground, v_free=args.wind_free,
              aniso_scale=snap.get("aniso_scale", 1.0),
              tt_sensor=getattr(args, "_tt_sensor_base", "strap"),
              strap_law=getattr(args, "strap_law", "sheet"),
              ltao_tt_theta0_gain=getattr(args, "ltao_tt_theta0_gain", None))
    t_s = budget.lgs_budget_terms(et, ef, args.telescope, "single",
                                  None, **kw)
    t_l = budget.lgs_budget_terms(et, ef, args.telescope, "ltao",
                                  snap.get("cn2_bins"), bw_factor=bw_factor,
                                  **kw)

    def static_nm(t):
        return float(np.sqrt(t["stat_tel"] ** 2 + t["stat_calib"] ** 2
                             + t["stat_dm"] ** 2 + t["stat_inst"] ** 2
                             + t["stat_reg"] ** 2))

    rows = [("fitting", "fit"), ("scintillation", "scint"),
            ("angular aniso", "ang"), ("bandwidth", "bw"),
            ("focal aniso /\ntomo+mismatch", "alt"),
            ("HO measurement", "meas"), ("Na focus", "nafoc"),
            ("static / calibration", None), ("HO margin", "margin"),
            ("tip-tilt", "tt")]

    def val(t, key):
        return static_nm(t) if key is None else float(t[key])

    def totals(t):
        ho = np.sqrt(sum(val(t, k) ** 2 for _, k in rows if k != "tt"))
        s = marechal_strehl(ho, lam_nm) * marechal_strehl(t["tt"], lam_nm)
        return ho, s

    ho_s, s_s = totals(t_s)
    ho_l, s_l = totals(t_l)

    fig = Figure(figsize=(9.0, 6.4), layout="constrained")
    ax = fig.add_subplot(111)
    y = np.arange(len(rows))[::-1]
    h = 0.38
    ax.barh(y + h / 2, [val(t_s, k) for _, k in rows], h,
            color="#C1443C", alpha=0.85, label="single-beacon LGS")
    ax.barh(y - h / 2, [val(t_l, k) for _, k in rows], h,
            color="#1B6CA8", alpha=0.85, label="LTAO")
    for yy, (_, k) in zip(y, rows):
        ax.text(val(t_s, k) + 2, yy + h / 2, f"{val(t_s, k):.0f}",
                va="center", fontsize=8, color="#C1443C")
        ax.text(val(t_l, k) + 2, yy - h / 2, f"{val(t_l, k):.0f}",
                va="center", fontsize=8, color="#1B6CA8")
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _ in rows], fontsize=9)
    ax.set_xlabel("wavefront error (nm RMS)", fontsize=9)
    ax.grid(alpha=0.25, axis="x")
    # headroom so the end-of-bar value labels stay inside the axes
    vmax = max(max(val(t_s, k), val(t_l, k)) for _, k in rows)
    ax.set_xlim(0, vmax * 1.10)
    # legend OUTSIDE the axes (below): any in-axes corner can be covered
    # by a long bar for some scenario -- lower-right sat exactly on the
    # tip-tilt row (2026-08-12)
    fig.legend(loc="outside lower center", ncols=2, fontsize=9)
    ax.set_title(
        f"PREDICTED SCENARIO — error-budget terms · {lam_label}\n"
        f"zenith DIMM {snap['eps_tot_zenith']:.2f}″ / MASS "
        f"{snap['eps_fa_zenith']:.2f}″ · ZA {snap['zenith_angle_deg']:g}° "
        f"(line of sight {et:.2f}″/{ef:.2f}″) · θ₀ {snap['theta0_los']:.1f}″"
        f" · m={snap.get('m', 0.0):.2f}\n"
        f"single-beacon: HO {ho_s:.0f} nm, tt {t_s['tt']:.0f} nm → "
        f"S {s_s:.3f}    LTAO: HO {ho_l:.0f} nm, tt {t_l['tt']:.0f} nm → "
        f"S {s_l:.3f}", fontsize=10, color="#B36A00")
    return fig
