"""Pure-matplotlib rendering of the FA pierce-point geometry (the
2026-07-21 FA-event note's Figure 3): the (a) plan view and (b) side
view, drawn onto caller-supplied axes so both the GUI dialog
(gui/tabs/fa_geometry.py) and offline report scripts share ONE
implementation.  Qt-free on purpose -- it only needs matplotlib + the
engine's pierce-point / lead-lag helpers.
"""
import numpy as np

from .budget import DEF_LGS_OFFSET
from .fa_advisory import event_lead_lag, pierce_points

BEAM = "#9e1b1b"          # the note figure's dark red
CONE = "#3c6fb0"          # its monitor-region blue
WIND = "#1B6B3A"          # arrow green (matches the app's ok cue)

_HEIGHTS_M = [4000.0, 8000.0, 16000.0]
_MARKERS = {4000.0: "o", 8000.0: "s", 16000.0: "D"}


def draw_fa_geometry(ax_plan, ax_side, az, el, winds, tname="target",
                     monitor_azel=None, monitor_candidates=None):
    """Draw the plan view onto ax_plan and the side view onto ax_side for
    a target at (az, el) degrees with GFS `winds` (night_winds() dict or
    None).  Degrades: no beam when the target is None / below the horizon.

    monitor_azel : None (default, the GUI's live use) leaves the monitor's
    pointing unknown -- its pierce region is the dashed circles and the
    lead/lag carries the +/- ignorance band.  Pass an explicit (az, el) to
    overlay a POSTULATED monitor pointing: its own beam and pierce points
    are drawn, and the lead/lag is pinned to that pointing (no band).

    monitor_candidates : sequence of (az, el, label, weight) -- the ranked
    catalog-model monitor pointings (mkam_catalog.top_monitor_orientations).
    Each candidate's beam + pierce points are drawn in fading blues with a
    summary box giving its probability and 16-km lead, plus a summit-zoom
    inset (the near-zenith beams cluster within a few km of the origin at
    the full plan scale); the per-height lead/lag on the target's pierce
    points is pinned to the MOST PROBABLE candidate.  Overrides
    monitor_azel.  In the side view each candidate appears as one line at
    its TRUE zenith angle, drawn in the +/- half by whether its azimuth is
    within 90 deg of the target's (the off-azimuth beams do not lie in the
    target-azimuth cut, so this shows real zenith distance, not an
    in-plane projection).
    Returns a short wind footer string."""
    heights_m = _HEIGHTS_M
    markers = _MARKERS
    cands = [c for c in (monitor_candidates or []) if c[1] > 0]
    shades = ["#3c6fb0", "#7b9fd4", "#aec3e2"]
    if cands:
        monitor_azel = (cands[0][0], cands[0][1])   # pin lead to top pick

    # ---- (a) plan view ---------------------------------------------------
    ax = ax_plan
    ax.set_title("(a) Plan view: pierce points at layer altitude",
                 fontsize=10)
    for h in heights_m:
        circ = np.linspace(0, 2 * np.pi, 181)
        ax.plot(h / 1e3 * np.sin(circ), h / 1e3 * np.cos(circ),
                "--", color=CONE, alpha=0.25 + 0.2 * (h / 16000.0), lw=1.2)
        # with a candidate box in the NE corner, ring labels move to the
        # NW diagonal so the 16-km one stays visible
        lx, ly = ((-h / 1e3 * 0.7071, h / 1e3 * 0.7071) if cands
                  else (0.0, h / 1e3))
        ax.annotate(f"h = {h / 1e3:g} km", xy=(lx, ly), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8,
                    color=CONE)
    ax.plot(0, 0, "^", color="black", ms=9, zorder=5)
    ax.annotate("summit\n(Keck / MKAM)", xy=(0, 0), xytext=(-8, 8),
                textcoords="offset points", ha="right", fontsize=8)

    have_beam = az is not None and el is not None and el > 0
    lim = 19.5
    bins = {b[0]: b for b in (winds or {}).get("bins_full", [])}
    if have_beam:
        pts = pierce_points(az, el, heights_m)
        far = pts[-1]
        lim = max(lim, np.hypot(*far) / 1e3 * 1.18)
        ax.plot([0, far[0] / 1e3 * 1.12], [0, far[1] / 1e3 * 1.12],
                "-", color=BEAM, lw=1.6, zorder=4)
        lead = {ll[0] * 1e3: ll for ll in event_lead_lag(
            az, el, [bins[h] for h in heights_m if h in bins],
            monitor_azel=monitor_azel)}
        # postulated monitor pointing: draw its beam + pierce points
        # (candidates, when given, replace this entirely -- drawn below)
        if not cands and monitor_azel is not None and monitor_azel[1] > 0:
            mpts = pierce_points(monitor_azel[0], monitor_azel[1], heights_m)
            mfar = mpts[-1]
            lim = max(lim, np.hypot(*mfar) / 1e3 * 1.18)
            ax.plot([0, mfar[0] / 1e3 * 1.12], [0, mfar[1] / 1e3 * 1.12],
                    "-", color=CONE, lw=1.4, zorder=4)
            for h, (pe, pn) in zip(heights_m, mpts):
                ax.plot(pe / 1e3, pn / 1e3, markers[h], mfc="none",
                        mec=CONE, ms=8, mew=1.4, zorder=5)
            ax.annotate(f"postulated monitor\naz {monitor_azel[0]:.0f}° "
                        f"el {monitor_azel[1]:.0f}°",
                        xy=(mfar[0] / 1e3 * 0.6, mfar[1] / 1e3 * 0.6),
                        xytext=(6, 6), textcoords="offset points",
                        fontsize=8, color=CONE)
        for h, (pe, pn) in zip(heights_m, pts):
            x, y = pe / 1e3, pn / 1e3
            ax.plot(x, y, markers[h], color=BEAM, ms=7, zorder=5)
            d = float(np.hypot(x, y))
            lbl = f"{h / 1e3:g} km ({d:.1f} km out)"
            if h in lead:
                _hk, c, r = lead[h]
                lbl += (f"\n{c:+.0f} min" if r < 0.05
                        else f"\n{c:+.0f}±{r:.0f} min")
            ax.annotate(lbl, xy=(x, y), xytext=(7, -3),
                        textcoords="offset points", fontsize=8, color=BEAM)
            if h in bins:
                _h, v, dir_from = bins[h]
                th = np.radians(dir_from)
                ue, un = -np.sin(th), -np.cos(th)   # toward (E, N)
                length = 0.35 * v                   # km per (m/s)
                ax.annotate(
                    "", xy=(x + ue * length, y + un * length), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=WIND, lw=1.6),
                    zorder=6)
                ax.annotate(f"{v:g} m/s",
                            xy=(x + ue * length, y + un * length),
                            xytext=(5 if ue >= 0 else -5, 4),
                            textcoords="offset points", fontsize=8,
                            ha="left" if ue >= 0 else "right", color=WIND)
        ax.annotate(f"az {az:.0f}°",
                    xy=(far[0] / 1e3 * 0.55, far[1] / 1e3 * 0.55),
                    xytext=(-8, 0), textcoords="offset points",
                    ha="right", fontsize=8, color=BEAM)
    elif cands:
        # keep clear of the summit-zoom inset (lower left)
        ax.text(0.985, 0.02, "no target above the horizon at the\n"
                "reference time — monitor region only",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="#555")
    else:
        ax.text(0.5, 0.06, "no target above the horizon at the reference "
                "time — monitor region only", transform=ax.transAxes,
                ha="center", fontsize=8, color="#555")
    # ranked catalog-model candidates: fading-blue beams + summary box +
    # summit-zoom inset.  Target-independent (the monitor points where it
    # points), so drawn with or without a Keck beam; the per-candidate
    # 16-km lead needs the beam and is omitted without one.
    if cands:
        box = ["monitor candidates (MKAM catalog model):"]
        for i, (caz, cel, clbl, cw) in enumerate(cands):
            col = shades[min(i, len(shades) - 1)]
            cpts = pierce_points(caz, cel, heights_m)
            cfar = cpts[-1]
            lim = max(lim, np.hypot(*cfar) / 1e3 * 1.35)
            ax.plot([0, cfar[0] / 1e3 * 1.12], [0, cfar[1] / 1e3 * 1.12],
                    "-", color=col, lw=1.6 - 0.25 * i, zorder=4)
            for h, (pe, pn) in zip(heights_m, cpts):
                ax.plot(pe / 1e3, pn / 1e3, markers[h], mfc="none",
                        mec=col, ms=7, mew=1.3, zorder=5)
            line = (f"{i + 1}. {clbl}  P={cw:.0%}  "
                    f"az {caz:.0f}° el {cel:.0f}°")
            if have_beam and 16000.0 in bins:
                (_h16, c16, _r16), = event_lead_lag(
                    az, el, [bins[16000.0]], monitor_azel=(caz, cel))
                line += f" → 16 km {c16:+.0f} min"
            box.append(line)
        ax.text(0.97, 0.97, "\n".join(box), transform=ax.transAxes,
                fontsize=7.6, va="top", ha="right", color="#2a4d7c",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="#3c6fb0", alpha=0.85), zorder=7)
        # summit-zoom inset: near-zenith candidate beams cross within
        # a few km of the origin -- unreadable at the full plan scale
        zr = 5.2
        axz = ax.inset_axes([0.015, 0.015, 0.34, 0.34])
        axz.set_xlim(-zr, zr); axz.set_ylim(-zr, zr)
        axz.set_aspect("equal")
        axz.set_xticks([]); axz.set_yticks([])
        axz.set_facecolor("#fcfcfc")
        circ = np.linspace(0, 2 * np.pi, 181)
        axz.plot(4.0 * np.sin(circ), 4.0 * np.cos(circ), "--",
                 color=CONE, alpha=0.4, lw=1.0)
        if have_beam:
            axz.plot([0, far[0] / 1e3], [0, far[1] / 1e3], "-",
                     color=BEAM, lw=1.6, zorder=4)
            for h, (pe, pn) in zip(heights_m, pts):
                axz.plot(pe / 1e3, pn / 1e3, markers[h], color=BEAM,
                         ms=7, zorder=5)
        for i, (caz, cel, _clbl, _cw) in enumerate(cands):
            col = shades[min(i, len(shades) - 1)]
            cpts = pierce_points(caz, cel, heights_m)
            cfar = cpts[-1]
            axz.plot([0, cfar[0] / 1e3 * 1.25],
                     [0, cfar[1] / 1e3 * 1.25],
                     "-", color=col, lw=1.8 - 0.25 * i, zorder=4)
            for h, (pe, pn) in zip(heights_m, cpts):
                axz.plot(pe / 1e3, pn / 1e3, markers[h], mfc="none",
                         mec=col, ms=8, mew=1.5, zorder=5)
            axz.annotate(f"{i + 1}", xy=(cfar[0] / 1e3, cfar[1] / 1e3),
                         xytext=(7, 7), textcoords="offset points",
                         ha="center", fontsize=9, color=col,
                         fontweight="bold", zorder=6,
                         bbox=dict(boxstyle="circle,pad=0.15",
                                   fc="white", ec=col, lw=0.8,
                                   alpha=0.9))
        axz.plot(0, 0, "^", color="black", ms=7, zorder=6)
        axz.set_title(f"summit zoom (±{zr:g} km)", fontsize=7.5, pad=2)
        ax.indicate_inset_zoom(axz, edgecolor="#777", alpha=0.7)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("East offset (km)"); ax.set_ylabel("North offset (km)")
    ax.annotate("N", xy=(-0.85 * lim, 0.74 * lim), fontsize=9, color="#444",
                ha="center")
    ax.annotate("", xy=(-0.85 * lim, 0.70 * lim),
                xytext=(-0.85 * lim, 0.54 * lim),
                arrowprops=dict(arrowstyle="-|>", color="#444"))
    ax.grid(alpha=0.2)

    # ---- (b) side view along the beam azimuth ----------------------------
    ax2 = ax_side
    ax2.set_title("(b) Side view along the Keck azimuth", fontsize=10)
    hmax = 17.5
    ax2.fill_betweenx([0, hmax], [0, -hmax], [0, hmax],
                      color=CONE, alpha=0.15, lw=0)
    ax2.plot([0, -hmax], [0, hmax], color=CONE, lw=1.2)
    ax2.plot([0, hmax], [0, hmax], color=CONE, lw=1.2)
    ax2.text(-11, 11.6, "MASS/DIMM\npierce region\n(star ≤ 45° from "
             "zenith)", fontsize=8, color=CONE, ha="center")
    for h in heights_m:
        ax2.axhline(h / 1e3, color="#999", ls=":", lw=1)
        ax2.annotate(f"{h / 1e3:g} km layer", xy=(-hmax + 0.7, h / 1e3),
                     xytext=(0, 3), textcoords="offset points",
                     fontsize=8, color="#777")
    xmax = 22.0
    if have_beam:
        tan_el = np.tan(np.radians(el))
        xmax = max(xmax, 16.0 / tan_el * 1.08)
        xb = min(hmax / tan_el, xmax)
        ax2.plot([0, xb], [0, xb * tan_el], color=BEAM, lw=1.8)
        for h in heights_m:
            d = h / tan_el / 1e3
            ax2.plot(d, h / 1e3, markers[h], color=BEAM, ms=7)
            lo = abs(d - h / 1e3)
            hi = d + h / 1e3
            ax2.annotate(f"sep {lo:.0f}–{hi:.0f} km",
                         xy=((d - h / 1e3) / 2, h / 1e3), xytext=(0, 4),
                         textcoords="offset points", ha="center",
                         fontsize=8, color=WIND)
        ax2.text(0.97, 0.30, f"Keck beam to\n{tname}\n(el {el:.1f}°)",
                 transform=ax2.transAxes, ha="right", fontsize=8,
                 color=BEAM)
    # one line per catalog-model candidate at its TRUE zenith angle; the
    # +/- half says only whether its azimuth is within 90 deg of the
    # target's (these beams do not lie in the target-azimuth cut)
    if cands:
        for i, (caz, cel, _clbl, _cw) in enumerate(cands):
            col = shades[min(i, len(shades) - 1)]
            toward = (az is None or
                      abs(((caz - az) + 180.0) % 360.0 - 180.0) <= 90.0)
            sgn = 1.0 if toward else -1.0
            tanc = np.tan(np.radians(cel))
            ax2.plot([0, sgn * hmax / tanc], [0, hmax], "-", color=col,
                     lw=1.5 - 0.2 * i, zorder=3)
            for h in heights_m:
                ax2.plot(sgn * h / tanc / 1e3, h / 1e3, markers[h],
                         mfc="none", mec=col, ms=7, mew=1.3, zorder=4)
            y_num = hmax - 3.0          # clear of the 16-km sep labels
            ax2.annotate(f"{i + 1}", xy=(sgn * y_num / tanc, y_num),
                         ha="center", va="center", fontsize=8.5, color=col,
                         fontweight="bold", zorder=6,
                         bbox=dict(boxstyle="circle,pad=0.12", fc="white",
                                   ec=col, lw=0.8, alpha=0.9))
        ax2.text(0.02, 0.02, "1–3: candidate monitor beams at true zenith "
                 "angle\n(±x: azimuth within 90° of the target's, or not)",
                 transform=ax2.transAxes, fontsize=7, color="#2a4d7c",
                 va="bottom")
    ax2.plot(0, 0, "^", color="black", ms=8)
    half = max(hmax + 1.0, xmax)
    ax2.set_xlim(-half, half); ax2.set_ylim(0, hmax)
    ax2.set_xlabel(f"Horizontal distance toward az {az:.0f}° (km)"
                   if have_beam else "Horizontal distance (km)")
    ax2.set_ylabel("Altitude above summit (km)")
    ax2.grid(alpha=0.2)

    return (f"GFS winds {winds['hours']}" if winds else
            "no GFS winds fetched — no wind vectors")


# KAPA LTAO uses a 4-beacon square asterism of 7.6" radius whose centre
# sits 4.97" off the science target (gui/constants.LGS_ASTERISM_*,
# budget.DEF_LGS_OFFSET["K1"] -- the 2026-08-07 campaign value, was 7").
# At 90 km that radius is only ~3.3 m
# aperture-projected, so its cone-effect edge recovery is modest -- LTAO's
# real gain is TOMOGRAPHIC reconstruction of the Cn2(h) profile.  The
# schematic exaggerates the beacon spread (and lowers the apex) for
# legibility; the real numbers are named on the plot.
KAPA_N_BEACONS = 4
KAPA_ASTERISM_RADIUS_ARCSEC = 7.6
KAPA_LGS_OFFSET_ARCSEC = DEF_LGS_OFFSET["K1"]   # single source (was 7.0)


def draw_cone_effect(ax, apex_km=24.0, aperture_m=5.0, view_km=20.0):
    """Schematic (NOT to scale -- sodium apex lowered, beacon spread
    exaggerated) contrasting single-LGS focal anisoplanatism with KAPA
    LTAO tomography in ONE axis: the science cylinder, the single beacon
    cone that under-samples the turbulence toward the cylinder edges (the
    cone effect, hatched red), and the KAPA 4-beacon asterism (green)
    that samples those layers from several angles so tomography can
    reconstruct them.  The true KAPA asterism (7.6\" radius, 7\" off-axis)
    is compact -- the spread here is enlarged so the geometry reads."""
    r = aperture_m
    y = np.linspace(0, view_km, 120)
    hw = r * (1.0 - y / apex_km)                   # cone half-width at y
    # exaggerated beacon offsets standing in for the 7.6"-radius asterism
    spread = 0.9 * r
    offs = np.linspace(-spread, spread, KAPA_N_BEACONS)

    ax.fill_betweenx(y, -r, r, color="#8fbce8", alpha=0.22, lw=0,
                     label="science cylinder (∞)")
    ax.fill_betweenx(y, hw, r, facecolor="none", edgecolor="#c0392b",
                     hatch="////", linewidth=0.0,
                     label="single-LGS unsensed (cone effect)")
    ax.fill_betweenx(y, -r, -hw, facecolor="none", edgecolor="#c0392b",
                     hatch="////", linewidth=0.0)
    lbl = f"KAPA {KAPA_N_BEACONS}-beacon asterism (tomography)"
    for off in offs:
        centre = off * y / apex_km
        ax.fill_betweenx(y, centre - hw, centre + hw, color="#2e8b57",
                         alpha=0.14, lw=0, label=lbl)
        lbl = None
        ax.plot(off, apex_km, "*", ms=9, color="#2e8b57", zorder=5,
                clip_on=False)
    ax.fill_betweenx(y, -hw, hw, color="#e8a33d", alpha=0.6, lw=0,
                     label="single-LGS cone")
    ax.plot(0, apex_km, "*", ms=14, color="#e8820d", zorder=6,
            clip_on=False)
    ax.plot([-r, r], [0, 0], color="k", lw=3)       # primary aperture
    ax.annotate("primary (10 m)", xy=(0, 0), xytext=(0, -14),
                textcoords="offset points", ha="center", fontsize=7)
    for h in (4, 8, 16):
        ax.axhline(h, color="#777", ls=":", lw=0.9)
        ax.annotate(f"{h} km", xy=(r, h), xytext=(3, 1),
                    textcoords="offset points", fontsize=7, color="#777")
    ax.annotate(f"sodium beacons ≈90 km\nsingle: 1 · KAPA: "
                f"{KAPA_N_BEACONS} on a {KAPA_ASTERISM_RADIUS_ARCSEC:g}\" "
                f"asterism\n(apex + spread exaggerated)",
                xy=(0, apex_km), xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=7)
    ax.set_xlim(-r * 2.7, r * 2.7)
    ax.set_ylim(0, apex_km + 3)
    ax.set_xlabel("aperture / horizontal (m, exaggerated)")
    ax.set_ylabel("altitude (km)")
    ax.set_title("(c) Cone effect: single-LGS vs KAPA LTAO", fontsize=10)
    ax.legend(fontsize=6.3, loc="lower center", framealpha=0.92)
