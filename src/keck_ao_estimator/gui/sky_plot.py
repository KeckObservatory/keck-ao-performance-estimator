"""Sky plot for the starlist dialog: where a target sits in Keck's pointing
space over the night.

A polar map with the zenith at the centre and elevation rings: radius =
zenith distance, azimuth N up and E right (map view, the same orientation
as the LGS-flux map). It draws:
  * the telescope's pointing limits (constants.POINTING_LIMITS): the
    Nasmyth-deck wedge, blocked below its floor; the vignetted band below
    18 deg elsewhere; the zenith ceiling; and the ring above which guiding
    is not guaranteed;
  * the target's path across the night (17:00-08:00 HST), solid where the
    Sun is below -12 deg and faint in twilight, with hour ticks;
  * the target at the dialog's "Evaluate at" time, coloured by its
    pointing state (open / vignetted / blocked);
  * the Moon at that time, with the dialog's 15 and 30 deg avoidance
    circles.
Pure drawing: the caller passes numbers computed by the engine
(night_track, moon_altaz_deg, pointing_state).
"""
import numpy as np

from ..constants import POINTING_LIMITS

BLOCKED_C = "#c23b22"
VIGNET_C = "#eda100"
OPEN_C = "#1baf7a"
TRACK_C = "#2a78d6"
MOON_C = "#8a8a8a"
STATE_C = {"open": OPEN_C, "vignetted": VIGNET_C, "blocked": BLOCKED_C}


def _zd(el):
    return 90.0 - np.asarray(el, float)


def small_circle(az0, el0, radius_deg, n=181):
    """Points (az, el) at angular distance radius_deg from (az0, el0)."""
    b = np.radians(np.linspace(0, 360, n))
    d, e0, a0 = np.radians(radius_deg), np.radians(el0), np.radians(az0)
    sin_el = np.sin(e0) * np.cos(d) + np.cos(e0) * np.sin(d) * np.cos(b)
    el = np.arcsin(np.clip(sin_el, -1, 1))
    daz = np.arctan2(np.sin(b) * np.sin(d) * np.cos(e0),
                     np.cos(d) - np.sin(e0) * sin_el)
    return (np.degrees(a0 + daz) % 360.0), np.degrees(el)


def _plot_masked(ax, az, el, mask, **kw):
    """Plot az/el as a polar line, broken where mask is False and where the
    azimuth wraps through north (a jump of more than 180 deg)."""
    az = np.asarray(az, float)
    th = np.radians(az)
    r = _zd(el)
    keep = np.asarray(mask, bool).copy()
    keep[1:] &= ~(np.abs(np.diff(az)) > 180.0)
    ax.plot(np.where(keep, th, np.nan), np.where(keep, r, np.nan), **kw)


def draw_sky(fig, telescope, track=None, now=None, moon=None, is_utc=False,
             title=None):
    """Draw the sky plot into `fig` (cleared first).

    track: dict from engine.night_track (times_hst, az, el, sun_alt) or None.
    now:   dict(az, el, state) for the "Evaluate at" time, or None.
    moon:  dict(az, el, illum_pct) or None.
    is_utc: label the hour ticks in UT instead of HST."""
    fig.clear()
    ax = fig.add_axes([0.08, 0.21, 0.84, 0.67], projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)                     # N up, E right (map view)
    ax.set_ylim(0, 90)
    lim = POINTING_LIMITS[telescope]
    a0, a1 = lim["wedge"]
    wedge_th = np.radians(np.linspace(a0, a1, 60))
    out_th = np.radians(np.linspace(a1, a0 + 360.0, 200))

    # pointing limits
    ax.fill_between(wedge_th, _zd(lim["wedge_floor"]), 90.0, color=BLOCKED_C,
                    alpha=0.30, lw=0)
    ax.fill_between(out_th, _zd(lim["open_floor"]), 90.0, color=VIGNET_C,
                    alpha=0.25, lw=0)
    full = np.radians(np.linspace(0, 360, 361))
    ax.fill_between(full, 0.0, _zd(lim["ceiling"]), color=BLOCKED_C, alpha=0.6, lw=0)
    ax.plot(full, np.full_like(full, _zd(lim["guide_warn"])), color=BLOCKED_C,
            lw=0.7, ls=":")
    ax.plot(wedge_th, np.full_like(wedge_th, _zd(lim["wedge_floor"])),
            color=BLOCKED_C, lw=1.2)
    ax.plot(out_th, np.full_like(out_th, _zd(lim["open_floor"])),
            color=VIGNET_C, lw=1.0)
    mid = np.radians(0.5 * (a0 + a1))
    ax.text(mid, 0.8 * 90.0 + 0.2 * _zd(lim["wedge_floor"]),
            f"deck\n<{lim['wedge_floor']:g}°", color=BLOCKED_C, fontsize=6.5,
            ha="center", va="center")

    # target path across the night
    if track is not None:
        el, az, sun = track["el"], track["az"], track["sun_alt"]
        up = el > 0
        _plot_masked(ax, az, el, up & (sun < -12), color=TRACK_C, lw=1.8)
        _plot_masked(ax, az, el, up & (sun >= -12), color=TRACK_C, lw=1.0,
                     alpha=0.35)
        for t, a, e, s in zip(track["times_hst"], az, el, sun):
            if e > 0 and t.minute == 0 and s < -12:
                h = (t.hour + (10 if is_utc else 0)) % 24
                ax.plot(np.radians(a), _zd(e), "o", ms=2.5, color=TRACK_C)
                ax.text(np.radians(a), _zd(e) + 3.5, f"{h:d}h", fontsize=6,
                        color=TRACK_C, ha="center", va="center")

    # the Moon, with the 15/30 deg avoidance circles
    if moon is not None and moon["el"] > -5:
        for rad, ls in ((15.0, "-"), (30.0, "--")):
            caz, cel = small_circle(moon["az"], moon["el"], rad)
            _plot_masked(ax, caz, cel, cel > 0, color=MOON_C, lw=0.8, ls=ls)
        if moon["el"] > 0:
            ax.plot(np.radians(moon["az"]), _zd(moon["el"]), "o", ms=9,
                    color="#d8d8d8", mec=MOON_C, mew=1.2)
            ax.text(np.radians(moon["az"]), _zd(moon["el"]) - 6,
                    f"Moon {moon.get('illum_pct', 0):.0f}%", fontsize=6.5,
                    color="#555555", ha="center")

    # the target now
    if now is not None and now["el"] > 0:
        ax.plot(np.radians(now["az"]), _zd(now["el"]), "*", ms=14,
                color=STATE_C.get(now.get("state"), TRACK_C), mec="black", mew=0.6)
    elif now is not None:
        fig.text(0.5, 0.125, "target below the horizon at the time above",
                 ha="center", fontsize=6.8, color=BLOCKED_C)

    ax.set_yticks([30, 60])
    ax.set_yticklabels(["60°", "30°"], fontsize=6.5, color="#555555")
    ax.set_rlabel_position(0.5 * (a0 + a1) + 180.0)   # opposite the deck label
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"], fontsize=8)
    ax.grid(alpha=0.35, lw=0.6)
    fig.suptitle(title or f"{telescope} sky (zenith centre, N up, E right)",
                 fontsize=8, y=0.985, va="top")
    for y, txt in ((0.08, "★ target now   — tonight's path (faint: twilight)"),
                   (0.05, "○ Moon, 15°/30° circles   red: blocked   amber: vignetted <18°"),
                   (0.02, f"dotted: guiding not guaranteed above {lim['guide_warn']:g}°")):
        fig.text(0.5, y, txt, ha="center", fontsize=6.3, color="#444444")
    return ax
