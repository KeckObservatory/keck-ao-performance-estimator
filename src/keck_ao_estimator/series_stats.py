"""Mean and standard deviation of Strehl, FWHM and WFE over a series of
measured frames (Measured SR "Series stats").

Pure: takes the per-frame `Nirc2StrehlResult`s the GO loop already
produces and returns the summary; nothing is re-measured here.  A frame is
left out of the statistics, and listed with the reason, when its
measurement failed, its SR is unphysical (outside (0, 1]) or the star is
saturated -- the same frames the log already flags as unusable.  The
standard deviation is the SAMPLE one (ddof=1; nan for a single frame), the
scatter of the frames themselves, not the error of the mean.
"""
from dataclasses import dataclass, field
import math

SERIES_METRICS = (("strehl", "SR", "{:.3f}"),
                  ("fwhm_mas", "FWHM (mas)", "{:.2f}"),
                  ("wfe_nm", "WFE (nm)", "{:.1f}"))


@dataclass(frozen=True)
class MetricStats:
    mean: float
    std: float          # sample standard deviation (ddof=1)
    min: float
    max: float


@dataclass(frozen=True)
class SeriesStats:
    n: int                              # frames used
    labels: tuple                       # their labels, in input order
    excluded: tuple = ()                # ((label, reason), ...)
    metrics: dict = field(default_factory=dict)   # key -> MetricStats


def exclusion_reason(result):
    """Why a frame is left out of the series statistics, or None."""
    if result is None:
        return "not measured"
    if isinstance(result, str):         # the GO loop's failure message
        return result or "measurement failed"
    if not result.ok:
        return result.error or "measurement failed"
    if result.unphysical:
        return f"unphysical SR {result.strehl:+.3f}"
    if result.saturated:
        return "saturated"
    return None


def _stats(values):
    n = len(values)
    mean = sum(values) / n
    std = (math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
           if n > 1 else float("nan"))
    return MetricStats(mean, std, min(values), max(values))


def summarize_series(items):
    """`items`: iterable of (label, result), where result may also be None
    (not measured) or the failure message string.  Returns SeriesStats
    over the usable frames (metrics empty when none is usable)."""
    used, excluded = [], []
    for label, r in items:
        why = exclusion_reason(r)
        if why is None:
            used.append((str(label), r))
        else:
            excluded.append((str(label), why))
    metrics = {}
    if used:
        for key, _name, _fmt in SERIES_METRICS:
            metrics[key] = _stats([float(getattr(r, key)) for _l, r in used])
    return SeriesStats(n=len(used), labels=tuple(l for l, _r in used),
                       excluded=tuple(excluded), metrics=metrics)


_LINE_FMT = (("strehl", "SR", "{:.3f}", ""),
             ("fwhm_mas", "FWHM", "{:.2f}", " mas"),
             ("wfe_nm", "WFE", "{:.1f}", " nm"))


def format_series_line(st, from_log=0):
    """The one-line log entry for a series: frame span, n used, how many
    were excluded / taken from the log, then mean±stdev per metric."""
    if not st.labels:
        span = "(none)"
    elif st.n == 1:
        span = st.labels[0]
    else:
        span = f"{st.labels[0]}..{st.labels[-1]}"
    notes = []
    if st.excluded:
        notes.append(f"{len(st.excluded)} excluded")
    if from_log:
        notes.append(f"{from_log} from log")
    head = f"Series {span}: n={st.n}" + (f" ({', '.join(notes)})"
                                          if notes else "")
    if not st.metrics:
        return head + "  no usable frames"
    parts = []
    for key, name, fmt, unit in _LINE_FMT:
        m = st.metrics[key]
        std = "—" if math.isnan(m.std) else fmt.format(m.std)
        parts.append(f"{name} {fmt.format(m.mean)}±{std}{unit}")
    return head + "  " + "  ".join(parts)


def format_series_stats(st):
    """Log lines: one per metric, then one per excluded frame."""
    head = f"Series stats: {st.n} frame(s)"
    if st.excluded:
        head += f", {len(st.excluded)} excluded"
    lines = [head]
    if st.labels:
        lines.append(f"  frames {st.labels[0]} .. {st.labels[-1]}"
                     if st.n > 2 else "  frames " + ", ".join(st.labels))
    for key, name, fmt in SERIES_METRICS:
        m = st.metrics.get(key)
        if m is None:
            continue
        std = "—" if math.isnan(m.std) else fmt.format(m.std)
        lines.append(f"  {name:<11s} mean {fmt.format(m.mean)}  "
                     f"stdev {std}  (min {fmt.format(m.min)}, "
                     f"max {fmt.format(m.max)})")
    if not st.metrics:
        lines.append("  no usable frames")
    for label, why in st.excluded:
        lines.append(f"  excluded {label}: {why}")
    return lines
