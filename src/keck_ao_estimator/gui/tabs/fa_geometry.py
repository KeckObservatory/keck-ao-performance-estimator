"""FA pierce-point geometry dialog: the live version of Figure 3 in the
2026-07-21 FA-event working note, the figure that made the Keck-vs-monitor
timing story click. Two panels, recreated for TONIGHT's target and winds:

  (a) plan view -- the monitor's pierce-region circles (its star sits
      within ~h of zenith at layer height h), the Keck beam's pierce points
      marching out toward the target azimuth, and (new vs the note's static
      figure) each upper layer's GFS WIND VECTOR drawn at its pierce point
      with the resulting lead/lag annotation;
  (b) side view along the Keck azimuth -- the monitor's pierce cone, the
      beam line at the target elevation, and the per-layer column
      separation ranges.

Both panels also carry the monitor's three most probable catalog stars
(mkam_catalog.top_monitor_orientations over the trailing 45 min up to the
reference time -- the stay-near-zenith scheduler model, so what the user
sees tracks the date and time): ranked beams + summit-zoom inset in (a),
one true-zenith-angle line each in (b), with the per-layer lead pinned to
the top candidate.  Degrades to the plain ignorance-band look if the
catalog model fails for any reason.

The reference time follows the Data-tab Period selector whenever it names
a single instant ('specific time', or Nighttime mode's last pull), so the
dialog shows the time the user is analysing; 'observing window' / 'whole
night' fall back to the last MASS sample.  A below-horizon target says so
in the title instead of silently dropping the beam.

Display-only, like the FA advisory itself: reads the target az/el and
winds; alters nothing. Non-modal dialog, recreated on every click (the
ranking-dialog pattern -- change the target or time, click again for the
updated geometry), references dropped on any close.
"""
from datetime import timedelta

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from qtcompat import QtWidgets

import keck_ao_estimator as engine
from keck_ao_estimator.constants import HST_TO_UTC_HOURS
from keck_ao_estimator.fa_geometry_plot import draw_fa_geometry


class FaGeometryMixin:
    def _show_fa_geometry(self):
        """Open (recreating if already open) the FA geometry dialog for the
        current advisory state. Degrades explicitly: no run -> message; no
        target/below horizon -> circles+cone only; no winds -> no arrows."""
        if self.res is None:
            self.status.setText("Run the estimator first — the FA geometry "
                                "needs a night loaded.")
            return
        if getattr(self, "_fa_geo_dialog", None) is not None:
            self._fa_geo_dialog.close()

        # reference time: follow the Data-tab Period selector when it names
        # a single instant ('specific time', or Nighttime mode's last pull),
        # so the dialog tracks the time the user is looking at instead of
        # pinning to the end of the night; else the legacy fallbacks.
        t_ref = None
        try:
            when, t_hst = self._when_time_from(self.stats_cond,
                                               self.stats_time)
            if when == "time" and t_hst is not None:
                t_ref = t_hst
        except Exception:
            t_ref = None
        if t_ref is None:
            t_ref = getattr(self, "_nighttime_last_pull", None)
        if t_ref is None and len(self.res.p_times):
            t_ref = self.res.p_times[-1]
        az = el = None
        if self.target_enable.isChecked() and t_ref is not None:
            eff = self._effective_target_coords()
            if eff is not None:
                try:
                    _am, elv, azv = engine.compute_airmass_curve(
                        eff[0], eff[1], [t_ref])
                    el, az = float(elv[0]), float(azv[0])
                except Exception:
                    el = az = None
        winds = getattr(self, "_gfs_winds_result", None)

        # the monitor's probable stars: stay-near-zenith catalog model over
        # the trailing 45 min (the scheduler sits on the recent best star),
        # az/el snapshot at the reference time.  t_ref is HST-naive like
        # everything else in the GUI; the model wants UTC.
        cands = None
        if t_ref is not None:
            try:
                utc_ref = t_ref + timedelta(hours=HST_TO_UTC_HOURS)
                window = [utc_ref - timedelta(minutes=m)
                          for m in range(45, -1, -5)]
                cands = [(c["az"], c["el"], c["pretty"], c["prob"])
                         for c in engine.top_monitor_orientations(
                             window, utc_ref, n=3)] or None
            except Exception:
                cands = None                  # plain ignorance-band look

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("FA pierce-point geometry — monitor vs Keck beam")
        dlg.resize(980, 540)
        lay = QtWidgets.QVBoxLayout(dlg)
        fig = Figure(figsize=(9.6, 4.9), layout="constrained")
        canvas = FigureCanvasQTAgg(fig)
        self._draw_fa_geometry(fig, az, el, winds, t_ref, cands)
        lay.addWidget(canvas, 1)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(dlg.close)
        lay.addWidget(close)
        dlg.finished.connect(lambda *_: setattr(self, "_fa_geo_dialog", None))
        dlg.show()
        self._fa_geo_dialog = dlg

    def _draw_fa_geometry(self, fig, az, el, winds, t_ref, cands=None):
        tname = self.tname_edit.text().strip() or "target"
        when = self._fmt_hm(t_ref) if t_ref is not None else "—"
        ax = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        wtxt = draw_fa_geometry(ax, ax2, az, el, winds, tname=tname,
                                monitor_candidates=cands)
        # say WHY there is no beam, right where the user is looking
        extra = (f" · {tname} below horizon here (el {el:.0f}°)"
                 if el is not None and el <= 0 else "")
        fig.suptitle(f"{tname} at {when} · {wtxt}{extra}", fontsize=10)
