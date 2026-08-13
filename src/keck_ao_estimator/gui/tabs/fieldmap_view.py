"""Field-map VIEW transforms: the cosmetic display-rotation control (Field
PA) and the backdrop/frame flip-X/Y controls, plus the pure helpers behind
them (a rotation transform, a compass indicator, an image mirror). Split out
of fieldmap_tab.py (which owns the actual grid/heatmap rendering) because
this is generic display-orientation math with no dependency on the field-map
grid itself -- reusable wherever a plotted sky view needs re-orienting or a
resampled image needs a parity fix, not just this one map.
"""
import numpy as np
from qtcompat import QtWidgets

from ..widgets import _dspin


class FieldMapViewMixin:
    # ---- widget construction (called from FieldMapMixin._build_field_map_tab) --
    def _init_fm_pa_control(self, row):
        """Build the Field-PA rotation spinbox and add it to `row`."""
        self.fm_pa = _dspin(0, 360, 5, 0.0, 0, "°")
        self.fm_pa.setToolTip(
            "Rotate the displayed field (North -> East), e.g. to match a "
            "loaded frame's true sky rotation. Display only -- does not "
            "change the underlying grid/physics.")
        row.addWidget(QtWidgets.QLabel("Field PA:"))
        row.addWidget(self.fm_pa)
        self.fm_pa.valueChanged.connect(self._on_fieldmap_input_changed)

    def _init_fm_flip_controls(self, row):
        """Build the backdrop/frame flip-X/Y checkboxes and add them to
        `row`. A loaded FITS is placed purely by its own WCS (imaging.py
        trusts it as-is -- no instrument-specific parity fixup); these let
        the user correct BY EYE the rare file whose stored pixels are
        genuinely mirrored relative to their WCS. Applied to the
        already-resampled array at draw time (a mirror, not a re-fetch --
        see _fm_flip_image), so toggling is instant."""
        row.addWidget(QtWidgets.QLabel("Backdrop flip:"))
        self.fm_bg_flip_x = QtWidgets.QCheckBox("X")
        self.fm_bg_flip_y = QtWidgets.QCheckBox("Y")
        for cb in (self.fm_bg_flip_x, self.fm_bg_flip_y):
            cb.setToolTip("Mirror the backdrop image -- use if its orientation "
                          "looks wrong (an untrustworthy WCS parity).")
            row.addWidget(cb)
        row.addWidget(QtWidgets.QLabel("  Frame flip:"))
        self.fm_fg_flip_x = QtWidgets.QCheckBox("X")
        self.fm_fg_flip_y = QtWidgets.QCheckBox("Y")
        for cb in (self.fm_fg_flip_x, self.fm_fg_flip_y):
            cb.setToolTip("Mirror the inscribed frame -- use if its orientation "
                          "looks wrong (an untrustworthy WCS parity).")
            row.addWidget(cb)
        for cb in (self.fm_bg_flip_x, self.fm_bg_flip_y,
                   self.fm_fg_flip_x, self.fm_fg_flip_y):
            cb.toggled.connect(self._on_fieldmap_input_changed)

    def _init_fm_img_pa_control(self, row):
        """Build the manual IMAGE-only PA override and add it to `row`. Unlike
        Field PA (which rotates the whole displayed field -- backdrop,
        markers, catalogue, everything -- together, purely cosmetically),
        this rotates ONLY the loaded backdrop/frame imagery relative to the
        (unmoved) catalogue and markers, to correct a file whose WCS
        orientation is wrong. That is a real astrometric override, not a
        cosmetic one, so it is clearly labelled and flagged on the map
        whenever it is non-zero (see _draw_img_pa_warning) -- misused it
        makes a correctly-placed image WRONG."""
        self.fm_img_pa = _dspin(-180, 180, 1, 0.0, 0, "°")
        self.fm_img_pa.setToolTip(
            "MANUAL OVERRIDE — rotate ONLY the loaded image (backdrop/frame) "
            "relative to the guide-star catalogue, to correct a wrong WCS "
            "orientation. This overrides the file's astrometry: use ONLY to "
            "match the image to KNOWN catalogue stars, and leave at 0 unless "
            "you can see it is needed. Flagged on the map while non-zero.")
        lbl = QtWidgets.QLabel("  Image PA (manual):")
        lbl.setToolTip(self.fm_img_pa.toolTip())
        row.addWidget(lbl)
        row.addWidget(self.fm_img_pa)
        self.fm_img_pa.valueChanged.connect(self._on_fieldmap_input_changed)

    # ---- rotation transform + compass ---------------------------------------
    def _fm_pa_deg(self):
        return self.fm_pa.value() if hasattr(self, "fm_pa") else 0.0

    def _fm_img_pa_deg(self):
        return self.fm_img_pa.value() if hasattr(self, "fm_img_pa") else 0.0

    def _fm_rotation_transform(self, ax):
        """Affine transform that rotates every field-map artist by the Field
        PA control (North->East), composed with the axes' own data
        transform. A pure DISPLAY rotation -- the underlying grid the
        physics were evaluated on is unchanged, only how it's laid out on
        screen (e.g. to match a loaded frame's true sky rotation).

        Affine2D.rotate_deg's standard counter-clockwise-positive sense
        already matches "North->East" here for free: this plot's x-axis is
        East=-x/West=+x (mirrored from a plain Cartesian x=East, see the
        axis label), so rotating counter-clockwise sweeps "up" (North)
        towards "left" (East) -- exactly what a positive PA means."""
        from matplotlib.transforms import Affine2D
        return Affine2D().rotate_deg(self._fm_pa_deg()) + ax.transData

    def _fm_image_transform(self, ax, center=(0.0, 0.0)):
        """The Field-PA transform (above) with the MANUAL image-PA override
        composed in, for the loaded backdrop/frame imagery ONLY. The image is
        rotated by fm_img_pa about its own displayed centre (`center`, the
        backdrop shift -- usually (0,0)) BEFORE the whole-field Field-PA
        rotation, so the net effect is the image turning by fm_img_pa
        relative to the (Field-PA-only) catalogue and markers. At
        fm_img_pa==0 this is exactly _fm_rotation_transform, so imagery lines
        up with everything else. See _init_fm_img_pa_control."""
        from matplotlib.transforms import Affine2D
        cx, cy = center
        return (Affine2D().rotate_deg_around(cx, cy, self._fm_img_pa_deg())
                + self._fm_rotation_transform(ax))

    def _draw_img_pa_warning(self, ax):
        """Flag a non-zero manual image-PA prominently on the map -- this
        override deliberately breaks the image away from its own astrometry,
        so it must never be mistaken for a correctly-placed image."""
        pa = self._fm_img_pa_deg()
        if pa == 0.0:
            return
        ax.annotate(f"⚠ IMAGE MANUALLY ROTATED {pa:+g}° (overrides its WCS)",
                    xy=(0.5, 0.965), xycoords="axes fraction", ha="center",
                    va="top", fontsize=8.5, fontweight="bold", color="white",
                    zorder=1000,
                    bbox=dict(boxstyle="round,pad=0.3", fc="#C0392B",
                              ec="white", lw=1.0, alpha=0.95))

    def _draw_compass(self, ax, transform):
        """N/E direction arrows, rotated with the field so they always point
        at the true cardinal directions -- the fixed 'N↑ E←' text label is
        only honest at PA=0. Anchored in a fixed corner (axes-fraction) and
        sized independent of the transform's own scale, only its rotation is
        used."""
        pa = self._fm_pa_deg()
        if pa == 0.0:
            ax.annotate("N↑  E←", xy=(0.03, 0.97), xycoords="axes fraction",
                        fontsize=9, fontweight="bold", va="top", color="w",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black",
                                  alpha=0.35, ec="none"))
            return
        th = np.radians(pa)
        # unit vectors for North/East in axes-fraction space, rotated the
        # same sense as the data transform above (derived the same way: at
        # pa=0, North=(0,1) "up", East=(-1,0) "left")
        n_vec = (-np.sin(th), np.cos(th))
        e_vec = (-np.cos(th), -np.sin(th))
        cx, cy, L = 0.075, 0.90, 0.055
        for vec, lbl in ((n_vec, "N"), (e_vec, "E")):
            ax.annotate(lbl, xy=(cx, cy), xytext=(cx + vec[0] * L, cy + vec[1] * L),
                        xycoords="axes fraction", textcoords="axes fraction",
                        fontsize=9, fontweight="bold", color="w", ha="center",
                        va="center", zorder=20,
                        arrowprops=dict(arrowstyle="-", color="w", lw=1.3),
                        bbox=dict(boxstyle="circle,pad=0.15", fc="black",
                                  alpha=0.4, ec="none"))

    # ---- image flip -----------------------------------------------------------
    @staticmethod
    def _fm_flip_image(img, flip_x, flip_y):
        """Mirror a resampled sky-image array in X and/or Y. row 0 = North,
        col 0 = East (see imaging.sky_image_from_fits), so flip-X (East<->
        West) flips columns and flip-Y (North<->South) flips rows. A display
        fix for an untrustworthy WCS parity, not a re-fetch -- call on a copy
        at draw time (never mutates the cached array), so toggling is
        instant and reversible either way."""
        if flip_x:
            img = np.fliplr(img)
        if flip_y:
            img = np.flipud(img)
        return img
