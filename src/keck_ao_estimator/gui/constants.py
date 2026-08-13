"""GUI-only display/scaling constants shared by MainWindow and the tab
mixins: WFE-summary term classification, field-map marker colours and
asterism geometry, science-camera FOVs, and sky-backdrop survey choices.
None of these are physics/budget values -- they never feed the engine.
"""

# Budget terms that are LGS-SPECIFIC and therefore do NOT project onto the NGS
# wavefront: focal anisoplanatism (single-beacon cone effect), sodium-layer
# focus, and the LTAO tomography residual. Everything else — DM fitting,
# bandwidth/servo-lag, scintillation, HO measurement, static/calibration,
# margin — is shared with NGS and DOES project. Two further conditions, applied
# in _ngs_added_nm(): angular anisoplanatism projects onto NGS only when the NGS
# is off-axis (ngs_offset > 0), and the per-telescope terms (DM fitting and the
# static telescope-aberration term) project only for the telescope actually in
# use (the other telescope's FITTING_ERR / STATIC_TEL is irrelevant).
NGS_LGS_ONLY_TERMS = {"FA_REF", "NAFOC", "TOMO_ERR"}

# How each adjustable budget term scales with the night's seeing (read off
# lgs_budget_terms in the engine): total-seeing terms scale ×(total/ref)^(5/6),
# free-atm terms ×(free-atm/ref)^(5/6), TOMO weakly ×(free-atm/ref)^(1/3), and
# the fixed terms not at all. Shown as per-row tags on the WFE sliders tab.
WFE_SCALING = {
    "FITTING_ERR_K1": "total seeing",
    "FITTING_ERR_K2": "total seeing",
    "BW_REF":         "total seeing (wind-wtd)",
    "SCINT_REF":      "total seeing",
    "FA_REF":         "free-atm seeing",
    "ANG_REF":        "free-atm seeing (× offset)",
    "HOMEAS":         "fixed",
    "NAFOC":          "fixed",
    "STATIC_TEL_K1":  "fixed",
    "STATIC_TEL_K2":  "fixed",
    "STATIC_CALIB":   "fixed",
    "STATIC_DM":      "fixed",
    "STATIC_INST":    "fixed",
    "STATIC_REG":     "fixed",
    "MARGIN":         "fixed",
    "TOMO_ERR":       "free-atm seeing (weak, ⅓)",
}

# Field-map marker colours and the LTAO laser-guide-star asterism geometry.
# The asterism CENTRE is the laser offset (radial at the LGS offset); the four
# 589 nm sodium beacons sit on a square of this radius about it. Anisoplanatism
# is evaluated at the asterism centre (the engine's LTAO model), so this is
# display geometry only.
FM_C_TARGET = "#1F6FE0"          # science target / field centre: blue
FM_C_LASER  = "#F5C400"          # 589 nm sodium laser / LGS: yellow
FM_C_STAR   = "#E24A00"          # NGS / TT reference star: orange-red
FM_C_FOR    = "#7A3FA0"          # field-of-regard / instrument FOV: purple
FM_C_MARKER = "#12A150"          # user-dropped science target(s): green
FM_C_CATSTAR = "#E8302A"         # catalogue guide-star candidates: red outline
                                  # (was white; red reads far better against a
                                  # loaded grayscale backdrop image)
FM_C_CATSTAR_RING = "#00E5FF"    # left-click-inspected star's ring: cyan, so
                                  # it stands out against the now-red markers
FM_C_WARN = "#D9822B"            # amber: 'verify, don't assume' warnings
FM_C_TSS  = "#0E7C7B"            # TSS reachability / vignetting overlay:
                                  # teal, unused by any other overlay so the
                                  # modelled limits never read as measured
                                  # geometry (FOR purple / laser yellow)
                                  # (e.g. an optically-reddened guide star whose
                                  # R mag was guessed from IR photometry)
LGS_ASTERISM_RADIUS_ARCSEC = 7.6
LGS_ASTERISM_PA_DEG = (0.0, 90.0, 180.0, 270.0)   # representative square

# Guide-star patrol field: a 60" RADIUS circle within which an LGS tip-tilt
# guide star can be selected. Display overlay only (no engine term).
FIELD_OF_REGARD_RADIUS_ARCSEC = 60.0

# Science-camera fields of view, per telescope/instrument, for the field map.
#   K2 = NIRC2 imager: three square FOVs.
#   K1 = OSIRIS: imager is a fixed 20x20; the spectrograph FOV is
#        (lenslet_across x plate_scale) x (64 x plate_scale) -- the user's
#        table is exactly this product, so we generate it from scale+lenslet.
NIRC2_FOVS_ARCSEC = (10.0, 20.0, 40.0)
OSIRIS_IMAGER_FOV_ARCSEC = 20.0
OSIRIS_SPEC_SCALES = (0.02, 0.035, 0.05, 0.10)     # arcsec / spaxel
OSIRIS_SPEC_LENSLETS = (16, 32, 36, 42, 45, 48)    # lenslets across (x 64 along)

# Sky-image overlay (field map): DSS/2MASS backdrops fetched from the CDS
# hips2fits service, or a local FITS. Label -> HiPS survey id.
HIPS_SURVEYS = {
    "DSS2 red":   "CDS/P/DSS2/red",
    "DSS2 color": "CDS/P/DSS2/color",
    "2MASS J":    "CDS/P/2MASS/J",
    "2MASS K":    "CDS/P/2MASS/K",
}
LOCAL_BACKDROP = "Local FITS…"  # backdrop-combo choice: a local wide image

# Field-map Conditions-combo choice tied to Nighttime mode (gui/tabs/
# nighttime.py): use the atmospheric sample nearest the last successful
# auto-pull, instead of a window/night median or a hand-picked time. Shared
# constant so nighttime.py, fieldmap_tab.py (builds the combo) and
# prediction.py (_fm_when_time, interprets the selection) all agree on the
# exact label.
NIGHTTIME_FM_COND = "time of last pull"
