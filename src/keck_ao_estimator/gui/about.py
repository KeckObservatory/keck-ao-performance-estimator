"""Bundled documentation lookup: the filenames of the shipped KAON PDFs
(reachable from the GUI's Help menu) and _bundled_doc(), which resolves one
to an absolute path from either the installed keck_ao_docs package data or
the dev source tree.
"""
import os

DOC_USER_MANUAL = "KAON_1556_keck_ao_estimator_gui_manual.pdf"
DOC_TECH_NOTE = "KAON_1542_keck_ao_performance_estimator.pdf"
DOC_BENCH_DIAGRAMS = "KAON 1488 - Keck AOB Block Diagrams.pdf"


def _bundled_doc(filename):
    """Absolute path to a bundled documentation PDF, from the installed
    keck_ao_docs package data or the dev source tree, or None if not found."""
    try:                                    # installed / on-path package data
        from importlib.resources import files
        p = files("keck_ao_docs").joinpath(filename)
        if p.is_file():
            return str(p)
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))    # dev source tree
    # dev tree: <repo_root>/src/keck_ao_estimator/gui/about.py
    #        -> <repo_root>/sr_estimator/keck_ao_docs/<filename>
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    cand = os.path.join(repo_root, "sr_estimator", "keck_ao_docs", filename)
    return os.path.abspath(cand) if os.path.isfile(cand) else None
