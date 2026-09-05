"""
edge_fusion.py  --  EDGE / NODE ("arc-node") fusion of QA-corrected boundary graphs,
                    with SATM's intra-edge key points and curvature-adaptive sampling.

Supersedes contour_fusion.py (per-region closed-contour averaging).

WHAT THIS IS
------------
The boundary graph is already arc-node ("coverage") topology: every element carries the TWO
region codes it separates. So cross-image correspondence is GIVEN by the labels, not inferred
-- that is the structural advantage over SATM. What SATM has that the raw edge/node method
lacks is intra-edge anchors: between two junctions, a pure arc-length pointwise mean is exactly
the Karcher-mean scheme that smooths protrusions and concavities away (the paper's worst
protrusion preserver, AVG_GL -10.450). This module imports SATM's Step 1 (key points) and
Step 2 (Hungarian matching with d_max + zero padding) and applies them LOCALLY, per arc.

PIPELINE
--------
    graph  -> ARCS (maximal same-code chains between junctions) + NODES (junctions)
    NODES  -> matched across subjects by incident region set; collisions resolved by
              Hungarian assignment with a cross-image displacement gate (SATM's d_max)  [F2,F3]
    ARCS   -> matched by (code, fused end-nodes)
    per arc group:
        key points  = curvature extrema, alpha-separated, BOTH signs (MAX/MIN/CURV)     [F1]
        matched     = Hungarian on the SATM benefit matrix over (x, y, gamma*l)          [F1]
        between consecutive matched anchors:
            n       = sagitta bound over ALL subjects' sub-arcs (NOT a fixed count)      [F1a]
            reparam = (1-lam)*arclength + lam*cumulative-turning                         [F1a]
            fuse    = weighted pointwise mean
        ends snapped to the fused nodes, correction blended along the arc (follow-along)
    REBUILD -> polygonize. THE POLYGONS ARE THE ATLAS.                                   [F5]
        face -> region id by INTERSECTING THE CODES of its bounding arcs;
        anything that rule cannot decide becomes an explicit UNLABELED region, so a gap
        shows up ON the atlas instead of silently vanishing.

EVERYTHING IS N-SUBJECT.  Two subjects with weights (1-w, w) reproduce the pairwise report. [F7]

CURVATURE IS COMPUTED AT A SCALE, NOT PER-VERTEX
------------------------------------------------
This is not optional and it is not in the source docs. A raw finite-difference curvature on a
polyline flattened at FLATTEN_PX = 0.05 over-reports |kappa| by ~27x (measured: 1.350 against a
true 0.050 on a circle of r=20), and it gets WORSE the tighter you flatten. F6 (tighten
FLATTEN_PX) and F1a (allocate points by curvature) are therefore in DIRECT CONFLICT unless
x(s), y(s) are Gaussian-smoothed at curv_sigma_px before differentiating. With smoothing:
0.057 vs 0.050, and the sagitta budget lands on 24 points instead of saturating N_MAX_SEG on
every real curve. See curvature().

COORDINATES are VOXELS throughout (x = column/SI, y = row/LR), the frame boundary_graph.py
uses. The SVG display transform is applied only at read/write. NOTHING IS QUANTISED
INTERNALLY -- full float end to end; grid_write applies on write only (AFAM F6 / SATM C8).
"""
from __future__ import annotations

import json
import math
import os
import time as _time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from shapely.geometry import LinearRing, LineString, MultiLineString, Point, Polygon
from shapely.ops import linemerge, polygonize, unary_union
from shapely.strtree import STRtree

try:                                                 # shapely >= 1.8
    from shapely.algorithms.polylabel import polylabel as _polylabel
except Exception:                                    # pragma: no cover
    _polylabel = None
try:
    from scipy.interpolate import splev, splprep     # F1b spline path
    _HAVE_SPLINE = True
except Exception:                                    # pragma: no cover
    _HAVE_SPLINE = False

class TopologyDivergence(RuntimeError):
    """Raised when strict_topo is on and the subjects' region-adjacency graphs disagree. [5.5.3.1]"""

class _StageTimer:
    """Context-manager timer. `store` a dict records seconds per named stage; None disables it."""
    def __init__(self, store):
        self.store = store
    def __call__(self, name):
        self._name = name
        return self
    def __enter__(self):
        self._t0 = _time.perf_counter()
        return self
    def __exit__(self, *exc):
        if self.store is not None:
            self.store[self._name] = self.store.get(self._name, 0.0) + (_time.perf_counter() - self._t0)
        return False

# =====================================================================================
# Parameters   (AFAM Part II.  Units are VOXELS; 1 voxel == 1 SVG px in this pipeline.)   [5.1.1]
# =====================================================================================
@dataclass
class FusionParams:
    # --- representation (F1b) --------------------------------------------------------
    representation: str = "polyline"   # {"polyline","spline"}
    spline_smooth: float = 0.0         # splprep s=. 0 = interpolating (no detail loss).

    # --- sampling (F1a) -- REPLACES any fixed N_RESAMPLE / N_FUSION_SAMPLES ------------
    fit_tol_px: float = 0.05     # max chord deviation from the true arc. DRIVES the point count.
    n_min_seg: int = 2           # a straight segment needs exactly 2 points.
    n_max_seg: int = 200         # clamp. Saturation is DIAGNOSTIC -> S7_budget_saturated.
    curv_lambda: float = 0.5     # 0 = arc-length param, 1 = pure turning param.

    # --- curvature scale (REQUIRED; see module docstring) -----------------------------
    curv_sigma_px: float | None = None  # Gaussian sigma on x(s), y(s) BEFORE differentiating. This is
                                 # the SCALE at which a feature counts as a feature. Too small ->
                                 # curvature is polyline noise. Too large -> real notches are
                                 # invisible to the detector. None -> min(1.0, kp_alpha/4), which
                                 # is the only safe coupling: the curvature scale must be FINER
                                 # than the detail scale kp_alpha claims to preserve.
    curv_sample_px: float = 0.25  # spacing of the throwaway dense working copy.

    # --- key points (F1, SATM Step 1) -------------------------------------------------
    keypoints: bool = True       # False -> nodes are the only anchors (Karcher behaviour)
    kp_alpha: float = 8.0        # px. Min spacing between key points. SETS THE SCALE OF
                                 # PRESERVED DETAIL. Tune this FIRST (AFAM selection order):
                                 # measure the smallest notch you refuse to lose, set alpha below it.
    kp_curv_thresh: float = 0.05  # 1/px. |kappa| floor for a key point. BOTH signs are kept:
                                  # +ve = convex feature, -ve = concavity (SATM's CURV, which
                                  # Fig. 1b shows is REQUIRED -- MAX+MIN alone smooth concavities
                                  # away, and an atlas is all notches, clefts and sulci).
    kp_gamma: float = 1.0 / 3.0  # weight on the relative arc-length coordinate. Paper's best.
    kp_l_scale: float | None = None
    # ^ px scale on the FRACTIONAL l coordinate, so that gamma*l is commensurable with x and y.
    #   AFAM writes "gamma * l" with l fractional in [0,1]; at gamma = 1/3 that term spans
    #   0..0.33 px against a kp_dmax of 5 px -- i.e. numerically inert, and gamma would do
    #   nothing. None -> scale by the mean arc length (the term then spans 0..L/3 px and
    #   actually disambiguates adjacent features, which is its stated purpose).
    #   Set 1.0 for AFAM's literal formula.
    kp_dmax: float = 5.0         # px. Max key-point displacement for a match (SATM's d_max).
                                 # Zero-padded assignment: further apart than this => REJECTED,
                                 # not force-matched. This is the mechanism that makes a feature
                                 # present in only one tracing harmless instead of corrupting.

    # --- correspondence (F2, F3) ------------------------------------------------------
    node_tol: float = 1.0        # WITHIN one subject: endpoints this close are one junction.
    node_match_max: float = 5.0  # ACROSS subjects: refuse to pair junctions further apart than
                                 # this. Set from measurement: ~3x S3_node_disp_p95_px.
    strict_topo: bool = True     # abort where the region-adjacency graphs diverge (SATM C2)
    strict_topo_min: float = 0.98
    # --- phantom regions (partial-coverage topology alignment; Situation 2 vs 3) -------
    phantom_regions: bool = True   # a region present in some subjects but not all, whose
                                 # absence in the others reduces to a POINT (its neighbours
                                 # meet at one junction there, or it is an island) or a single
                                 # LINE (it sits on one shared border there), is given a
                                 # ZERO-AREA phantom in the subjects that lack it, so all
                                 # subjects share ONE topology and it fuses to its coverage
                                 # fraction (k/n of full area) instead of hard-stopping. An
                                 # absence that does NOT so reduce stays a Situation-3 abort.
    phantom_eps_px: float = 1e-3   # length given to a POINT phantom so its arcs survive dedup
                                 # (a LINE phantom already has length along the border it rides).
    fragment_match_tol_px: float = 50.0   # canonicalization: max centroid distance to match a
                                          # subject fragment onto a canonical fragment
    phantom_window_margin_px: float = 30.0  # phantom lookups search only this far around the
                                            # template fragment (safety net for residual
                                            # duplicate wall/junction codes, e.g. lenses)
    label_all_fragments: bool = True      # QA labels: every fragment (True) or only the
                                          # largest fragment per base region (False)
    halt_on_situation3: bool = True   # a partial region that cannot reduce to a LINE, POINT
                                      # or CLUSTER locus stops the slice, regardless of
                                      # strict_topo. Averaging across a topology change gives
                                      # a locally wrong map with no warning. Set False only
                                      # to survey how often Situation 3 occurs.

    # --- policy (F4) ------------------------------------------------------------------
    orphan_policy: str = "passthrough"   # {"passthrough","drop","reference"}
    #   passthrough : keep an arc seen by >= min_arc_support subjects, average those. Asserts
    #                 the tracer who DREW it was right. Right default for an ATLAS.
    #   drop        : discard any arc not seen by ALL subjects (SATM's behaviour). Asserts the
    #                 tracer who OMITTED it was right. Right for a CONSENSUS task. Note this
    #                 opens the region -> it becomes an UNLABELED face, visibly, by design.
    #   reference   : take the reference subject's version verbatim; drop if the ref lacks it.
    min_arc_support: int = 1     # passthrough only. len(subjects) = unanimity; ceil(N/2) = vote.
    reference_sid: str | None = None   # None -> the subject with the most arcs

    # --- closed loops (islands) ------------------------------------------------------- [C5]
    loop_align: str = "fft"      # {"fft","coarse"}. fft = exact global optimum over all N cyclic
                                 # shifts in O(N log N). coarse = the old n//50 stepped search,
                                 # kept as a toggle for contours where the cheaper, stickier
                                 # behaviour is wanted. FFT is the default.
    loop_coarse_steps: int = 50

    # --- rebuild (F5 / C3): THE POLYGONS ARE THE ATLAS ---------------------------------
    outer_code: int = 0          # DB09 background. No real region has id 0, so no collision.
    unlabeled_id_start: int = 8001   # a face the code-set rule cannot decide becomes an explicit
                                     # UNLABELED region from here up, so a QA gap appears ON the
                                     # atlas instead of vanishing into the unbounded face.
    new_id_start: int = 9001     # a region the EXPERT drew (closed path, no parseable id).
    node_arcs: bool = True       # unary_union the arcs before polygonize, so crossing arcs get
                                 # split at their crossings. polygonize REQUIRES a NODED input:
                                 # without this a self-intersecting fused arc silently loses faces.
    repair_dangling: bool = True   # snap a dangling arc end onto the nearest node within
    dangle_repair_px: float = 1.5  # this distance, so a small QA gap does not open a region.
    min_face_area: float = 0.0   # drop faces below this (0 = keep all). Slivers are diagnostic.

    build_polygons: bool = False  # False -> STOP at the fused LINES. rebuild_atlas/polygonize,
                                 # region labelling, S5/S6/S7 and the registry update are all
                                 # skipped. Output has arcs+graph but regions={} (see run_fusion).
    diag_ignore_unlabeled: bool = False  # S6/S7 ONLY: exclude unlabeled (8001+) and expert
                                 # (9001+) ids from the comparative shape metrics, so a QA gap
                                 # you left on purpose does not dominate every worst-region line.
                                 # The atlas, registry and S1..S5 still see every face.

    # --- precision (F6 / SATM C8): NOTHING is quantised internally ---------------------
    flatten_px: float = 0.05     # POLYLINE PATH ONLY. Bezier flattening. MUST be <= fit_tol_px.
                                 # UNUSED when representation == "spline" (AFAM F6: the spline
                                 # IS the curve, so there is no flattening step and no parameter).
    grid_write: float | None = 0.1   # quantisation ON WRITE ONLY. None = full float.
    svg_decimals: int = 4

    # --- metrics ----------------------------------------------------------------------
    # MASTER SWITCH for the comparative (subject-vs-fused) diagnostics S6 + S7. When False,
    # run_fusion SKIPS _subject_regions (the per-subject atlas rebuild) and both diag_quality
    # and diag_shape entirely -- i.e. it does ONE rebuild instead of (1 + n_subjects), and does
    # no per-region IoU/Hausdorff/curvature work. The atlas itself (regions, anchors, faces) and
    # the cheap pipeline-correctness checks S1..S5 are ALWAYS produced. Turn this on when you want
    # to KNOW HOW GOOD the fused shape is; leave it off for a fast QA-geometry pass.
    run_comparative_diag: bool = True
    # The three EXPENSIVE per-region metrics, each independently switchable (only consulted when
    # run_comparative_diag is True). The other S7 metrics -- peri_dev (M2), round_dev (M1),
    # topo_delta (M6) -- are ~free once the shared unions are computed, so they are not toggle-able.
    metrics_avg_gl: bool = True   # M3' protrusion preservation. ~1s/120 regions (convex-hull dist).
    metrics_curv_ks: bool = True  # M7 curvature distribution KS. ~0.6s/120 regions.
    metrics_skele: bool = False   # M4 skeleton/hull ratio. Needs skimage + rasterisation.

    def __post_init__(self):
        if self.curv_sigma_px is None:
            self.curv_sigma_px = min(1.0, self.kp_alpha / 4.0)
        if self.flatten_px > self.fit_tol_px:
            warnings.warn(
                f"flatten_px ({self.flatten_px}) > fit_tol_px ({self.fit_tol_px}). You would be "
                f"destroying the curve at {self.flatten_px}px, then finding key points on the "
                f"wreckage, then carefully sampling that wreckage to {self.fit_tol_px}px. AFAM F6.")
        if self.curv_sigma_px > self.kp_alpha / 4.0:
            warnings.warn(
                f"curv_sigma_px ({self.curv_sigma_px}) > kp_alpha/4 ({self.kp_alpha / 4:.2f}). The "
                f"curvature SCALE is coarser than the detail you claim to preserve: features below "
                f"~{self.curv_sigma_px}px are invisible to the key-point detector.")
        if self.representation == "spline" and not _HAVE_SPLINE:
            raise ImportError("representation='spline' needs scipy.interpolate.splprep")
        if self.orphan_policy not in ("passthrough", "drop", "reference"):
            raise ValueError(f"orphan_policy must be passthrough|drop|reference, "
                             f"got {self.orphan_policy!r}")
        if self.loop_align not in ("fft", "coarse"):
            raise ValueError(f"loop_align must be fft|coarse, got {self.loop_align!r}")

# =====================================================================================
# Data model   [5.2.2]
# =====================================================================================
@dataclass
class Arc:
    """One maximal boundary chain of constant code (a, b) between two junction nodes.
    A closed island loop has closed=True, n0 == n1 == None, and does not repeat its first point."""
    code: tuple
    pts: np.ndarray
    closed: bool = False
    n0: int | None = None
    n1: int | None = None
    sid: str | None = None
    phantom: bool = False   # True for a zero-area region inserted to align partial coverage
    plabels: tuple | None = None   # (label0, label1) per endpoint: build_nodes groups ends
                                   # with the SAME label into one node regardless of geometry,
                                   # so coincident phantom triple-points stay DISTINCT until
                                   # after fusion. None on an end -> normal geometric clustering.
    def length(self) -> float:
        P = np.vstack([self.pts, self.pts[0]]) if self.closed else self.pts
        return float(np.hypot(*np.diff(P, axis=0).T).sum())

# =====================================================================================
# 0b.  FRAGMENT IDENTITY -- one unique code per connected piece of a region   [3.1.2, 5.1.3.2]
# =====================================================================================
# A region id may cover several disconnected pieces ("fragments"). Codes are made unique
# at the SOURCE (raster relabelling before tracing): largest piece keeps the base id,
# piece j >= 1 gets base*FRAGMENT_MULT + j. Decoding is arithmetic (no table).
# Fragments are INDEPENDENT for seeds/fusion/phantoms/registry records; ONE region for
# QA colour, names, and the finished atlas (merge + rollup helpers below).

FRAGMENT_MULT = 10000        # every base id (template, 8001+, 9001+) is below this

def fragment_base(code):
    """Base region id of a fragment code. [3.1.2.1]"""
    code = int(code)
    return code // FRAGMENT_MULT if code >= FRAGMENT_MULT else code

def fragment_index(code):
    """0 for the largest/only fragment (base code), 1,2,... [3.1.2.1]"""
    code = int(code)
    return code % FRAGMENT_MULT if code >= FRAGMENT_MULT else 0

def fragment_relabel_slice(label_slice, min_voxels=1):
    """Relabel a raster so each connected piece of each region has a unique code. [3.1.2]
    Largest piece keeps the base id; others get base*FRAGMENT_MULT + j ordered by size. Run
    AFTER hemisphere masking."""
    from scipy.ndimage import label as cc_label
    lab = np.asarray(label_slice).copy()
    for rid in [int(r) for r in np.unique(lab) if int(r) != 0]:
        comps, n = cc_label(lab == rid)
        if n <= 1:
            continue
        stats = []
        for ci in range(1, n + 1):
            m = comps == ci
            iy, ix = np.argwhere(m)[0]
            stats.append((-int(m.sum()), int(iy), int(ix), ci))
        stats.sort()
        for j, (negsz, _iy, _ix, ci) in enumerate(stats):
            if j == 0:
                continue                                   # largest keeps the base id
            if -negsz < min_voxels:
                lab[comps == ci] = 0
            else:
                lab[comps == ci] = rid * FRAGMENT_MULT + j
    return lab

def fragment_canon_table(label_slice_relabelled):
    """{base: [(canonical fragment code, centroid (x, y)), ...]} from the ALREADY relabelled
    template slice. [5.1.3.2]"""
    tab = defaultdict(list)
    lab = np.asarray(label_slice_relabelled)
    for code in [int(c) for c in np.unique(lab) if int(c) != 0]:
        ys, xs = np.nonzero(lab == code)
        tab[fragment_base(code)].append((code, (float(xs.mean()), float(ys.mean()))))
    return dict(tab)

def _graph_code_centroids(g):
    acc = defaultdict(lambda: [0.0, 0.0, 0])
    for e in g.elements:
        for c in (int(e[2]), int(e[3])):
            if c == 0:
                continue
            for ni in (int(e[0]), int(e[1])):
                x, y = float(g.nodes[ni][0]), float(g.nodes[ni][1])
                a = acc[c]
                a[0] += x; a[1] += y; a[2] += 1
    return {c: (v[0] / v[2], v[1] / v[2]) for c, v in acc.items() if v[2]}

def canonicalize_fragments(graphs, canon_table, params):
    """Remap each subject graph's fragment codes onto the canonical codes, per base region, by
    nearest-centroid assignment (near-copy assumption). [5.1.3.2]"""
    tol = float(getattr(params, "fragment_match_tol_px", 50.0))
    notes = []
    for sid, g in graphs.items():
        cents = _graph_code_centroids(g)
        by_base = defaultdict(list)
        for c in cents:
            by_base[fragment_base(c)].append(c)
        remap = {}
        for base, codes in sorted(by_base.items()):
            canon = sorted(canon_table.get(base, []), key=lambda t: fragment_index(t[0]))
            if len(codes) <= 1 and len(canon) <= 1:
                continue                                   # one piece everywhere: nothing to do
            C = sorted(codes, key=fragment_index)
            if canon:
                D = np.array([[float(np.hypot(cents[c][0] - k[1][0],
                                              cents[c][1] - k[1][1]))
                               for k in canon] for c in C])
                ri, ci = linear_sum_assignment(D)
                for r, k in zip(ri, ci):
                    if D[r, k] <= tol and C[r] != canon[k][0]:
                        remap[C[r]] = canon[k][0]
            matched = set(remap) | {c for c in C if any(c == k[0] for k in canon)}
            nxt = max([fragment_index(k[0]) for k in canon] + [0]) + 1
            for c in C:
                if c not in matched:
                    remap[c] = base * FRAGMENT_MULT + nxt   # unmatched: fresh canonical index
                    nxt += 1
        if remap:
            for gi, e in enumerate(g.elements):
                a, b = int(e[2]), int(e[3])
                na, nb = remap.get(a, a), remap.get(b, b)
                if na != a or nb != b:
                    if isinstance(e, list):
                        e[2], e[3] = na, nb
                    else:
                        g.elements[gi] = e[:2] + type(e)((na, nb)) + e[4:]
            notes.extend((sid, a, b) for a, b in sorted(remap.items()))
    return notes

def merge_fragment_regions(regions):
    """{fragment code: faces} -> {base id: faces}: the FINISHED-atlas view (one region). [6.1.5]"""
    out = defaultdict(list)
    for rid, faces in regions.items():
        out[fragment_base(rid)].extend(faces)
    return dict(out)

def filter_anchors_largest_fragment(anchors):
    """Keep ONE label anchor per BASE region: the base-coded fragment (relabelling makes that the
    largest), else the greatest-clearance anchor. [5.6.4.4]"""
    best = {}
    for a in anchors:
        b = fragment_base(a[0])
        key = (fragment_index(a[0]) != 0, -float(a[3]))
        if b not in best or key < best[b][0]:
            best[b] = (key, a)
    return [best[b][1] for b in sorted(best)]

def registry_base_status(reg, slice_index):
    """Roll per-fragment registry records up to base regions. [5.7.4.1]
    A base is inactive only when EVERY fragment of it is inactive on the slice; purge and delete
    decisions consult this, never one fragment record."""
    regs = reg.get("slices", {}).get(str(int(slice_index)), {}).get("regions", {})
    out = {}
    for rid, v in regs.items():
        b = fragment_base(int(rid))
        active = (v.get("status") == "active") or (out.get(b) == "active")
        out[b] = "active" if active else "inactive"
    return out

# =====================================================================================
# 1.  Curvature, arc length, curvature-adaptive sampling                          [F1a]
#    [5.6.3, 5.6.6.1]
# =====================================================================================
def arclength(P, closed=False):
    """Cumulative arc length along P, one value per vertex. [5.6.3]"""
    Q = np.vstack([P, P[0]]) if closed else np.asarray(P, float)
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(Q, axis=0).T))])

def _dedup(P, eps=1e-12):
    P = np.asarray(P, float)
    if len(P) < 2:
        return P
    keep = np.r_[True, (np.abs(np.diff(P, axis=0)) > eps).any(axis=1)]
    return P[keep]

def densify(P, spacing, closed=False):
    """Throwaway dense working copy at uniform arc-length spacing. [5.6.3]"""
    P = _dedup(P)
    if len(P) < 2:
        return P
    Q = np.vstack([P, P[0]]) if closed else P
    s = arclength(Q)
    if s[-1] <= 0:
        return P
    n = max(2, int(math.ceil(s[-1] / max(spacing, 1e-9))) + 1)
    t = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(t, s, Q[:, 0]), np.interp(t, s, Q[:, 1])])

def curvature(P, sigma_px=1.0, spacing=None, closed=False):
    """Discrete SIGNED curvature AT A SCALE; P must be arc-length parameterised. [5.6.3]
    x(s) and y(s) are Gaussian-smoothed at sigma_px before differentiating, without which a
    finely flattened polyline over-reports |kappa| by ~27x."""
    P = np.asarray(P, float)
    if len(P) < 5:
        return np.zeros(len(P))
    if spacing is None:
        s = arclength(P, closed)
        spacing = max(s[-1] / max(len(P) - 1, 1), 1e-9)
    sig = max(sigma_px / max(spacing, 1e-9), 0.6)
    mode = "wrap" if closed else "nearest"
    X = gaussian_filter1d(P[:, 0], sig, mode=mode)
    Y = gaussian_filter1d(P[:, 1], sig, mode=mode)
    Q = np.column_stack([X, Y])
    d1 = np.gradient(Q, axis=0)
    d2 = np.gradient(d1, axis=0)
    num = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    den = (d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5 + 1e-12
    return num / den

def max_deviation(P, R):
    """Max distance from the vertices of P to the polyline R. [5.6.6.1.1]"""
    P = np.asarray(P, float); R = np.asarray(R, float)
    if len(R) < 2 or len(P) == 0:
        return 0.0
    A, B = R[:-1], R[1:]
    AB = B - A
    L2 = (AB ** 2).sum(1) + 1e-12
    AP = P[:, None, :] - A[None]
    t = np.clip((AP * AB[None]).sum(2) / L2[None], 0.0, 1.0)
    proj = A[None] + t[..., None] * AB[None]
    d = np.linalg.norm(P[:, None, :] - proj, axis=2)
    return float(d.min(axis=1).max())

def segment_budget(sub_arcs, tol_px, n_min, n_max, sigma_px, spacing, lam=0.5):
    """F1a sagitta bound: chords needed to stay within tol_px of the true arc. [5.6.6.1.1]
    m = ceil(L / sqrt(8 * tol * r)), clamped to [n_min_seg, n_max_seg]; hitting the clamp is
    reported as S7_budget_saturated."""
    n = n_min
    for P in sub_arcs:
        P = _dedup(P)
        if len(P) < 2:
            continue
        L = float(arclength(P)[-1])
        if L <= 0:
            continue
        k = np.abs(curvature(P, sigma_px, spacing))
        kmax = float(k.max()) if len(k) else 0.0
        m = 2 if kmax < 1e-9 else int(math.ceil(L / math.sqrt(8.0 * tol_px * (1.0 / kmax))))
        m = int(np.clip(m, n_min, n_max))
        while m < n_max:                                # verify, then grow
            R = resample_by_curvature(P, m, lam, sigma_px, spacing, abs_k=k)  # reuse k (same P)
            if max_deviation(P, R) <= tol_px:
                break
            m = min(n_max, int(m * 1.6) + 1)
        n = max(n, m)
    return int(np.clip(n, n_min, n_max))

def resample_by_curvature(P, n, lam=0.5, sigma_px=1.0, spacing=None, closed=False, abs_k=None):
    """F1a reparameterisation, sampling s' uniformly. [5.6.6.2]
    s' = (1-lam)*arclength + lam*cumulative|curvature|."""
    P = _dedup(P)
    if len(P) == 0:
        return np.zeros((n, 2))
    if len(P) < 2:
        return np.repeat(P[:1], n, axis=0)
    Q = np.vstack([P, P[0]]) if closed else P
    s = arclength(Q)
    if s[-1] <= 0:
        return np.repeat(Q[:1], n, axis=0)
    # abs_k: |curvature| for THESE points, if the caller already computed it (segment_budget does).
    # Valid only when it matches Q's length; otherwise recompute. Saves a gaussian_filter1d + two
    # np.gradient passes per resample, which the verify-and-grow loop calls repeatedly.
    if abs_k is not None and len(abs_k) == len(Q):
        k = abs_k
    else:
        k = np.abs(curvature(Q, sigma_px, spacing, closed))
    ck = np.concatenate([[0.0], np.cumsum(k[:-1] * np.diff(s))])
    if lam <= 0 or ck[-1] < 1e-9:                       # <- the guard
        u = s / s[-1]
    else:
        u = (1 - lam) * s / s[-1] + lam * ck / ck[-1]
    u = np.maximum.accumulate(u)
    u = u + np.linspace(0.0, 1e-9, len(u))              # strict monotonicity for np.interp
    u = (u - u[0]) / max(u[-1] - u[0], 1e-12)
    t = np.linspace(0.0, 1.0, n)
    return np.column_stack([np.interp(t, u, Q[:, 0]), np.interp(t, u, Q[:, 1])])

def resample_open(pts, n):
    """Resample an open polyline to n points, evenly by arc length. [5.6.6.2]"""
    P = _dedup(pts)
    if len(P) < 2:
        return np.repeat(np.asarray(pts, float)[:1], n, axis=0)
    s = arclength(P)
    if s[-1] <= 0:
        return np.repeat(P[:1], n, axis=0)
    t = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])])

def resample_closed(pts, n):
    """Resample a closed contour to n points, evenly by arc length. [5.6.6.2]"""
    P = _dedup(pts)
    if len(P) < 2:
        return np.repeat(np.asarray(pts, float)[:1], n, axis=0)
    Q = np.vstack([P, P[0]]) if np.hypot(*(P[0] - P[-1])) > 1e-12 else P
    s = arclength(Q)
    if s[-1] <= 0:
        return np.repeat(Q[:1], n, axis=0)
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, s, Q[:, 0]), np.interp(t, s, Q[:, 1])])

def signed_area(P):
    """Shoelace area of a closed contour; the sign gives its winding. [5.6.2.1]"""
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

# =====================================================================================
# 2.  Arc representation: polyline (F1a) or spline (F1b), behind one interface   [5.6.3.1]
# =====================================================================================
class ArcRep:
    """A curve that can be EVALUATED at any fractional position u in [0,1], and whose curvature can
    be asked for at those positions. [5.6.3.1]"""

    def __init__(self, P, params: FusionParams, closed=False):
        self.closed = closed
        self.params = params
        P = _dedup(P)
        if closed and len(P) > 2 and np.hypot(*(P[0] - P[-1])) < 1e-12:
            P = P[:-1]
        self.raw = P
        self.fit_residual = 0.0
        self.dense = densify(P, params.curv_sample_px, closed)
        self._D = np.vstack([self.dense, self.dense[0]]) if closed else self.dense
        s = arclength(self._D)
        self._u = s / max(s[-1], 1e-12)
        self._len = float(s[-1])
        self._k = None

    # -- evaluation -------------------------------------------------------------------
    def eval(self, u):
        u = np.atleast_1d(np.asarray(u, float))
        return np.column_stack([np.interp(u, self._u, self._D[:, 0]),
                                np.interp(u, self._u, self._D[:, 1])])

    def curvature_at(self, u):
        u = np.atleast_1d(np.asarray(u, float))
        if self._k is None:
            self._k = curvature(self._D, self.params.curv_sigma_px,
                                self.params.curv_sample_px, self.closed)
        return np.interp(u, self._u, self._k)

    def length(self):
        return self._len

    def sub_raw(self, u0, u1, m=64):
        """The piece between u0 and u1 as a dense polyline (used for the sagitta budget)."""
        m = max(m, int(abs(u1 - u0) * len(self._D)) + 2)
        return self.eval(np.linspace(u0, u1, m))

    def sub(self, u0, u1, n, lam):
        """The piece between u0 and u1, resampled to n points with F1a's curvature reparam.
        On the spline path the segment is fit as a cubic spline with anchor endpoints, so
        corners survive."""
        P = self.sub_raw(u0, u1, 4 * max(n, 8))
        if self.params.representation == "spline" and len(P) >= 5:
            try:
                tck, _u = splprep([P[:, 0], P[:, 1]], s=self.params.spline_smooth, k=3)
                t = np.linspace(0.0, 1.0, max(8 * n, 64))
                x, y = splev(t, tck)
                E = np.column_stack([x, y])
                self.fit_residual = max(self.fit_residual,
                                        float(cKDTree(E).query(P)[0].max()))   # -> S7_fit_residual
                P = E
            except Exception as e:                                             # never crash
                warnings.warn(f"spline fit failed on a segment ({e}); using the polyline")
        return resample_by_curvature(P, n, lam, self.params.curv_sigma_px,
                                     self.params.curv_sample_px)

# =====================================================================================
# 3.  Key points (SATM Step 1) and their matching (SATM Step 2)                   [F1]   [5.6.4]
# =====================================================================================
def edge_keypoints(rep: ArcRep, params: FusionParams):
    """Local |curvature| maxima, at least kp_alpha apart, above kp_curv_thresh. [5.6.4.1]
    Returns fractional positions u in (0,1), sorted."""
    if not params.keypoints:
        return []
    L = rep.length()
    if L < 2 * params.kp_alpha:
        return []                                       # too short to hold an interior anchor
    n = max(32, int(L / max(params.curv_sample_px, 1e-6)))
    u = np.linspace(0.0, 1.0, n)
    a = np.abs(rep.curvature_at(u))

    interior = np.arange(1, n - 1)
    loc = interior[(a[1:-1] >= a[:-2]) & (a[1:-1] >= a[2:])]
    cand = [int(i) for i in loc if a[i] >= params.kp_curv_thresh]

    margin = params.kp_alpha / L                        # keep key points off the node anchors
    keep = []
    for i in sorted(cand, key=lambda j: -a[j]):         # strongest first: SATM's greedy alpha rule
        if u[i] < margin or u[i] > 1.0 - margin:
            continue
        if all(abs(u[i] - u[j]) * L >= params.kp_alpha for j in keep):
            keep.append(i)
    return sorted(float(u[i]) for i in keep)

def match_keypoints(repA, uA, repB, uB, params: FusionParams, l_scale=None):
    """SATM Step 2, applied per arc. [5.6.4.2]
    Returns [(uA_k, uB_r), ...], sorted, strictly monotone in BOTH arcs."""
    if not uA or not uB:
        return [], 0
    if l_scale is None:
        l_scale = params.kp_l_scale
    if l_scale is None:
        l_scale = 0.5 * (repA.length() + repB.length())

    PA, PB = repA.eval(np.array(uA)), repB.eval(np.array(uB))
    g = params.kp_gamma * float(l_scale)
    A3 = np.column_stack([PA, g * np.asarray(uA, float)])
    B3 = np.column_stack([PB, g * np.asarray(uB, float)])

    D = np.linalg.norm(A3[:, None, :] - B3[None, :, :], axis=2)
    M, R = len(uA), len(uB)
    Ben = np.zeros((M + R, M + R))
    Ben[:M, :R] = np.maximum(0.0, params.kp_dmax - D)   # SATM's benefit matrix, zero-padded
    r, c = linear_sum_assignment(-Ben)

    pairs = sorted((uA[i], uB[j]) for i, j in zip(r, c)
                   if i < M and j < R and Ben[i, j] > 0.0)
    out, crossed = [], 0
    for ua, ub in pairs:                                # a crossing match would invert a segment
        if out and (ua <= out[-1][0] or ub <= out[-1][1]):
            crossed += 1
            continue
        out.append((ua, ub))
    return out, crossed

def anchor_table(reps, ref, params: FusionParams):
    """N-subject generalisation of SATM's pairwise key-point matching. [5.6.4.3]
    Returns ({sid: [0.0, ..., 1.0]}, n_anchors, n_dropped), same length for every subject; an
    anchor non-monotone in any subject is dropped (S7_kp_nonmonotone)."""
    sids = list(reps)
    uref = edge_keypoints(reps[ref], params)
    if not uref or len(sids) == 1:
        return {s: [0.0, 1.0] for s in sids}, 0, 0

    matched = {ref: {u: u for u in uref}}
    dropped = 0
    for s in sids:
        if s == ref:
            continue
        us = edge_keypoints(reps[s], params)
        pairs, crossed = match_keypoints(reps[ref], uref, reps[s], us, params)
        dropped += crossed
        matched[s] = dict(pairs)

    keep = [u for u in uref if sum(1 for s in sids if u in matched[s]) >= 2]
    if not keep:
        return {s: [0.0, 1.0] for s in sids}, 0, dropped

    table = {}
    for s in sids:
        m = matched[s]
        ku = [0.0] + [u for u in keep if u in m] + [1.0]
        kv = [0.0] + [m[u] for u in keep if u in m] + [1.0]
        table[s] = [0.0] + [float(np.interp(u, ku, kv)) for u in keep] + [1.0]

    ok = [i for i in range(1, len(keep) + 1)
          if all(table[s][i] > table[s][i - 1] + 1e-6 and table[s][i] < table[s][i + 1] - 1e-6
                 for s in sids)]
    dropped += len(keep) - len(ok)
    for s in sids:
        table[s] = [0.0] + [table[s][i] for i in ok] + [1.0]
    return table, len(ok), dropped

# =====================================================================================
# 4.  Graph <-> arcs   [5.2.2, 5.6.7]
# =====================================================================================
def _code_of(e):
    a, b = int(e[2]), int(e[3])
    return (min(a, b), max(a, b))

def graph_to_arcs(g, sid=None):
    """Split a BoundaryGraph into ARCS: maximal chains of same-code elements between JUNCTION
    nodes. [5.2.2]"""
    nodes = np.asarray(g.nodes, float)
    inc = defaultdict(list)
    for ei, e in enumerate(g.elements):
        inc[int(e[0])].append(ei)
        inc[int(e[1])].append(ei)

    def is_junction(n):
        eis = inc[n]
        if len(eis) != 2:
            return True
        return _code_of(g.elements[eis[0]]) != _code_of(g.elements[eis[1]])

    junc = {n for n in inc if is_junction(n)}
    used, arcs = set(), []

    def walk(start_node, start_e, stop_at_junction):
        path, cur_n, cur_e = [start_node], start_node, start_e
        while True:
            used.add(cur_e)
            e = g.elements[cur_e]
            nxt = int(e[1]) if int(e[0]) == cur_n else int(e[0])
            path.append(nxt)
            if (stop_at_junction and nxt in junc) or nxt == start_node:
                break
            cand = [k for k in inc[nxt] if k != cur_e and k not in used]
            if not cand:
                break                                   # dangling end -> S2 flags it
            cur_e, cur_n = cand[0], nxt
        return path

    for j in sorted(junc):
        for ei in list(inc[j]):
            if ei in used:
                continue
            path = walk(j, ei, stop_at_junction=True)
            closed = (path[0] == path[-1] and len(path) > 3)
            pts = nodes[path[:-1]] if closed else nodes[path]
            arcs.append(Arc(_code_of(g.elements[ei]), pts, closed, sid=sid))

    for ei, e in enumerate(g.elements):                 # leftovers: node-free island loops
        if ei in used:
            continue
        path = walk(int(e[0]), ei, stop_at_junction=False)
        closed = (path[0] == path[-1] and len(path) > 3)
        pts = nodes[path[:-1]] if closed else nodes[path]
        arcs.append(Arc(_code_of(e), pts, closed, sid=sid))

    return [a for a in arcs if len(_dedup(a.pts)) >= 2]

def arcs_to_graph(arcs, weld=1e-9):
    """Arcs -> BoundaryGraph. [5.6.7]"""
    import boundary_graph as bg
    g = bg.BoundaryGraph(nodes=np.empty((0, 2)), elements=[])
    index, xy = {}, []

    def nid(p):
        k = (round(float(p[0]) / weld), round(float(p[1]) / weld))
        if k not in index:
            index[k] = len(xy)
            xy.append((float(p[0]), float(p[1])))
        return index[k]

    for arc in arcs:
        a, b = int(arc.code[0]), int(arc.code[1])
        P = np.vstack([arc.pts, arc.pts[0]]) if arc.closed else arc.pts
        prev = None
        for p in P:
            i = nid(p)
            if prev is not None and prev != i:
                g.elements.append([prev, i, a, b])
            prev = i
    g.nodes = np.asarray(xy, float) if xy else np.zeros((0, 2))
    return g

def arcs_to_node_graph(arcs, weld=1e-9, verify=True):
    """Fused ARCS -> a COMPLETE node/element PSLG: EVERY polyline vertex becomes a node, every
    segment becomes one element carrying both region codes. [5.2.2]"""
    import boundary_graph as bg
    g = bg.BoundaryGraph(nodes=np.empty((0, 2)), elements=[])
    index, xy = {}, []

    def nid(p):
        k = (round(float(p[0]) / weld), round(float(p[1]) / weld))
        if k not in index:
            index[k] = len(xy)
            xy.append((float(p[0]), float(p[1])))
        return index[k]

    for arc in arcs:
        a, b = int(arc.code[0]), int(arc.code[1])
        P = np.vstack([arc.pts, arc.pts[0]]) if arc.closed else arc.pts
        prev = None
        for p in P:
            i = nid(p)
            if prev is not None and prev != i:
                g.elements.append([prev, i, a, b])
            prev = i
    g.nodes = np.asarray(xy, float) if xy else np.zeros((0, 2))

    if verify:
        # every element separates two DIFFERENT codes; no self-loops; no duplicate segments
        seen, dup, selfpair, selfloop = set(), 0, 0, 0
        for e in g.elements:
            if e[0] == e[1]:
                selfloop += 1
            if e[2] == e[3]:
                selfpair += 1
            k = (min(e[0], e[1]), max(e[0], e[1]))
            if k in seen:
                dup += 1
            seen.add(k)
        g._node_graph_report = {"n_nodes": len(g.nodes), "n_elements": len(g.elements),
                                "dup_segments": dup, "self_pairs": selfpair,
                                "self_loops": selfloop,
                                "ok": (dup == 0 and selfloop == 0)}
    return g

# =====================================================================================
# 5.  Nodes: cluster within a subject, then match across subjects            [F2, F3]
#    [5.2.2.1, 5.5.1]
# =====================================================================================
def build_nodes(arcs, params: FusionParams):
    """Cluster the endpoints of all OPEN arcs into junction nodes (single-link at node_tol).
    [5.2.2.1]"""
    pts, ref, labels = [], [], []
    for i, arc in enumerate(arcs):
        if arc.closed:
            continue
        pl = getattr(arc, 'plabels', None)
        pts.append(arc.pts[0]);  ref.append((i, 0)); labels.append(pl[0] if pl else None)
        pts.append(arc.pts[-1]); ref.append((i, 1)); labels.append(pl[1] if pl else None)
    if not pts:
        return np.zeros((0, 2)), [], []

    P = np.asarray(pts, float)
    parent = list(range(len(P)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i, j in cKDTree(P).query_pairs(params.node_tol):
        if ref[i][0] == ref[j][0]:                      # never merge one arc's two ends
            continue
        if labels[i] is not None or labels[j] is not None:  # labeled ends do NOT cluster by
            continue                                        # geometry (superimposed phantom junctions)
        union(i, j)
    lab_groups = defaultdict(list)                      # ends sharing a label -> ONE node
    for _idx, _lab in enumerate(labels):
        if _lab is not None:
            lab_groups[_lab].append(_idx)
    for _members in lab_groups.values():
        for _m in _members[1:]:
            union(_members[0], _m)

    groups = defaultdict(list)
    for i in range(len(P)):
        groups[find(i)].append(i)

    node_xy, node_regions, node_phantom = [], [], []
    for k, root in enumerate(sorted(groups)):
        idxs = groups[root]
        node_xy.append(P[idxs].mean(axis=0))
        regs, all_phantom, has_label = set(), True, False
        for i in idxs:
            ai, end = ref[i]
            regs.update(arcs[ai].code)
            all_phantom = all_phantom and bool(getattr(arcs[ai], 'phantom', False))
            if labels[i] is not None:                   # a labeled end = a phantom triple-point
                has_label = True
            if end == 0:
                arcs[ai].n0 = k
            else:
                arcs[ai].n1 = k
        node_regions.append(frozenset(regs))
        node_phantom.append(all_phantom or has_label)
    return np.asarray(node_xy, float), node_regions, node_phantom

def match_nodes(subj_nodes, weights, params: FusionParams, subj_phantom=None):
    """Nodes are grouped by their incident REGION SET -- a topological key that is GIVEN by the
    labels, where SATM has to infer correspondence from geometry. [5.5.1]"""
    sids = list(subj_nodes)
    per, keys = {}, set()
    for sid, (xy, regs) in subj_nodes.items():
        d = defaultdict(list)
        for i, k in enumerate(regs):
            d[k].append(i)
        per[sid] = d
        keys |= set(d)

    fused_xy, node_map = [], {sid: {} for sid in sids}
    # region ids present in only ONE subject -> their junctions are 'isolating', not conflicts
    _rid_subjects = defaultdict(set)
    for sid, (xy, regs) in subj_nodes.items():
        for rset in regs:
            for r in rset:
                _rid_subjects[int(r)].add(sid)
    _single_subject_rids = {r for r, ss in _rid_subjects.items() if len(ss) == 1}

    def _is_isolating(key):
        # every non-background code at this junction belongs to a single-subject region
        return all((int(r) == int(params.outer_code)) or (int(r) in _single_subject_rids)
                   for r in key)

    stats = {"collision_keys": [], "unmatched": [], "disp": [], "rejected_far": [],
             "unmatched_isolating": [], "rejected_isolating": [],
             "single_subject_rids": sorted(_single_subject_rids)}

    for key in sorted(keys, key=lambda s: tuple(sorted(s))):
        counts = {sid: len(per[sid].get(key, [])) for sid in sids}
        if max(counts.values()) > 1:
            stats["collision_keys"].append((tuple(sorted(key)), max(counts.values())))
        order = sorted(sids, key=lambda s: -counts[s])
        slots = []

        for sid in order:
            idx = per[sid].get(key, [])
            if not idx:
                if _is_isolating(key):
                    stats["unmatched_isolating"].append((sid, tuple(sorted(key))))
                else:
                    stats["unmatched"].append((sid, tuple(sorted(key)))) 
                continue
            B = subj_nodes[sid][0][idx]
            if not slots:
                slots = [{sid: i} for i in idx]
                continue
            A = np.array([_slot_pos(s, subj_nodes, weights) for s in slots])
            C = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
            r, c = linear_sum_assignment(C)
            taken = set()
            for ri, ci in zip(r, c):
                # phantom nodes are DEFINED to sit at a collapsed locus, so they are legitimately
                # far from the real boundary they align to -> exempt them from the displacement gate.
                _cand_phantom = bool(subj_phantom and subj_phantom.get(sid, [False]*99999)[idx[ci]])
                _slot_phantom = any(bool(subj_phantom and subj_phantom.get(s2, [False]*99999)[i2])
                                    for s2, i2 in slots[ri].items()) if subj_phantom else False
                if C[ri, ci] > params.node_match_max and not (_cand_phantom or _slot_phantom):  # [F2]
                    _rec = (sid, tuple(sorted(key)), round(float(C[ri, ci]), 2))
                    if _is_isolating(key):
                        stats["rejected_isolating"].append(_rec)
                    else:
                        stats["rejected_far"].append(_rec)
                    continue
                slots[ri][sid] = idx[ci]
                taken.add(ci)
                stats["disp"].append(float(C[ri, ci]))
            for ci in range(len(idx)):
                if ci not in taken:
                    slots.append({sid: idx[ci]})

        for slot in slots:
            fi = len(fused_xy)
            fused_xy.append(_slot_pos(slot, subj_nodes, weights))
            for sid, i in slot.items():
                node_map[sid][i] = fi

    return np.asarray(fused_xy, float), node_map, stats

def _slot_pos(slot, subj_nodes, weights):
    w = np.array([weights[sid] for sid in slot], float)
    w = w / w.sum()
    P = np.array([subj_nodes[sid][0][i] for sid, i in slot.items()], float)
    return (w[:, None] * P).sum(axis=0)

# =====================================================================================
# 5b.  PHANTOM REGIONS -- partial-coverage topology alignment      [Situation 2 vs 3]   [5.3]
# =====================================================================================
# Identity is the FRAGMENT CODE (section 0b): unique per connected piece, canonical
# across subjects. Per fragment code absent from some subjects:
#   point      -- island (closed loop, <=1 real neighbour): tiny copy at the seed
#   line       -- lens (exactly 2 neighbours): collapse onto the host border
#   tree/<n>   -- 3+ neighbours: pin junctions via spokes [Thm 1], route the boundary
#                 along the tree of new arcs [Thm 2], absorb the <n> diagonal arcs.

def _present_region_ids(arcs):
    """Set of region ids that appear on any arc code in this subject (excludes outer/background
    only if you ask; here it keeps everything, callers filter)."""
    s = set()
    for a in arcs:
        s.update(int(c) for c in a.code)
    return s

def _project_to_polyline(P, q):
    """(t_arclen, foot_xy) of the closest point on open polyline P to point q. Used to place a
    phantom's endpoints onto an existing border of the subject that lacks the region."""
    P = np.asarray(P, float); q = np.asarray(q, float)
    best_d, best_foot, best_t, acc = np.inf, P[0], 0.0, 0.0
    for i in range(len(P) - 1):
        a, b = P[i], P[i + 1]
        ab = b - a; L2 = float(ab @ ab)
        u = 0.0 if L2 == 0 else min(1.0, max(0.0, float((q - a) @ ab) / L2))
        foot = a + u * ab
        d = float(np.hypot(*(q - foot)))
        seglen = float(np.hypot(*ab))
        if d < best_d:
            best_d, best_foot, best_t = d, foot, acc + u * seglen
        acc += seglen
    return best_t, best_foot

def _split_polyline_at(P, feet):
    """Split open polyline P at interior points feet=[(t_arclen, foot_xy), ...]; each foot
    becomes a shared endpoint. Returns the ordered list of sub-polylines (each >=2 pts)."""
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(np.asarray(P, float), axis=0).T))]
    cuts = sorted(feet, key=lambda f: f[0])
    P = np.asarray(P, float)
    pieces, cur, ci = [], [P[0]], 0
    for i in range(1, len(P)):
        while ci < len(cuts) and d[i - 1] <= cuts[ci][0] <= d[i] + 1e-12:
            foot = np.asarray(cuts[ci][1], float)
            if np.hypot(*(foot - cur[-1])) > 1e-9:
                cur.append(foot)
            pieces.append(np.asarray(cur, float)); cur = [foot]; ci += 1
        if np.hypot(*(P[i] - cur[-1])) > 1e-9:
            cur.append(P[i])
    pieces.append(np.asarray(cur, float))
    return [p for p in pieces if len(p) >= 2]

def _arcs_rep_bbox(arcsC):
    P = np.vstack([np.asarray(a.pts, float) for a in arcsC])
    return P.mean(axis=0), (P.min(axis=0), P.max(axis=0))

def _in_window(xy, bbox, margin):
    lo, hi = bbox
    return (lo[0] - margin <= xy[0] <= hi[0] + margin and
            lo[1] - margin <= xy[1] <= hi[1] + margin)

def _fragment_template(all_arcs, node_xy, node_regs, R, params):
    """From a subject that HAS fragment R: ordered boundary cycle, cyclic neighbours,
    junction positions, spoke codes, far-end keys. Returns (template, None) or (None, why)."""
    R = int(R)
    Rarcs = [a for a in all_arcs if R in (int(a.code[0]), int(a.code[1]))]
    if not Rarcs:
        return None, f"fragment {R} has no arcs"
    rep, bbox = _arcs_rep_bbox(Rarcs)
    neigh = set()
    for a in Rarcs:
        neigh.update(int(c) for c in a.code if int(c) != R)
    open_c = [a for a in Rarcs if not a.closed]
    if not open_c:                                             # island fragment
        return {"kind": "island", "arcs": Rarcs, "neigh": neigh,
                "rep": rep, "bbox": bbox}, None
    byn = defaultdict(list)
    for a in open_c:
        if a.n0 is None or a.n1 is None:
            return None, f"fragment {R} boundary arc missing a node"
        byn[a.n0].append(a); byn[a.n1].append(a)
    if any(len(v) != 2 for v in byn.values()) or len(byn) != len(open_c):
        return None, f"fragment {R} boundary is not a single cycle (relabelling gap?)"
    cyc, vlist = [open_c[0]], []
    cur, node, start = open_c[0], open_c[0].n1, open_c[0].n0
    while True:                                                # walk the closed chain
        vlist.append(node)
        if node == start:
            break
        nxt = [a for a in byn[node] if a is not cur][0]
        cyc.append(nxt); cur = nxt
        node = nxt.n1 if nxt.n0 == node else nxt.n0
    k = len(cyc)                                               # arc i runs v[i-1] -> v[i]
    N = [next(int(c) for c in a.code if int(c) != R) for a in cyc]
    base = {"kind": "cycle", "arcs": Rarcs, "cycle": cyc, "N": N, "k": k,
            "neigh": neigh, "rep": rep, "bbox": bbox,
            "v_xy": [np.asarray(node_xy[v], float) for v in vlist]}
    if len(neigh) == 2:                                        # lens: no spokes to build
        base["spoke"] = None; base["far_key"] = None
        return base, None
    inc = defaultdict(list)
    for a in all_arcs:
        if not a.closed:
            if a.n0 is not None: inc[a.n0].append(a)
            if a.n1 is not None: inc[a.n1].append(a)
    spoke, far_key = [], []
    for i, v in enumerate(vlist):
        others = [a for a in inc[v] if a is not cyc[i] and a is not cyc[(i + 1) % k]]
        if len(others) != 1:                                   # B1: junction not trivalent
            return None, f"junction of fragment {R} has degree {2 + len(others)} (B1)"
        sp = others[0]
        pair = tuple(sorted((N[i], N[(i + 1) % k])))
        if tuple(sorted(int(c) for c in sp.code)) != pair:
            return None, f"arc at junction of fragment {R} is not the expected wall {pair}"
        far = sp.n1 if sp.n0 == v else sp.n0
        spoke.append(pair)
        far_key.append(frozenset(int(x) for x in node_regs[far]))
    base["spoke"] = spoke; base["far_key"] = far_key
    return base, None

def _classify_fragment(tmpl, params):
    outer = int(params.outer_code)
    if tmpl["kind"] == "island":
        if len(tmpl["neigh"] - {outer}) <= 1:
            return "point"
        return "complex: island fragment with 2+ neighbours"
    return "line" if len(tmpl["neigh"]) == 2 else "tree"

def _inject_point_phantom(R, s, tmpl, seed, subj_arcs, params):
    c = np.asarray(tmpl["rep"] if seed is None else seed, float)
    eps = float(params.phantom_eps_px)
    allp = np.vstack([np.asarray(a.pts, float) for a in tmpl["arcs"]])
    ctr = allp.mean(axis=0)
    span = max(float(np.hypot(*(allp.max(0) - allp.min(0)))), 1e-9)
    for a in tmpl["arcs"]:
        q = c + (np.asarray(a.pts, float) - ctr) * (eps / span)
        subj_arcs[s].append(Arc(tuple(int(x) for x in a.code), q,
                                closed=bool(a.closed), phantom=True))
    return tuple(float(x) for x in c), None

def _inject_line_phantom(R, s, tmpl, subj_arcs, params):
    pair = tuple(sorted(int(x) for x in tmpl["neigh"]))
    m = float(params.phantom_window_margin_px)
    hosts = [a for a in subj_arcs[s] if not a.closed
             and tuple(sorted(int(c) for c in a.code)) == pair
             and _in_window(np.asarray(a.pts, float).mean(axis=0), tmpl["bbox"], m)]
    if len(hosts) != 1:
        return None, f"host border {pair} not unique in window ({len(hosts)} found)"
    H = hosts[0]
    feet = [_project_to_polyline(H.pts, q) for q in tmpl["v_xy"][:2]]
    ts = sorted(f[0] for f in feet)
    if not (1e-6 < ts[0] and ts[-1] < H.length() - 1e-6 and ts[-1] - ts[0] > 1e-6):
        return None, f"feet of fragment {R} not interior to host border {pair}"
    pieces = _split_polyline_at(H.pts, feet)
    subj_arcs[s].remove(H)
    for p in pieces:
        subj_arcs[s].append(Arc(tuple(int(c) for c in H.code), p, closed=False))
    f0, f1 = sorted(feet, key=lambda f: f[0])
    seg = np.vstack([np.asarray(f0[1], float), np.asarray(f1[1], float)])
    for a in tmpl["arcs"]:
        if not a.closed:
            subj_arcs[s].append(Arc(tuple(int(c) for c in a.code), seg.copy(),
                                    closed=False, phantom=True))
    return tuple(float(x) for x in seg.mean(axis=0)), None

def _tree_reduction(R, tmpl, s, subj_arcs, node_xy, node_regs, params):
    """Pin R's junctions in subject s [Thm 1] and route its boundary along the tree of new arcs
    [Thm 2].
    Returns ((t_nodes, paths, NEW, spoke_ends), None) or (None, reason)."""
    k, m = tmpl["k"], float(params.phantom_window_margin_px)
    neigh = set(int(x) for x in tmpl["neigh"])
    Sarcs = [a for a in subj_arcs[s] if not a.closed]
    t_nodes, spoke_ends, taken = [], [], set()
    for i in range(k):                                   # -- 1. pinning [Thm 1; B2 tiebreak]
        pair, vxy, fk = tmpl["spoke"][i], tmpl["v_xy"][i], tmpl["far_key"][i]
        cands = [a for a in Sarcs
                 if tuple(sorted(int(c) for c in a.code)) == pair
                 and (_in_window(node_xy[a.n0], tmpl["bbox"], m)
                      or _in_window(node_xy[a.n1], tmpl["bbox"], m))]
        if not cands:
            return None, f"spoke {pair} of fragment {R} missing in window"
        a = min(cands, key=lambda a: min(
            np.hypot(*(np.asarray(node_xy[a.n0], float) - vxy)),
            np.hypot(*(np.asarray(node_xy[a.n1], float) - vxy))))
        k0 = frozenset(int(x) for x in node_regs[a.n0])
        k1 = frozenset(int(x) for x in node_regs[a.n1])
        if (k0 == fk) != (k1 == fk):
            t = a.n1 if k0 == fk else a.n0               # far end identified by region set
        else:                                            # duplicate far keys: nearest wins
            d0 = np.hypot(*(np.asarray(node_xy[a.n0], float) - vxy))
            d1 = np.hypot(*(np.asarray(node_xy[a.n1], float) - vxy))
            t = a.n0 if d0 <= d1 else a.n1
        end = 0 if t == a.n0 else 1
        if (id(a), end) in taken:
            return None, f"spoke end contested at junction {i} of fragment {R}"
        taken.add((id(a), end))
        spoke_ends.append((a, end))
        t_nodes.append(t)
    spoke_ids = {id(x[0]) for x in spoke_ends}           # -- 2. the NEW arcs [Thm 2a-b]
    NEW = []
    for a in Sarcs:
        if id(a) in spoke_ids:
            continue
        if not set(int(c) for c in a.code) <= neigh:
            continue
        if not (_in_window(node_xy[a.n0], tmpl["bbox"], m)
                and _in_window(node_xy[a.n1], tmpl["bbox"], m)):
            continue
        if not (set(int(x) for x in node_regs[a.n0]) <= neigh
                and set(int(x) for x in node_regs[a.n1]) <= neigh):
            continue
        NEW.append(a)
    J = set(t_nodes)
    for a in NEW:
        J.add(a.n0); J.add(a.n1)
    if len(J) != len(NEW) + 1:                           # -- 3. tree + obstruction checks
        return None, (f"new arcs around fragment {R} are not a tree "
                      f"({len(NEW)} arcs, {len(J)} junctions)")
    adj = defaultdict(list)
    for a in NEW:
        adj[a.n0].append((a.n1, a)); adj[a.n1].append((a.n0, a))
    seen, stack = {next(iter(J))}, [next(iter(J))]
    while stack:
        u = stack.pop()
        for vtx, _a in adj[u]:
            if vtx not in seen:
                seen.add(vtx); stack.append(vtx)
    if seen != J:
        return None, f"new arcs around fragment {R} are disconnected from a pin"
    new_ids = {id(a) for a in NEW}
    for a in Sarcs:                                      # foreign arc at a pinned junction:
        if id(a) in new_ids or id(a) in spoke_ids:       # catches T1 flips (C1) and adjacent
            continue                                     # missing clusters (B3) in one net
        if a.n0 in J or a.n1 in J:
            return None, (f"foreign arc {tuple(sorted(int(c) for c in a.code))} ends at "
                          f"a pinned junction of fragment {R}")
    paths, use = [], defaultdict(int)                    # -- 4. paths [Thm 2d] + cover [2e]
    for i in range(k):
        a0, b0 = t_nodes[i - 1], t_nodes[i]
        prev, q = {a0: None}, [a0]
        while q:
            u = q.pop(0)
            if u == b0:
                break
            for vtx, arc in adj[u]:
                if vtx not in prev:
                    prev[vtx] = (u, arc); q.append(vtx)
        if b0 not in prev:
            return None, f"no tree path for side {i} of fragment {R}"
        seq, u = [], b0
        while prev[u] is not None:
            pu, arc = prev[u]
            seq.append((pu, u, arc)); u = pu
        seq.reverse()
        for _pu, _u, arc in seq:
            use[id(arc)] += 1
        paths.append(seq)
    if any(use[id(a)] != 2 for a in NEW):
        return None, f"double-cover check failed around fragment {R}"
    return (t_nodes, paths, NEW, spoke_ends), None

def _inject_tree_phantom(R, s, tmpl, t_nodes, paths, NEW, spoke_ends,
                         subj_arcs, node_xy, params):
    """Insert R's zero-area boundary along the tree, relabel spoke near-ends, absorb
    the diagonals. Returns (xy, number of diagonals)."""
    k = tmpl["k"]
    lab = [f"phT{int(R)}_{s}_{i}" for i in range(k)]
    for i, (a, end) in enumerate(spoke_ends):            # generalised wall relabelling
        pl = list(a.plabels) if getattr(a, "plabels", None) else [None, None]
        pl[end] = lab[i]
        a.plabels = tuple(pl)
    for i in range(k):                                   # boundary arc i: v[i-1] -> v[i]
        if not paths[i]:
            p = np.asarray(node_xy[t_nodes[i]], float)
            pts = np.vstack([p, p])                      # zero-length arc
        else:
            pts = None
            for (u, _vtx, arc) in paths[i]:
                q = np.asarray(arc.pts, float)
                if arc.n0 != u:
                    q = q[::-1]
                pts = q if pts is None else np.vstack([pts, q[1:]])
        ph = Arc(tuple(int(c) for c in tmpl["cycle"][i].code), pts,
                 closed=False, phantom=True)
        ph.plabels = (lab[i - 1], lab[i])
        subj_arcs[s].append(ph)
    new_ids = {id(a) for a in NEW}                       # ABSORPTION [decision A4]
    subj_arcs[s][:] = [a for a in subj_arcs[s] if id(a) not in new_ids]
    return tuple(float(x) for x in node_xy[t_nodes[0]]), len(NEW)

def inject_phantom_regions(subj_arcs, subj_xy, subj_regs, seeds, params):
    """Fragment-code-scoped phantom injection (Situation 2); mutates subj_arcs in place. [5.3]
    Records: inserted [(rid, sid, kind, xy)] with kind in {point, line, tree/<n>diag}; complex_
    [(rid, sid, reason, xy)]."""
    if not params.phantom_regions:
        return [], []
    sids = list(subj_arcs)
    present = {s: _present_region_ids(subj_arcs[s]) for s in sids}
    outer = int(params.outer_code)
    all_ids = set().union(*present.values()) - {outer}
    seed_xy = {int(r): np.asarray(pt, float) for pt, r in seeds}

    inserted, complex_ = [], []
    for R in sorted(all_ids):
        haves = [s for s in sids if R in present[s]]
        lacks = [s for s in sids if R not in present[s]]
        if not lacks or not haves:
            continue
        ref = haves[0]
        if getattr(params, "reference_sid", None) in haves:
            ref = params.reference_sid
        tmpl, why = _fragment_template(subj_arcs[ref], subj_xy[ref], subj_regs[ref],
                                       R, params)
        if tmpl is None:
            complex_.append((R, ref, why, (float("nan"),) * 2))
            continue
        agree = True                                     # have-subjects agree [decision A8]
        for s2 in haves[1:]:
            t2, _w = _fragment_template(subj_arcs[s2], subj_xy[s2], subj_regs[s2],
                                        R, params)
            if t2 is None or set(t2["neigh"]) != set(tmpl["neigh"]):
                complex_.append((R, s2, f"subjects disagree on neighbours of fragment {R}",
                                 tuple(float(x) for x in tmpl["rep"])))
                agree = False
                break
        if not agree:
            continue
        kind = _classify_fragment(tmpl, params)
        for s in lacks:
            if any(int(nb) != outer and int(nb) not in present[s]
                   for nb in tmpl["neigh"]):             # adjacent missing cluster [A7/B3]
                complex_.append((R, s, f"fragment {R} borders a region absent in subject "
                                 f"{s} (adjacent missing cluster, B3)",
                                 tuple(float(x) for x in tmpl["rep"])))
                continue
            if kind.startswith("complex"):
                complex_.append((R, s, kind[len("complex: "):],
                                 tuple(float(x) for x in tmpl["rep"])))
                continue
            if kind == "point":
                xy, why = _inject_point_phantom(R, s, tmpl, seed_xy.get(int(R)),
                                                subj_arcs, params)
            elif kind == "line":
                xy, why = _inject_line_phantom(R, s, tmpl, subj_arcs, params)
            else:
                res, why = _tree_reduction(R, tmpl, s, subj_arcs, subj_xy[s],
                                           subj_regs[s], params)
                if res is not None:
                    t_nodes, paths, NEW, spoke_ends = res
                    xy, nd = _inject_tree_phantom(R, s, tmpl, t_nodes, paths, NEW,
                                                  spoke_ends, subj_arcs, subj_xy[s],
                                                  params)
                    inserted.append((R, s, f"tree/{nd}diag", xy))
                    continue
                xy = None
            if xy is not None:
                inserted.append((R, s, kind, xy))
            else:
                complex_.append((R, s, why, tuple(float(x) for x in tmpl["rep"])))
    return inserted, complex_

def _acr(rid, ctx):
    """Region acronym for reports, falling back to the numeric id when no LUT is supplied."""
    if ctx and ctx.get("acronyms"):
        return ctx["acronyms"].get(int(rid), str(int(rid)))
    return str(int(rid))

def _ctx_prefix(ctx):
    """'[run_label | slice N] ' prefix for report lines ('' when no context supplied)."""
    if not ctx:
        return ""
    parts = []
    if ctx.get("run_label"):
        parts.append(str(ctx["run_label"]))
    if ctx.get("slice_index") is not None:
        parts.append(f"slice {ctx['slice_index']}")
    return ("[" + " | ".join(parts) + "] ") if parts else ""

# =====================================================================================
# 6.  Closed-loop (island) alignment                                            [C5]   [5.6.2.1]
# =====================================================================================
def fft_align_closed(A, B):
    """Cyclic-shift + winding alignment of closed contour B onto A in O(N log N). [5.6.2.1]"""
    zA = A[:, 0] + 1j * A[:, 1]
    best = None
    for Bo in (B, B[::-1]):
        zB = Bo[:, 0] + 1j * Bo[:, 1]
        corr = np.fft.ifft(np.conj(np.fft.fft(zA)) * np.fft.fft(zB))
        k = int(np.argmax(corr.real))
        aligned = np.roll(Bo, -k, axis=0)
        ssd = float(np.sum((A - aligned) ** 2))
        if best is None or ssd < best[0]:
            best = (ssd, aligned)
    return best[1]

def coarse_align_closed(A, B, steps=50):
    """The cheaper stepped search (the notebook's original scheme). [5.6.2.1]"""
    n = len(B)
    step = max(1, n // max(steps, 1))
    best = None
    for Bo in (B, B[::-1]):
        for k in range(0, n, step):
            aligned = np.roll(Bo, -k, axis=0)
            ssd = float(np.sum((A - aligned) ** 2))
            if best is None or ssd < best[0]:
                best = (ssd, aligned)
    return best[1]

def align_closed(A, B, params: FusionParams):
    """Dispatch closed-contour alignment to the fft or coarse method. [5.6.2.1]"""
    if params.loop_align == "coarse":
        return coarse_align_closed(A, B, params.loop_coarse_steps)
    return fft_align_closed(A, B)

# =====================================================================================
# 7.  Arc matching + arc fusion (SATM anchors + F1a sampling)   [5.6.1, 5.6.6]
# =====================================================================================
def _arc_key(arc, nmap):
    if arc.closed or arc.n0 is None or arc.n1 is None:
        return (arc.code, "loop")
    a, b = nmap[arc.n0], nmap[arc.n1]
    return (arc.code, min(a, b), max(a, b))

def _arc_dist(a1, a2, n=24):
    if a1.closed != a2.closed:
        return float("inf")
    if a1.closed:
        A, B = resample_closed(a1.pts, n), resample_closed(a2.pts, n)
        return float(np.hypot(*(A.mean(0) - B.mean(0))))
    A, B = resample_open(a1.pts, n), resample_open(a2.pts, n)
    return float(min(np.abs(A - B).sum(), np.abs(A - B[::-1]).sum()) / n)

def match_arcs(subj_arcs, node_map):
    """Group arcs across subjects by (code, fused end-nodes). [5.6.1.2]"""
    per, keys = {}, set()
    for sid, arcs in subj_arcs.items():
        d = defaultdict(list)
        for arc in arcs:
            d[_arc_key(arc, node_map[sid])].append(arc)
        per[sid] = d
        keys |= set(d)

    groups = []
    for key in keys:
        counts = {sid: len(per[sid].get(key, [])) for sid in subj_arcs}
        order = sorted(subj_arcs, key=lambda s: -counts[s])
        slots = []
        for sid in order:
            cand = per[sid].get(key, [])
            if not cand:
                continue
            if not slots:
                slots = [{sid: a} for a in cand]
                continue
            C = np.array([[_arc_dist(next(iter(s.values())), a) for a in cand] for s in slots])
            r, c = linear_sum_assignment(C)
            taken = set()
            for ri, ci in zip(r, c):
                slots[ri][sid] = cand[ci]
                taken.add(ci)
            for ci in range(len(cand)):
                if ci not in taken:
                    slots.append({sid: cand[ci]})
        for s in slots:
            groups.append((key, s))
    return groups

def _blank_stat(support, dropped=False):
    return {"support": support, "snap": 0.0, "saturated": 0, "n_anchors": 0, "kp_dropped": 0,
            "fit_residual": 0.0, "dir_gap": 1.0, "dropped": dropped, "n_pts": 0,
            "len_F": 0.0, "len_blend": 0.0}

def fuse_arc_group(key, group, fused_xy, node_map, weights, params: FusionParams, ref_sid):
    """Average one arc across the subjects that have it -- WITH SATM's intra-edge anchors. [5.6.6]"""
    sids = sorted(group, key=lambda s: (s != ref_sid, s))            # reference first
    ref = sids[0]
    w = np.array([weights[s] for s in sids], float)
    w = w / w.sum()
    arcs = [group[s] for s in sids]
    st = _blank_stat(len(sids))

    if key[1] == "loop":
        Ls = [a.length() for a in arcs]
        n0 = int(np.clip(round(float(np.average(Ls, weights=w)) /
                               max(params.curv_sample_px, 1e-6)), 16, 8192))
        R = resample_closed(arcs[0].pts, n0)
        if signed_area(R) < 0:
            R = R[::-1]
        rolled = [R] + [align_closed(R, resample_closed(a.pts, n0), params) for a in arcs[1:]]
        reps = {s: ArcRep(P, params, closed=False) for s, P in zip(sids, rolled)}
    else:
        nA, nB = key[1], key[2]
        oriented, gaps = [], []
        for s, a in zip(sids, arcs):
            P = np.asarray(a.pts, float)
            if node_map[s][a.n0] != nA:                 # arc runs nB -> nA; flip it
                P = P[::-1]
            if oriented:
                A0, B0 = resample_open(oriented[0], 32), resample_open(P, 32)
                f = float(np.sum((A0 - B0) ** 2))
                r = float(np.sum((A0 - B0[::-1]) ** 2))
                gaps.append(abs(f - r) / max(f, r, 1e-9))
            oriented.append(P)
        st["dir_gap"] = float(min(gaps)) if gaps else 1.0
        reps = {s: ArcRep(P, params, closed=False) for s, P in zip(sids, oriented)}

    st["fit_residual"] = max((r.fit_residual for r in reps.values()), default=0.0)
    st["len_blend"] = float(np.average([reps[s].length() for s in sids], weights=w))

    # ---- SATM key points -> anchor table  [F1] ---------------------------------------
    table, n_anch, dropped = anchor_table(reps, ref, params)
    st["n_anchors"], st["kp_dropped"] = n_anch, dropped

    # ---- piecewise fusion between anchors  [F1a] -------------------------------------
    # ---- piecewise fusion between anchors  [F1a] -------------------------------------
    # NOTE (ordering): this runs INSIDE fuse_arc_group, which is called from fuse_graphs AFTER
    # inject_phantom_regions. So every sample-point budget/resample below sees the FINAL arc set
    # (phantoms included). A phantom LINE arc traces the seam and has real length, so it is sampled
    # and averaged pointwise against R's real arc -- pulling R's boundary smoothly onto the seam
    pieces = []
    for i in range(len(table[ref]) - 1):
        subs = [reps[s].sub_raw(table[s][i], table[s][i + 1]) for s in sids]
        n = segment_budget(subs, params.fit_tol_px, params.n_min_seg, params.n_max_seg,
                           params.curv_sigma_px, params.curv_sample_px, params.curv_lambda)
        if n >= params.n_max_seg:
            st["saturated"] += 1                        # -> S7_budget_saturated
        S = np.stack([reps[s].sub(table[s][i], table[s][i + 1], n, params.curv_lambda)
                      for s in sids])
        pieces.append((w[:, None, None] * S).sum(axis=0))

    F = pieces[0]
    for p in pieces[1:]:
        F = np.vstack([F, p[1:]])                       # dedup_join: drop the repeated anchor
    F = _dedup(F)

    if key[1] == "loop":
        if len(F) > 3 and np.hypot(*(F[0] - F[-1])) < 1e-9:
            F = F[:-1]
        out = Arc(key[0], F, closed=True)
        st["n_pts"], st["len_F"] = len(F), out.length()
        return out, st

    nA, nB = key[1], key[2]
    dA = fused_xy[nA] - F[0]
    dB = fused_xy[nB] - F[-1]
    t = arclength(F)
    t = t / t[-1] if t[-1] > 0 else np.zeros(len(F))
    F = F + (1 - t)[:, None] * dA + t[:, None] * dB     # snap ends + drag the body smoothly
    F[0], F[-1] = fused_xy[nA], fused_xy[nB]            # exact, so arcs_to_graph welds them
    out = Arc(key[0], F, closed=False, n0=nA, n1=nB)
    st["snap"] = float(max(np.hypot(*dA), np.hypot(*dB)))
    st["n_pts"], st["len_F"] = len(F), out.length()
    return out, st

# =====================================================================================
# 8.  The fusion driver                                                        [F4, F7]   [5.5-5.6]
# =====================================================================================
@dataclass
class FusionResult:
    """Everything one fused slice produces: arcs, graph, nodes, regions, stats."""
    arcs: list = field(default_factory=list)
    graph: object = None
    fused_xy: np.ndarray = None
    subj_arcs: dict = field(default_factory=dict)
    subj_nodes: dict = field(default_factory=dict)
    node_map: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

def fuse_graphs(graphs, weights=None, params: FusionParams = None, seeds=None, report_ctx=None):
    """Fuse one slice. [5.5-5.6]
    graphs {sid: BoundaryGraph}, weights {sid: float} or None for equal. N-subject throughout
    [F7]; two subjects at (1-w, w) reproduce the pairwise report."""
    params = params or FusionParams()
    sids = list(graphs)
    if not sids:
        raise ValueError("fuse_graphs: no subjects")
    if weights is None:
        weights = {s: 1.0 / len(sids) for s in sids}
    else:
        tot = float(sum(weights[s] for s in sids))
        if tot <= 0:
            raise ValueError("fuse_graphs: weights sum to zero")
        weights = {s: float(weights[s]) / tot for s in sids}

    subj_arcs = {sid: graph_to_arcs(graphs[sid], sid=sid) for sid in sids}

    # ---- PHANTOM REGIONS: align partial coverage BEFORE node matching [Situation 2] ----
    # First-pass nodes only CLASSIFY each missing region's locus (point/line/complex). The
    # injection then mutates subj_arcs (adds zero-area phantoms; splits a host border for the
    # scratch on the second pass, so the throwaway first-pass indices do not leak.
    pre_xy, pre_regs = {}, {}
    for sid in sids:
        xy0, regs0, _ = build_nodes(subj_arcs[sid], params)
        pre_xy[sid], pre_regs[sid] = xy0, regs0
    phantom_inserted, phantom_complex, phantom_independent = inject_phantom_regions(
        subj_arcs, pre_xy, pre_regs, seeds or [], params)

    subj_nodes, subj_phantom = {}, {}
    for sid in sids:
        xy, regs, phan = build_nodes(subj_arcs[sid], params)
        subj_nodes[sid], subj_phantom[sid] = (xy, regs), phan

    # ---- REFERENCE SUBJECT selection -- AFTER phantom injection so node counts are current ----
    # The reference is the arc/anchor orientation baseline (fuse_arc_group orders it first) and
    # the fallback the orphan_policy='reference' uses. It is chosen PER SLICE by this ladder:
    #   0. MANUAL override: params.reference_sid set to a real sid wins outright. (The TEST
    #      notebook exposes this as a per-slice toggle; None/'AUTO' -> the auto ladder below.)
    #   1. the subject with the MOST NODES after injection (only Situation-1 coverage moves this,
    _manual = params.reference_sid
    if _manual is not None and str(_manual).upper() != "AUTO":
        if _manual not in sids:
            raise KeyError(f"reference_sid={_manual!r} is not one of {sids}")
        ref_sid = _manual
    else:
        _nodes = {sid: len(subj_nodes[sid][0]) for sid in sids}
        _phantoms = {sid: 0 for sid in sids}
        for (_R, _s, _kind, _xy) in phantom_inserted:
            if _s in _phantoms:
                _phantoms[_s] += 1
        # sort key: most nodes (-count), then fewest phantoms (+count), then first sid (stable)
        if phantom_independent:
            print(f"  note: {len(phantom_independent)} independent region(s) "
              f"{[_acr(r, report_ctx) for r in phantom_independent[:8]]} carried through at "
              f"full geometry (Situation 1: no shared boundary to pin a phantom to).")
        ref_sid = min(sids, key=lambda s: (-_nodes[s], _phantoms[s], sids.index(s)))

    fused_xy, node_map, nstats = match_nodes(subj_nodes, weights, params, subj_phantom)

    n_multi = sum(1 for fi in range(len(fused_xy))
                  if sum(1 for s in node_map if fi in node_map[s].values()) > 1)
    # no junction nodes at all (e.g. a slice of only closed island loops) means there is nothing
    # for the subjects to disagree on at a junction -> vacuously matched, not 0% matched.
    match_rate = (n_multi / len(fused_xy)) if len(fused_xy) else 1.0

    # ---- PHANTOM insertion notes (Situation 2: auto-inserted, carried through) ----------
    for (_R, _s, _kind, _xy) in phantom_inserted:
        print("  " + _ctx_prefix(report_ctx) +
              f"PHANTOM region {_acr(_R, report_ctx)} inserted for subject {_s} as a {_kind} "
              f"at ({float(_xy[0]):.1f}, {float(_xy[1]):.1f}): absent here but present in another "
              f"subject; fused to its coverage fraction (k/n area), not hard-stopped.")

    # ---- STRICT_TOPO gate (SATM C2: a drop is a HARD STOP, not a warning) --------------
    # A junction isolating a region exclusive to one subject is coverage the other lacks
    # (Situation 1) -- carried through. A partial region whose absence reduced to a point/line
    # was phantomised above (Situation 2). What remains is Situation 3: a partial region that
    # could NOT be so reduced (phantom_complex), or a genuine conflict on a SHARED boundary.
    # Both abort, and both are reported with the SAME detail (run prefix / slice / acronyms /
    n_conflict = len(nstats["rejected_far"]) + len(nstats["unmatched"])
    n_isolating = len(nstats["rejected_isolating"]) + len(nstats["unmatched_isolating"])
    if len(sids) > 1 and (phantom_complex or n_conflict):
        lines = ["TOPOLOGY DIVERGENCE (Situation 3): the region-adjacency graphs disagree in a "
                 "way that is NOT a clean partial-coverage difference, so averaging is unsafe here."]
        if phantom_complex:
            lines.append(f"  {len(phantom_complex)} region absence(s) could not be reduced to a "
                         "single point or border:")
            for (_R, _s, _reason, _xy) in phantom_complex:
                lines.append(f"    - region {_acr(_R, report_ctx)} in subject {_s} "
                             f"@({float(_xy[0]):.1f}, {float(_xy[1]):.1f}): {_reason}")
        if n_conflict:
            lines.append(f"  S3_node_match_rate = {match_rate:.3f} (diagnostic only); "
                         f"{len(nstats['rejected_far'])} conflicting junction(s) beyond "
                         f"node_match_max = {params.node_match_max}px; "
                         f"{len(nstats['unmatched'])} unmatched; {n_isolating} isolating KEPT.")
            for (_s, _key, _d) in nstats["rejected_far"][:6]:
                lines.append(f"    - rejected-far subject {_s} regions "
                             f"{[_acr(_r, report_ctx) for _r in _key]} at {float(_d):.1f}px")
            for (_s, _key) in nstats["unmatched"][:6]:
                lines.append(f"    - unmatched subject {_s} regions "
                             f"{[_acr(_r, report_ctx) for _r in _key]}")
        lines.append("  Fix the QA, raise node_match_max, or set strict_topo=False to proceed.")
        msg = _ctx_prefix(report_ctx) + chr(10).join(lines)
        if phantom_complex and params.halt_on_situation3:
            raise TopologyDivergence(msg)     # Situation 3 is never downgraded to a warning
        if params.strict_topo:
            raise TopologyDivergence(msg)
        warnings.warn(msg)
    elif len(sids) > 1 and n_isolating:
        print(f"  note: {n_isolating} isolating junction(s) from {len(nstats['single_subject_rids'])} "
              f"single-subject region(s) carried through (coverage the other subject lacks, "
              f"not a topology conflict).")

    groups = match_arcs(subj_arcs, node_map)
    n_sub = len(sids)
    fused_arcs, astats = [], []

    for key, group in groups:
        if len(group) < n_sub:                          # ---- ORPHAN POLICY  [F4] --------
            if params.orphan_policy == "drop":
                astats.append(_blank_stat(len(group), dropped=True))
                continue
            if params.orphan_policy == "reference":
                if ref_sid not in group:
                    astats.append(_blank_stat(len(group), dropped=True))
                    continue
                group = {ref_sid: group[ref_sid]}
            elif len(group) < params.min_arc_support:
                astats.append(_blank_stat(len(group), dropped=True))
                continue

        arc, st = fuse_arc_group(key, group, fused_xy, node_map, weights, params, ref_sid)
        fused_arcs.append(arc)                          # NOT rounded: full float internally
        astats.append(st)

    return FusionResult(arcs=fused_arcs, graph=arcs_to_graph(fused_arcs), fused_xy=fused_xy,
                        subj_arcs=subj_arcs, subj_nodes=subj_nodes, node_map=node_map,
                        stats={"nodes": nstats, "arcs": astats, "n_subjects": n_sub,
                               "weights": weights, "ref_sid": ref_sid,
                               "node_match_rate": match_rate})

# =====================================================================================
# 9.  REBUILD -- THE POLYGONS ARE THE ATLAS                                 [F5 / C3]   [5.7]
# =====================================================================================
# The fused ARCS are the line network; the FACES are the deliverable, because the next step is
# a volumetric model and a filled polygon rasterises directly.
#
# Face -> region id by INTERSECTING THE CODES OF THE ARCS ON ITS BOUNDARY: a face bounded by
# (3,5), (3,8), (3,9) belongs to region 3, because 3 is the only code common to all of them.
#
def _arc_line(arc):
    P = np.vstack([arc.pts, arc.pts[0]]) if arc.closed else arc.pts
    P = _dedup(P)
    if len(P) < 2 or not np.isfinite(P).all():
        return None
    return LineString([tuple(p) for p in P])

def prepare_arcs(arcs, params: FusionParams):
    """P1/P2/P4/P5: clean the arc set and report exactly what was wrong with it. [5.7.1]"""
    rep = {"dropped_degenerate": 0, "dropped_duplicate": 0,
           "dangling_repaired": [], "dangling_unrepaired": []}
    clean, seen = [], set()
    for a in arcs:
        P = _dedup(a.pts)
        if len(P) < 2 or not np.isfinite(P).all():
            rep["dropped_degenerate"] += 1
            continue
        key = (a.code, a.closed, hash(np.round(P, 6).tobytes()))
        if key in seen:
            rep["dropped_duplicate"] += 1
            continue
        seen.add(key)
        clean.append(Arc(a.code, P, a.closed, a.n0, a.n1, a.sid))

    ends, owner = [], []
    for i, a in enumerate(clean):
        if a.closed:
            continue
        ends.append(a.pts[0]);  owner.append((i, 0))
        ends.append(a.pts[-1]); owner.append((i, 1))
    if not ends:
        return clean, rep

    E = np.asarray(ends, float)
    tree = cKDTree(E)
    deg = np.array([len(tree.query_ball_point(p, 1e-6)) for p in E])
    for j in np.where(deg == 1)[0]:
        i, which = owner[j]
        cand = [k for k in tree.query_ball_point(E[j], params.dangle_repair_px)
                if owner[k][0] != i]
        if params.repair_dangling and cand:
            k = min(cand, key=lambda k: np.hypot(*(E[k] - E[j])))
            d = float(np.hypot(*(E[k] - E[j])))
            P = clean[i].pts.copy()
            P[0 if which == 0 else -1] = E[k]
            clean[i] = Arc(clean[i].code, P, clean[i].closed,
                           clean[i].n0, clean[i].n1, clean[i].sid)
            E[j] = E[k]
            rep["dangling_repaired"].append((clean[i].code, round(d, 3)))
        else:
            rep["dangling_unrepaired"].append(
                (clean[i].code, tuple(np.round(E[j], 2))))
    return clean, rep

def polygonize_arcs(arcs, params: FusionParams):
    """P3/P6/P7/P8. [5.7.2]"""
    rep = {"arc_crossings": 0, "noded": False, "sliver_faces": 0, "polygonize_error": None}
    lines = [ln for ln in (_arc_line(a) for a in arcs) if ln is not None]
    if not lines:
        return [], [], rep

    # P3: COUNT interior crossings for the S5_arc_crossings diagnostic. unary_union below does the
    # actual noding; this loop only reports how many crossings there were. An STRtree prunes it to
    # bbox-overlapping pairs -- identical count, ~20x fewer predicate calls than the old all-pairs
    # double loop (which was O(arcs^2) and a measured hotspot).
    try:
        _tree = STRtree(lines)
        _seen = 0
        for i, ln in enumerate(lines):
            for j in _tree.query(ln):
                j = int(j)
                if j > i and lines[i].crosses(lines[j]):
                    _seen += 1
        rep["arc_crossings"] = _seen
    except Exception:                                    # STRtree unavailable -> exact fallback
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                if lines[i].crosses(lines[j]):
                    rep["arc_crossings"] += 1

    try:
        src = lines
        if params.node_arcs:
            u = unary_union(lines)                       # P3: node the arrangement
            src = list(u.geoms) if isinstance(u, MultiLineString) else [u]
            rep["noded"] = True
        faces = list(polygonize(src))
    except Exception as e:                               # P7
        rep["polygonize_error"] = f"{type(e).__name__}: {e}"
        try:
            faces = list(polygonize(lines))
        except Exception as e2:
            rep["polygonize_error"] += f" | retry: {type(e2).__name__}: {e2}"
            return [], lines, rep

    if params.min_face_area > 0:                         # P8
        keep = [f for f in faces if f.area >= params.min_face_area]
        rep["sliver_faces"] = len(faces) - len(keep)
        faces = keep
    return faces, lines, rep

def chainer_fallback(arcs, params: FusionParams):
    """P7 last resort: if polygonize cannot produce faces at all, rebuild one ring per region by
    linemerging that region's arcs. [5.7.2.4]"""
    by_region = defaultdict(list)
    for a in arcs:
        ln = _arc_line(a)
        if ln is None:
            continue
        for c in a.code:
            if c != params.outer_code:
                by_region[int(c)].append(ln)
    out = {}
    for rid, lns in by_region.items():
        try:
            m = linemerge(lns)
            geoms = list(m.geoms) if hasattr(m, "geoms") else [m]
            polys = []
            for gm in geoms:
                cs = list(gm.coords)
                if len(cs) >= 4:
                    if cs[0] != cs[-1]:
                        cs.append(cs[0])
                    p = Polygon(cs)
                    if not p.is_valid:
                        p = p.buffer(0)
                    if p.area > 0:
                        polys.append(p)
            if polys:
                out[rid] = [max(polys, key=lambda p: p.area)]
        except Exception:
            continue
    return out

def assign_faces(faces, arcs, seeds, params: FusionParams, next_unlabeled=None, next_new=None):
    """Face -> region id, in four passes ordered by how much each assumes. [5.7.3]
    Code-set intersection, then the neighbour rule, then the distance-transform seed, then an
    explicit UNLABELED id."""
    OUT = int(params.outer_code)
    seed_pts = [(Point(float(xy[0]), float(xy[1])), int(rid)) for xy, rid in seeds]
    arc_faces = [0] * len(arcs)
    lines = [_arc_line(a) for a in arcs]

    # --- candidate finding via STRtree (was two O(faces x arcs) / O(faces x seeds) loops) -----
    # The exact predicates below are UNCHANGED (bnd.intersection(ln).length >= 0.5*ln.length for
    # pruned to those whose bounding boxes overlap, which is what makes it fast. STRtree.query
    # returns POSITIONAL indices into the list it was built from.
    on_per_face = [[] for _ in faces]
    arc_to_face = defaultdict(list)
    ftree = STRtree(faces) if faces else None
    for i, ln in enumerate(lines):
        if ln is None:
            continue
        cand = (int(fi) for fi in ftree.query(ln)) if ftree is not None else range(len(faces))
        for fi in cand:
            try:
                if faces[fi].boundary.intersection(ln).length >= 0.5 * ln.length:
                    on_per_face[fi].append(i)
            except Exception:
                continue
    face_seeds = [[] for _ in faces]
    for pt, rid in seed_pts:
        cand = (int(fi) for fi in ftree.query(pt)) if ftree is not None else range(len(faces))
        for fi in cand:
            if faces[fi].contains(pt):
                face_seeds[fi].append(rid)

    info = []
    for fi, face in enumerate(faces):
        on = sorted(on_per_face[fi])              # arc-index order, matching the old 0..n loop
        for i in on:
            arc_faces[i] += 1
            arc_to_face[i].append(fi)
        codes = None
        for i in on:
            c = {int(x) for x in arcs[i].code}
            codes = c if codes is None else (codes & c)
        info.append({"face": face, "codes": (codes or set()) - {OUT},
                     "all_codes": (codes or set()), "seeds": face_seeds[fi], "on": on})

    flags = []

    # ---- PASS 1: the code-set intersection ------------------------------------------
    for it in info:
        if len(it["codes"]) == 1:
            it["rid"] = int(next(iter(it["codes"])))
            if it["seeds"] and it["rid"] not in it["seeds"]:
                flags.append(("face_code_conflict", it["rid"], tuple(sorted(it["seeds"])),
                              round(it["face"].area, 2)))

    # ---- PASS 2: the NEIGHBOUR RULE, to a fixed point --------------------------------
    for _ in range(len(info) + 2):
        changed = False
        for fi, it in enumerate(info):
            if "rid" in it or len(it["all_codes"]) < 2:
                continue
            votes = set()
            for i in it["on"]:
                pair = {int(x) for x in arcs[i].code}
                others = [j for j in arc_to_face[i] if j != fi]
                if not others:
                    other_rid = OUT                       # the UNBOUNDED face is background
                else:
                    known = [info[j]["rid"] for j in others if "rid" in info[j]]
                    if not known:
                        continue
                    other_rid = known[0]
                cand = pair - {other_rid}
                if len(cand) == 1:
                    votes.add(int(next(iter(cand))))
            if len(votes) == 1:
                it["rid"] = int(next(iter(votes)))
                changed = True
        if not changed:
            break

    # ---- PASS 3 (seed) and PASS 4 (UNLABELED) ----------------------------------------
    nu = int(params.unlabeled_id_start if next_unlabeled is None else next_unlabeled)
    nn = int(params.new_id_start if next_new is None else next_new)
    for it in info:
        if "rid" in it:
            if len(it["seeds"]) > 1:
                flags.append(("merged_regions", tuple(sorted(it["seeds"])),
                              round(it["face"].area, 2)))
            continue
        if len(it["seeds"]) > 1:
            # TWO OR MORE SEEDS IN ONE FACE = a separating boundary is MISSING and the regions
            # have MERGED. Handing the blob to one of them would hide the failure.
            flags.append(("merged_regions", tuple(sorted(it["seeds"])),
                          round(it["face"].area, 2)))
            rid = None
        elif len(it["seeds"]) == 1 and (not it["codes"] or it["seeds"][0] in it["codes"]):
            rid = int(it["seeds"][0])                     # PASS 3
        else:
            rid = None
        if rid is None:                                   # PASS 4: make the gap VISIBLE
            rid = nu
            nu += 1
            why = ("empty_code_set" if not it["all_codes"]
                   else "multi_seed" if len(it["seeds"]) > 1
                   else "ambiguous_no_seed")
            flags.append(("unlabeled", rid, why, tuple(sorted(it["seeds"])),
                          tuple(sorted(it["all_codes"])), round(it["face"].area, 2)))
        it["rid"] = rid

    regions = defaultdict(list)
    for it in info:
        if it["rid"] != OUT:
            regions[int(it["rid"])].append(it["face"])
    return dict(regions), arc_faces, flags, nu, nn

def rebuild_atlas(arcs, seeds, params: FusionParams, next_unlabeled=None, next_new=None):
    """arcs -> POLYGONS. [5.7]"""
    clean, prep = prepare_arcs(arcs, params)
    faces, lines, prep2 = polygonize_arcs(clean, params)
    prep.update(prep2)
    prep["fallback_chainer"] = False

    if not faces:
        fb = chainer_fallback(clean, params)
        prep["fallback_chainer"] = True
        prep["polygonize_error"] = (prep.get("polygonize_error")
                                    or "polygonize produced 0 faces")
        return {"regions": fb, "faces": [], "arcs": clean, "arc_faces": [0] * len(clean),
                "flags": [("polygonize_failed", prep["polygonize_error"])],
                "report": prep, "next_unlabeled": next_unlabeled, "next_new": next_new}

    regions, arc_faces, flags, nu, nn = assign_faces(faces, clean, seeds, params,
                                                     next_unlabeled, next_new)
    return {"regions": regions, "faces": faces, "arcs": clean, "arc_faces": arc_faces,
            "flags": flags, "report": prep, "next_unlabeled": nu, "next_new": nn}

def poly_union(faces):
    """Union a list of faces, repairing invalid ones with buffer(0). [5.7.5]"""
    return unary_union([f if f.is_valid else f.buffer(0) for f in faces])

def region_anchors(regions, min_area=1.0):
    """One label anchor per FACE, so islands and split regions each get their own label. [3.1.6]"""
    out = []
    for rid, faces in regions.items():
        for f in faces:
            if f.area < min_area:
                continue
            try:
                pt = _polylabel(f, tolerance=0.25) if _polylabel else f.representative_point()
            except Exception:
                pt = f.representative_point()
            out.append({"rid": int(rid), "x": float(pt.x), "y": float(pt.y),
                        "clearance": float(f.boundary.distance(pt)), "area": float(f.area)})
    return out

def anchors_from_regions(regions, min_area=1.0):
    """Label anchors in the SAME tuple shape the notebook's `_place_labels` already eats: (rid, x,
    y, clearance, n_instances). [3.1.6]"""
    a = region_anchors(regions, min_area)
    n = defaultdict(int)
    for r in a:
        n[r["rid"]] += 1
    return [(r["rid"], r["x"], r["y"], r["clearance"], n[r["rid"]]) for r in a]

# =====================================================================================
# 10.  Diagnostics: S1..S6 (pipeline correctness) + S7 (SHAPE QUALITY)   [5.8.1]
# =====================================================================================
# The S1..S6 table is stage-localised, so one failing label localises the break. Its blind spot,
# named by AFAM: it measures PIPELINE CORRECTNESS and almost nothing about whether the fused
# SHAPE is a good shape. The paper's central finding is that JAC alone is misleading -- GEMS
# scored the BEST Jaccard while producing the WORST HD, PERI, TOPO and AVG_GL, because it buys
# overlap by breaking topology. A GEMS-like failure would look HEALTHY to S1..S6.
#
THRESHOLDS = {
    # --- pipeline correctness -------------------------------------------------------
    "S1_label_yield":        (">=", 0.95),
    "S1_selfpair_count":     ("==", 0),
    "S1_code_degree_min":    (">=", 1),
    "S2_dangling_ends":      ("==", 0),
    "S2_intra_over_inter":   ("<",  1.0),
    "S2_short_arcs":         ("==", 0),
    "S3_node_match_rate":    (">=", 0.98),
    "S3_node_rejected_far":  ("==", 0),
    "S4_snap_dist_max_px":   ("<=", 1.0),
    "S4_arcs_dropped":       ("==", 0),
    "S5_orphan_arcs":        ("==", 0),
    "S5_rings_nonsimple":    ("==", 0),
    "S5_merged_regions":     ("==", 0),
    "S5_unlabeled_faces":    ("==", 0),
    "S5_arc_crossings":      ("==", 0),
    "S5_dangling_unrepaired": ("==", 0),
    "S5_fallback_chainer":   ("==", False),
    "S5_region_yield":       (">=", 1.0),
    "G_overlap_area":        ("<=", 1e-6),
    "G_euler_defect":        ("==", 0),
    # --- S7 shape quality (AFAM Part III) --------------------------------------------
    "S7_budget_saturated":   ("==", 0),
    "S7_fit_residual_px":    ("<=", 0.05),
    "S7_peri_dev_max":       ("<=", 0.02),
    "S7_round_dev_max":      ("<=", 0.05),
    "S7_avg_gl_dev_mean":    (">=", 0.0),
    "S7_topo_delta":         ("==", 0),
    "S7_hd_asym":            ("<=", 0.35),
    "S7_curv_ks":            ("<",  0.10),
    # NOTE: NO threshold on any IoU metric. The worst region is always the SMALLEST region,
    # where a 1px nudge is a large fractional change, so an absolute floor is a false-alarm
    # generator. S6 reports the worst region as a LOCATOR and passes no verdict.
}

def _ok(name, val):
    if name not in THRESHOLDS or val is None:
        return None
    op, ref = THRESHOLDS[name]
    try:
        return {">=": val >= ref, "<=": val <= ref, "<": val < ref, "==": val == ref}[op]
    except TypeError:
        return None

def _poly(P):
    g = Polygon(P)
    return g if g.is_valid else g.buffer(0)

# ---- S1 parse -----------------------------------------------------------------------
def diag_parse(n_paths, n_coded, arcs, curve_dev=0.0, params=None):
    """S1 parse stage: path, code and arc counts plus curve deviation. [5.8.1]"""
    params = params or FusionParams()
    deg = defaultdict(int)
    for a in arcs:
        for c in a.code:
            if c != params.outer_code:
                deg[int(c)] += 1
    return {"S1_label_yield": round(n_coded / max(n_paths, 1), 3),
            "S1_curve_dev_px": round(float(curve_dev), 4),
            "S1_selfpair_count": sum(1 for a in arcs if a.code[0] == a.code[1]),
            "S1_code_degree_min": (min(deg.values()) if deg else 0),
            "S1_n_arcs": len(arcs)}

# ---- S2 nodes -----------------------------------------------------------------------
def diag_nodes(arcs, node_xy, node_regions, params: FusionParams):
    """S2 node stage: junction degrees, dangling ends and short arcs. [5.8.1]"""
    degree = defaultdict(int)
    for a in arcs:
        if a.closed:
            continue
        degree[a.n0] += 1
        degree[a.n1] += 1
    hist = defaultdict(int)
    for k in range(len(node_xy)):
        hist[degree[k]] += 1
    dangling = [k for k in range(len(node_xy)) if degree[k] == 1]

    intra = 0.0
    for a in arcs:
        if a.closed:
            continue
        for n, p in ((a.n0, a.pts[0]), (a.n1, a.pts[-1])):
            intra = max(intra, float(np.hypot(*(p - node_xy[n]))))
    inter = float("inf")
    if len(node_xy) > 1:
        d, _ = cKDTree(node_xy).query(node_xy, k=2)
        inter = float(d[:, 1].min())
    short = [a.code for a in arcs if not a.closed and a.length() < params.node_tol]
    return {"S2_n_nodes": len(node_xy),
            "S2_dangling_ends": len(dangling),
            "S2_dangling_coords": [tuple(np.round(node_xy[k], 2)) for k in dangling[:10]],
            "S2_degree_hist": dict(sorted(hist.items())),
            "S2_min_inter_node_gap_px": round(inter, 3) if np.isfinite(inter) else None,
            "S2_intra_over_inter": (round(intra / inter, 3)
                                    if np.isfinite(inter) and inter > 0 else None),
            "S2_short_arcs": len(short), "S2_short_arc_codes": short[:8]}

# ---- S3 node matching ---------------------------------------------------------------
def diag_match(subj_nodes, node_map, fused_xy, nstats, match_rate):
    """S3 match stage: node match rate, displacement and region-set collisions. [5.8.1]"""
    disp = nstats["disp"]
    p95 = round(float(np.percentile(disp, 95)), 3) if disp else 0.0
    return {"S3_fused_nodes": len(fused_xy),
            "S3_node_match_rate": round(match_rate, 3),
            "S3_regionset_collisions": len(nstats["collision_keys"]),
            "S3_collision_sets": nstats["collision_keys"][:6],
            "S3_node_rejected_far": len(nstats["rejected_far"]),
            "S3_rejected_detail": nstats["rejected_far"][:6],
            "S3_node_disp_p95_px": p95,
            "S3_node_disp_max_px": round(float(max(disp)), 3) if disp else 0.0,
            "S3_suggested_node_match_max": round(3.0 * p95, 2) if p95 else None,  # AFAM 4d
            "S3_unmatched": nstats["unmatched"][:6]}

# ---- S4 arc fusion ------------------------------------------------------------------
def diag_arcs(astats, n_subjects, params: FusionParams):
    """S4 arc stage: fused arc count, snap distance, anchors and drops. [5.8.1]"""
    live = [s for s in astats if not s.get("dropped")]
    snaps = [s["snap"] for s in live]
    return {"S4_n_fused_arcs": len(live),
            "S4_snap_dist_max_px": round(max(snaps), 3) if snaps else 0.0,
            "S4_arcs_dropped": sum(1 for s in astats if s.get("dropped")),
            "S4_arcs_partial": sum(1 for s in live if s["support"] < n_subjects),
            "S4_dir_ambiguous": sum(1 for s in live if s["dir_gap"] < 0.05),
            "S4_support_hist": dict(sorted(
                {k: sum(1 for s in live if s["support"] == k)
                 for k in {s["support"] for s in live}}.items())),
            "S4_kp_anchors_total": sum(s["n_anchors"] for s in live),
            "S4_kp_anchors_mean": (round(float(np.mean([s["n_anchors"] for s in live])), 2)
                                   if live else 0.0),
            "S4_kp_nonmonotone": sum(s["kp_dropped"] for s in live),
            "S4_pts_total": sum(s["n_pts"] for s in live),
            "S4_orphan_policy": params.orphan_policy}

# ---- S5 rebuild ---------------------------------------------------------------------
def diag_rebuild(rb, expected_ids, params: FusionParams):
    """S5 rebuild stage: faces, orphan arcs, unlabeled ids and region yield. [5.8.1]"""
    regions, arcs, arc_faces = rb["regions"], rb["arcs"], rb["arc_faces"]
    flags, prep = rb["flags"], rb["report"]
    nonsimple = [rid for rid, fs in regions.items() for f in fs
                 if not LinearRing(f.exterior.coords).is_simple]
    orphan = [i for i, c in enumerate(arc_faces) if c == 0]
    got = set(regions) - {params.outer_code}
    exp = set(int(x) for x in expected_ids) - {params.outer_code}
    unlab = [f for f in flags if f[0] == "unlabeled"]
    return {"S5_n_faces": len(rb["faces"]),
            "S5_region_yield": round(len(got & exp) / max(len(exp), 1), 3),
            "S5_regions_missing": sorted(exp - got)[:12],
            "S5_regions_extra": sorted(got - exp)[:12],
            "S5_unlabeled_faces": len(unlab),
            "S5_unlabeled_detail": unlab[:8],
            "S5_orphan_arcs": len(orphan),
            "S5_orphan_codes": [arcs[i].code for i in orphan[:8]],
            "S5_rings_nonsimple": len(nonsimple),
            "S5_nonsimple_ids": nonsimple[:8],
            "S5_merged_regions": sum(1 for f in flags if f[0] == "merged_regions"),
            "S5_face_code_conflicts": sum(1 for f in flags if f[0] == "face_code_conflict"),
            "S5_arc_crossings": prep.get("arc_crossings", 0),
            "S5_dropped_degenerate": prep.get("dropped_degenerate", 0),
            "S5_dropped_duplicate": prep.get("dropped_duplicate", 0),
            "S5_dangling_repaired": len(prep.get("dangling_repaired", [])),
            "S5_dangling_unrepaired": len(prep.get("dangling_unrepaired", [])),
            "S5_sliver_faces": prep.get("sliver_faces", 0),
            "S5_fallback_chainer": bool(prep.get("fallback_chainer", False)),
            "S5_polygonize_error": prep.get("polygonize_error")}

# ---- S6 overlap quality (NO VERDICT on IoU; worst region is a LOCATOR) ---------------
def diag_quality(regions, subj_regions, weights, params: FusionParams):
    """IoU / Hausdorff / area vs each subject. [5.8.1]"""
    ious, areas, hd = defaultdict(dict), {}, defaultdict(dict)
    acons = []
    for rid, fs in regions.items():
        F = poly_union(fs)
        areas[rid] = F.area
        exp = 0.0
        for sid, sr in subj_regions.items():
            if rid not in sr:
                continue
            S = poly_union(sr[rid])
            u = F.union(S).area
            ious[sid][rid] = (F.intersection(S).area / u) if u else 0.0
            hd[sid][rid] = float(F.boundary.hausdorff_distance(S.boundary))
            exp += weights[sid] * S.area
        if exp > 0:
            acons.append(abs(F.area - exp) / exp)

    polys = [poly_union(fs) for fs in regions.values()]
    overlap = 0.0
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polys[i].intersects(polys[j]):
                overlap += polys[i].intersection(polys[j]).area

    mean_sid, wmean = {}, []
    for sid, per in ious.items():
        if not per:
            continue
        mean_sid[sid] = round(float(np.mean(list(per.values()))), 3)
        wmean.append(float(np.average(list(per.values()),
                                      weights=[areas[r] for r in per])))
    flat = [(r, v) for per in ious.values() for r, v in per.items()]
    worst = min(flat, key=lambda t: t[1]) if flat else (None, None)
    allhd = [(r, v) for per in hd.values() for r, v in per.items()]
    worst_hd = max(allhd, key=lambda t: t[1]) if allhd else (None, None)

    # M5: two-sided per-region Hausdorff asymmetry. A global max hides a weight/alignment bias.
    asym = []
    for rid in regions:
        vals = [hd[s][rid] for s in hd if rid in hd[s]]
        if len(vals) > 1 and max(vals) > 1e-9:
            asym.append((rid, (max(vals) - min(vals)) / max(vals)))
    worst_asym = max(asym, key=lambda t: t[1]) if asym else (None, 0.0)

    return {"S6_iou_area_weighted": round(float(np.mean(wmean)), 3) if wmean else None,
            "S6_iou_mean_per_subject": mean_sid,
            "S6_iou_worst_region": (int(worst[0]), round(float(worst[1]), 3))
                                   if worst[0] is not None else None,
            "S6_hausdorff_worst_region": (int(worst_hd[0]), round(float(worst_hd[1]), 3))
                                         if worst_hd[0] is not None else None,
            "S6_area_cons_dev": round(float(max(acons)), 4) if acons else None,
            "S7_hd_asym": round(float(worst_asym[1]), 3),
            "S7_hd_asym_region": worst_asym[0],
            "G_overlap_area": round(float(overlap), 6)}

# ---- G Euler ------------------------------------------------------------------------
def diag_euler(fused_arcs, fused_xy, faces):
    """Planar-graph identity V - E + F_bounded = C. [5.8.1]"""
    loops = [a for a in fused_arcs if a.closed]
    V = len(fused_xy) + len(loops)
    E = len(fused_arcs)
    F = len(faces)
    parent = list(range(max(V, 1)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in fused_arcs:
        if not a.closed and a.n0 is not None and a.n1 is not None:
            ra, rb = find(a.n0), find(a.n1)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    C = len({find(i) for i in range(V)}) if V else 0
    return {"G_V": V, "G_E": E, "G_F": F, "G_components": C,
            "G_euler_defect": (V - E + F) - C}

# =====================================================================================
# 11.  S7 SHAPE-QUALITY SUITE  (M1 roundness, M2 PERI, M3' AVG_GL, M4 SKELE,   [5.8.2]
#                               M6 TOPO delta, M7 curvature KS)
# =====================================================================================
def _boundary_pts(poly, spacing=0.5):
    out = []
    for ring in [poly.exterior] + list(poly.interiors):
        P = np.asarray(ring.coords, float)
        out.append(densify(P, spacing, closed=False))
    return out

def avg_gl(poly, spacing=0.5):
    """M3' -- protrusion preservation, the paper's dedicated measure (SATM was the ONLY method with
    a POSITIVE score, +0.273; Karcher lost -10.450 of protrusion length). [5.8.2.2]"""
    hull = poly.convex_hull.exterior
    vals = []
    for P in _boundary_pts(poly, spacing):
        if len(P) < 5:
            continue
        d = np.array([hull.distance(Point(p)) for p in P])
        loc = np.arange(1, len(d) - 1)
        pk = loc[(d[1:-1] >= d[:-2]) & (d[1:-1] >= d[2:]) & (d[1:-1] > 1e-6)]
        vals += [float(d[i]) for i in pk]
    return float(np.mean(vals)) if vals else 0.0

def skele_ratio(poly, px=1.0):
    """M4 -- skeleton length / convex-hull area. [5.8.2.4]"""
    try:
        from skimage.morphology import skeletonize
    except Exception:
        return None
    minx, miny, maxx, maxy = poly.bounds
    W = max(2, int((maxx - minx) / px) + 2)
    H = max(2, int((maxy - miny) / px) + 2)
    if W * H > 4_000_000:
        return None
    yy, xx = np.mgrid[0:H, 0:W]
    X = minx + xx * px
    Y = miny + yy * px
    from shapely.vectorized import contains as _c            # shapely>=2 has this
    try:
        M = _c(poly, X, Y)
    except Exception:
        return None
    if M.sum() < 4:
        return None
    sk = skeletonize(M)
    return float(sk.sum() * px) / max(poly.convex_hull.area, 1e-9)

def curv_hist(poly, sigma_px, spacing=0.25):
    """Pooled |curvature| samples along a polygon boundary, for the M7 KS test. [5.8.2.3]"""
    k = []
    for P in _boundary_pts(poly, spacing):
        if len(P) >= 5:
            k.append(np.abs(curvature(P, sigma_px, spacing)))
    return np.concatenate(k) if k else np.zeros(0)

def diag_shape(regions, subj_regions, weights, astats, params: FusionParams):
    """S7. [5.8.2]"""
    live = [s for s in astats if not s.get("dropped")]
    out = {
        "S7_budget_saturated": sum(s["saturated"] for s in live),
        "S7_fit_residual_px": round(max((s["fit_residual"] for s in live), default=0.0), 4),
        "S7_representation": params.representation,
    }

    peri, rnd, gl, sk, topo, ks = [], [], [], [], [], []
    peri_signed = []
    for rid, fs in regions.items():
        F = poly_union(fs)
        if F.is_empty or F.area <= 0:
            continue
        Fp = list(F.geoms) if hasattr(F, "geoms") else [F]

        present = [s for s in subj_regions if rid in subj_regions[s]]
        if not present:
            continue
        wsum = sum(weights[s] for s in present)
        w = {s: weights[s] / wsum for s in present}

        LF = float(F.length)
        AF = float(F.area)
        Lb = sum(w[s] * poly_union(subj_regions[s][rid]).length for s in present)
        Ab = sum(w[s] * poly_union(subj_regions[s][rid]).area for s in present)
        if Lb > 0:                                                   # M2 / S7_peri_dev
            d = (LF - Lb) / Lb
            peri.append(abs(d)); peri_signed.append(d)
        if Lb > 0 and Ab > 0 and AF > 0:                             # M1 / S7_round_dev
            RF = LF ** 2 / (4 * math.pi * AF)
            Rb = sum(w[s] * (poly_union(subj_regions[s][rid]).length ** 2) /
                     (4 * math.pi * max(poly_union(subj_regions[s][rid]).area, 1e-9))
                     for s in present)
            if Rb > 0:
                rnd.append((rid, (RF - Rb) / Rb))

        if params.metrics_avg_gl:                                    # M3' (opt-in; ~1s/120 regions)
            gF = sum(avg_gl(p) * p.area for p in Fp) / max(AF, 1e-9)
            gb = 0.0
            for s in present:
                S = poly_union(subj_regions[s][rid])
                Sp = list(S.geoms) if hasattr(S, "geoms") else [S]
                gb += w[s] * (sum(avg_gl(p) * p.area for p in Sp) / max(S.area, 1e-9))
            if gb > 1e-9:
                gl.append((rid, (gF - gb) / gb))

        nF = len(Fp)                                                 # M6 / TOPO delta
        nb = sum(w[s] * (len(poly_union(subj_regions[s][rid]).geoms)
                         if hasattr(poly_union(subj_regions[s][rid]), "geoms") else 1)
                 for s in present)
        topo.append((rid, nF - nb))

        if params.metrics_curv_ks:                                   # M7
            try:
                from scipy.stats import ks_2samp
                kF = curv_hist(F if not hasattr(F, "geoms") else max(Fp, key=lambda p: p.area),
                               params.curv_sigma_px)
                kb = np.concatenate([curv_hist(poly_union(subj_regions[s][rid])
                                               if not hasattr(poly_union(subj_regions[s][rid]),
                                                              "geoms")
                                               else max(list(poly_union(subj_regions[s][rid]).geoms),
                                                        key=lambda p: p.area),
                                               params.curv_sigma_px) for s in present])
                if len(kF) > 8 and len(kb) > 8:
                    ks.append((rid, float(ks_2samp(kF, kb).statistic)))
            except Exception:
                pass

        if params.metrics_skele:                                     # M4
            a = skele_ratio(max(Fp, key=lambda p: p.area))
            b = 0.0
            for s in present:
                S = poly_union(subj_regions[s][rid])
                Sp = list(S.geoms) if hasattr(S, "geoms") else [S]
                v = skele_ratio(max(Sp, key=lambda p: p.area))
                if v is None:
                    b = None
                    break
                b += w[s] * v
            if a is not None and b:
                sk.append((rid, (a - b) / b))

    def _worst(lst):
        return (int(max(lst, key=lambda t: abs(t[1]))[0]),
                round(float(max(lst, key=lambda t: abs(t[1]))[1]), 4)) if lst else None

    out.update({
        "S7_peri_dev_max": round(float(max(peri)), 4) if peri else 0.0,
        "S7_peri_dev_mean_signed": round(float(np.mean(peri_signed)), 4) if peri_signed else 0.0,
        "S7_round_dev_max": round(float(max(abs(v) for _, v in rnd)), 4) if rnd else 0.0,
        "S7_round_dev_worst_region": _worst(rnd),
        "S7_avg_gl_dev_mean": round(float(np.mean([v for _, v in gl])), 4) if gl else 0.0,
        "S7_avg_gl_dev_worst_region": _worst(gl),
        "S7_topo_delta": int(round(sum(v for _, v in topo))) if topo else 0,
        "S7_topo_delta_regions": [(int(r), round(float(v), 2)) for r, v in topo
                                  if abs(v) > 1e-6][:8],
        "S7_curv_ks": round(float(np.mean([v for _, v in ks])), 4) if ks else None,
        "S7_curv_ks_worst_region": _worst(ks),
        "S7_skele_dev": round(float(np.mean([v for _, v in sk])), 4) if sk else None,
    })
    return out

# =====================================================================================
# 12.  Reporting   [5.8]
# =====================================================================================
def print_report(d, title="fusion diagnostics", verbose=False):
    """Print the diagnostics dict grouped by stage. [5.8]"""
    HIDE = {"S2_dangling_coords", "S2_short_arc_codes", "S3_collision_sets", "S3_unmatched",
            "S3_rejected_detail", "S5_orphan_codes", "S5_nonsimple_ids", "S5_regions_missing",
            "S5_regions_extra", "S5_unlabeled_detail", "S5_topo_delta_regions"}
    print(f"\n=== {title} ===")
    stage = None
    for k in sorted(d, key=lambda k: (k.split("_")[0], k)):
        if not verbose and k in HIDE:
            continue
        s = k.split("_")[0]
        if s != stage:
            stage = s
            print(f"  --- {stage} ---")
        ok = _ok(k, d[k])
        tag = "     " if ok is None else ("PASS " if ok else "FAIL ")
        print(f"  {tag}{k:28s} = {d[k]}")
    bad = [k for k in d if _ok(k, d[k]) is False]
    print(f"  => {'ALL CHECKS PASS' if not bad else 'FAILING: ' + ', '.join(sorted(bad))}")
    if not verbose and bad:
        for k in sorted(bad):
            for det in (k.replace("_faces", "_detail"), k + "_detail",
                        k.replace("S5_orphan_arcs", "S5_orphan_codes"),
                        k.replace("S3_node_rejected_far", "S3_rejected_detail")):
                if det in d and d[det]:
                    print(f"     {det}: {d[det]}")
                    break
    return bad

# =====================================================================================
# 13.  Region registry  (persistent, PER SLICE)   [5.7.4]
# =====================================================================================
# The report's registry assumes one image pair = one map. Here a region legitimately does not
# appear on most slices, so a global registry would mark ~99% of DB09 inactive every run.
# Keyed by slice: "absent" is measured against what the TEMPLATE says should be on THIS slice.
def load_registry(path):
    """Read the per-slice region registry, returning a fresh one if absent. [5.7.4]"""
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            warnings.warn(f"registry unreadable ({e}); starting a fresh one")
    return {"next_new_id": None, "next_unlabeled_id": None, "slices": {}}

def save_registry(reg, path):
    """Write the per-slice region registry as JSON. [5.7.4]"""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(reg, f, indent=2, sort_keys=True)
    return path

def registry_next_new(reg, params):
    """Next expert-drawn region id to allocate. [5.7.4]"""
    return int(reg.get("next_new_id") or params.new_id_start)

def registry_next_unlabeled(reg, params):
    """Next UNLABELED region id to allocate. [5.7.4]"""
    return int(reg.get("next_unlabeled_id") or params.unlabeled_id_start)

def registry_update(reg, slice_index, present, expected, run_id,
                    next_new=None, next_unlabeled=None, params=None):
    """present = region ids in the FUSED map for this slice expected = region ids the template says
    should be on this slice Expected-but-absent is FLAGGED inactive, never deleted. [5.7.4]"""
    params = params or FusionParams()
    key = str(int(slice_index))
    rec = reg["slices"].setdefault(key, {"regions": {}})
    known = set(int(k) for k in rec["regions"]) | set(int(x) for x in expected)
    for rid in sorted(set(int(x) for x in present)):
        r = rec["regions"].setdefault(str(rid), {"first_seen_run": run_id})
        r["status"] = "active"
        r["last_seen_run"] = run_id
        r["source"] = ("unlabeled" if rid >= params.unlabeled_id_start and rid < params.new_id_start
                       else "expert" if rid >= params.new_id_start else "template")
        r.pop("deleted_run", None)
    for rid in sorted(known - set(int(x) for x in present)):
        r = rec["regions"].setdefault(str(rid), {"first_seen_run": run_id})
        if r.get("status") != "inactive":
            r["status"] = "inactive"
            r["deleted_run"] = run_id
    if next_new is not None:
        reg["next_new_id"] = int(next_new)
    if next_unlabeled is not None:
        reg["next_unlabeled_id"] = int(next_unlabeled)
    return reg

def registry_purge(reg, slice_index=None):
    """Delete inactive region records, for one slice or all. [5.7.4]"""
    keys = [str(int(slice_index))] if slice_index is not None else list(reg["slices"])
    n = 0
    for k in keys:
        regs = reg["slices"].get(k, {}).get("regions", {})
        for r in [r for r, v in regs.items() if v.get("status") == "inactive"]:
            regs.pop(r)
            n += 1
    return reg, n

def registry_inactive(reg, slice_index):
    """Region ids marked inactive on one slice. [5.7.4]"""
    regs = reg["slices"].get(str(int(slice_index)), {}).get("regions", {})
    return sorted(int(r) for r, v in regs.items() if v.get("status") == "inactive")

# =====================================================================================
# 14.  One-call driver   [5]
# =====================================================================================
def _subject_regions(res, seeds, params):
    """Each subject's OWN polygons -- the reference the S6/S7 metrics are measured against."""
    out = {}
    for sid, arcs in res.subj_arcs.items():
        # exclude phantom arcs: a subject's own reference atlas is its ORIGINAL geometry
        rb = rebuild_atlas([a for a in arcs if not getattr(a, 'phantom', False)], seeds, params)
        out[sid] = rb["regions"]
    return out

def run_fusion(graphs, seeds, expected_ids, weights=None, params: FusionParams = None,
               registry=None, run_id=1, slice_index=0, verbose=True, verbose_diag=False,
               timings=None, report_ctx=None):
    """Fuse -> rebuild (polygons = the atlas) -> label -> registry -> diagnose, in one call. [5]
    S6/S7 are gated by params.run_comparative_diag, which costs one atlas rebuild per subject."""
    params = params or FusionParams()
    _t = _StageTimer(timings)

    with _t("fuse"):
        # report_ctx defaults to slice_index alone so hard-stops/notes still name the slice
        # even when the caller passes no run label / acronym LUT.
        res = fuse_graphs(graphs, weights=weights, params=params, seeds=seeds,
                          report_ctx=report_ctx or {"slice_index": slice_index})

    # ---- STAGES 3-5 GATE: build_polygons=False stops at the fused LINES ----------------
    # graph, node_graph) is still produced, so the renderer can draw the averaged contours.
    if params.build_polygons:
        nn = registry_next_new(registry, params) if registry is not None else None
        nu = registry_next_unlabeled(registry, params) if registry is not None else None
        with _t("rebuild"):
            rb = rebuild_atlas(res.arcs, seeds, params, next_unlabeled=nu, next_new=nn)
        regions = rb["regions"]
        anchors = anchors_from_regions(regions)

        if registry is not None:
            registry = registry_update(registry, slice_index, present=list(regions),
                                       expected=expected_ids, run_id=run_id,
                                       next_new=rb["next_new"], next_unlabeled=rb["next_unlabeled"],
                                       params=params)
    else:
        rb = {"regions": {}, "faces": [], "arcs": res.arcs, "flags": [],
              "next_new": None, "next_unlabeled": None}
        regions, anchors = {}, []
        print("  [build_polygons=False] STOPPED AT LINES: no polygonize, no labels, "  # TROUBLESHOOTING PRINT
              "no S5/S6/S7, registry untouched.")

    ref = res.stats["ref_sid"]
    xy, rg = res.subj_nodes[ref]

    # ---- S1..S5: cheap, use only the fused result. ALWAYS run. ----------------------
    d = {}
    with _t("diag_cheap"):
        d.update(diag_parse(len(res.arcs), len(res.arcs), res.arcs, params=params))
        d.update(diag_nodes(res.subj_arcs[ref], xy, rg, params))
        d.update(diag_match(res.subj_nodes, res.node_map, res.fused_xy,
                            res.stats["nodes"], res.stats["node_match_rate"]))
        d.update(diag_arcs(res.stats["arcs"], res.stats["n_subjects"], params))
        if params.build_polygons:                          # S5 + Euler need faces
            d.update(diag_rebuild(rb, expected_ids, params))
            d.update(diag_euler(res.arcs, res.fused_xy, rb["faces"]))

    # ---- S6 + S7: comparative (subject vs fused). GATED. ----------------------------
    if params.build_polygons and params.run_comparative_diag:
        with _t("subject_rebuild"):
            subj_regions = _subject_regions(res, seeds, params)
        # diag_ignore_unlabeled: hide QA-gap ids (8001+) and expert ids (9001+) from the
        if params.diag_ignore_unlabeled:
            diag_regions = {rid: fs for rid, fs in regions.items()
                            if rid < params.unlabeled_id_start}
            print(f"  [diag_ignore_unlabeled] S6/S7 computed over {len(diag_regions)} labelled "  # TROUBLESHOOTING PRINT
                  f"region(s); {len(regions) - len(diag_regions)} unlabeled/expert hidden from metrics.")
        else:
            diag_regions = regions
        with _t("diag_comparative"):
            d.update(diag_quality(diag_regions, subj_regions, res.stats["weights"], params))
            d.update(diag_shape(diag_regions, subj_regions, res.stats["weights"],
                                res.stats["arcs"], params))
    else:
        subj_regions = {}      # not computed; S6/S7 keys are simply absent from `d`

    if verbose:
        title = (f"slice {slice_index}: edge/node + SATM fusion "
                 f"({len(graphs)} subjects, {len(res.arcs)} arcs, {len(regions)} regions, "
                 f"rep={params.representation}, orphan={params.orphan_policy}"
                 f"{'' if params.run_comparative_diag else ', S6/S7 OFF'})")
        print_report(d, title, verbose=verbose_diag)
    node_graph = arcs_to_node_graph(res.arcs, verify=True)      # every vertex is a node now
    return {"result": res, "graph": res.graph, "node_graph": node_graph,
            "arcs": res.arcs, "rebuild": rb,
            "faces": rb["faces"], "regions": regions, "anchors": anchors,
            "flags": rb["flags"], "subj_regions": subj_regions,
            "registry": registry, "diag": d}

# =====================================================================================
# 15.  S7_sweep_identity + the weight-sweep stability curve            [M8, AFAM prio 2]
#    [5.8.3, 5.8.4]
# =====================================================================================
def _sample_arcs(arcs, step=0.25):
    P = []
    for a in arcs:
        Q = np.vstack([a.pts, a.pts[0]]) if a.closed else a.pts
        P.append(densify(Q, step))
    return np.vstack(P) if P else np.zeros((0, 2))

def _curve_dist(arcsA, arcsB, step=0.25):
    """Symmetric max POINT-TO-POLYLINE distance (not point-cloud, which has a step/2 floor)."""
    PA, PB = _sample_arcs(arcsA, step), _sample_arcs(arcsB, step)
    if not len(PA) or not len(PB):
        return float("nan")
    LA = [np.vstack([a.pts, a.pts[0]]) if a.closed else a.pts for a in arcsA]
    LB = [np.vstack([a.pts, a.pts[0]]) if a.closed else a.pts for a in arcsB]
    dA = np.min([[max_deviation(p[None], R) for R in LB] for p in PA], axis=1).max()
    dB = np.min([[max_deviation(p[None], R) for R in LA] for p in PB], axis=1).max()
    return float(max(dA, dB))

def sweep_identity(graphs, params: FusionParams = None, step=0.5):
    """S7_sweep_identity: all weight on subject i must return subject i's own map. [5.8.3]
    Bound is fit_tol*3 + node_tol/2; needs no reference data."""
    params = params or FusionParams()
    sids = list(graphs)
    bound = params.fit_tol_px * 3 + params.node_tol / 2
    out = {}
    for i in sids:
        w = {s: (1.0 if s == i else 0.0) for s in sids}
        w[i] = 1.0
        p2 = FusionParams(**{**params.__dict__})
        p2.strict_topo = False                      # a one-hot run must not abort on topology
        p2.phantom_regions = False                  # nor inject phantoms (would perturb identity)
        res = fuse_graphs(graphs, weights={s: (1.0 if s == i else 1e-12) for s in sids},
                          params=p2)
        out[i] = _curve_dist(res.arcs, graph_to_arcs(graphs[i]), step)
    worst = max(out.values()) if out else 0.0
    return {"S7_sweep_identity_px": round(worst, 4),
            "S7_sweep_identity_bound_px": round(bound, 4),
            "S7_sweep_identity_per_subject": {k: round(v, 4) for k, v in out.items()},
            "S7_sweep_identity_ok": bool(worst <= bound)}

THRESHOLDS["S7_sweep_identity_ok"] = ("==", True)

def sweep_weights(graphs, seeds, expected_ids, pair=None, steps=11,
                  params: FusionParams = None, metrics=("S7_peri_dev_mean_signed",
                                                        "S7_round_dev_max",
                                                        "S7_avg_gl_dev_mean",
                                                        "S6_iou_area_weighted",
                                                        "S7_topo_delta")):
    """M8 weight-sweep stability curve: sweep w from 0 to 1 between two subjects. [5.8.4]
    Returns one dict per w; non-monotone metrics in between indicate instability."""
    params = params or FusionParams()
    sids = list(graphs)
    a, b = pair or (sids[0], sids[-1])
    rows = []
    for w in np.linspace(0.0, 1.0, steps):
        wts = {s: 1e-12 for s in sids}
        wts[a] = max(1.0 - w, 1e-12)
        wts[b] = max(w, 1e-12)
        p2 = FusionParams(**{**params.__dict__})
        p2.strict_topo = False
        p2.phantom_regions = False                  # weight sweep must recover inputs at w=0/1
        try:
            out = run_fusion({a: graphs[a], b: graphs[b]}, seeds, expected_ids, weights=wts,
                             params=p2, slice_index=0, verbose=False)
            row = {"w": round(float(w), 3)}
            row.update({m: out["diag"].get(m) for m in metrics})
            row["n_regions"] = len(out["regions"])
        except Exception as e:
            row = {"w": round(float(w), 3), "error": f"{type(e).__name__}: {e}"}
        rows.append(row)
    return rows
