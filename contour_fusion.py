"""
contour_fusion.py  --  LEGACY per-region contour averaging.   *** SUPERSEDED ***

Fusion now happens in edge_fusion.py (arc/node method). This module is kept ONLY as
(a) the A/B baseline and (b) the synthetic circle sanity check in the verification cell.
Do not use fuse_slice_contours() on real slices. Three reasons, all verified:

  1. DOUBLE-AVERAGED SHARED BORDERS. The A|B border lives inside A's loop AND inside B's
     loop, so it is resampled, start-aligned and averaged TWICE, independently. The two
     copies disagree -> gaps, overlaps and doubled lines in the fused map.
  2. ISLANDS ARE DROPPED. fused_contours_to_graph / the caller's _order_edge_loop keep only
     the LARGEST connected component of a region's boundary. A region with an island in it
     has two boundary loops; the second one is discarded. (Measured on a synthetic slice:
     region with an island kept 57/77 boundary nodes; the donut kept 30/42.)
  3. NEIGHBOUR CODE LOST. The fused graph is written with elements (rid, 0), so every region
     claims to border background. The single-line invariant and the double-line renderer
     both break.

edge_fusion.py fixes all three by construction: each shared arc is stored and averaged
exactly ONCE, islands are first-class closed arcs, and both codes ride on every element.
"""

import numpy as np


# 5.6.6.2.1  resample a closed contour to n points, evenly spaced by arc length
def resample_closed_contour(points, n=200):
    """Resample a closed contour to n points, evenly spaced by arc length."""
    pts = np.asarray(points, float)
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])              # close it
    seg = np.diff(pts, axis=0)
    dist = np.r_[0, np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))]
    total = dist[-1]
    if total == 0:
        return np.repeat(pts[:1], n, axis=0)
    targets = np.linspace(0, total, n, endpoint=False)
    out = np.empty((n, 2))
    for k, t in enumerate(targets):
        j = np.searchsorted(dist, t) - 1
        j = np.clip(j, 0, len(seg)-1)
        local = (t - dist[j]) / (dist[j+1] - dist[j] + 1e-12)
        out[k] = pts[j] + local * (pts[j+1] - pts[j])
    return out


# 5.6.2.1  force consistent winding, then roll the start point onto the reference
def _align_direction_and_start(ref, cand):
    """Reverse cand if it runs opposite to ref, then roll it to the best start offset."""
    # direction: compare signed area (orientation) of the two contours
    def signed_area(p):
        x, y = p[:, 0], p[:, 1]
        return 0.5*np.sum(x*np.roll(y, -1) - np.roll(x, -1)*y)
    if np.sign(signed_area(cand)) != np.sign(signed_area(ref)):
        cand = cand[::-1]
    # start offset: roll cand to minimise sum of squared distances to ref
    n = len(ref)
    best, best_k = np.inf, 0
    # coarse search every few points for speed, then refine
    for k in range(0, n, max(1, n//50)):
        d = np.sum((np.roll(cand, -k, axis=0) - ref)**2)
        if d < best:
            best, best_k = d, k
    return np.roll(cand, -best_k, axis=0)


# 5.6.6.3  pointwise mean of one region's contour across subjects
def average_region_contours(subject_contours, n=200):
    """Average a list of closed contours (one per subject) for ONE region.
    Returns the mean contour (n,2)."""
    res = [resample_closed_contour(c, n) for c in subject_contours if len(c) >= 3]
    if not res:
        return None
    ref = res[0]
    aligned = [ref] + [_align_direction_and_start(ref, c) for c in res[1:]]
    return np.mean(np.stack(aligned, axis=0), axis=0)


# 5.6  average every region on one slice (whole-contour path; edge_fusion is the per-arc path)
def fuse_slice_contours(subject_region_contours, n=200):
    """Average all regions on one slice across subjects.
    `subject_region_contours` = {region_id: [contour_subj1, contour_subj2, ...]}.
    Returns {region_id: mean_contour}."""
    fused = {}
    for region_id, contours in subject_region_contours.items():
        mc = average_region_contours(contours, n)
        if mc is not None:
            fused[region_id] = mc
    return fused


# 5.7 / 3.1.6  mean contours into a BoundaryGraph plus one centroid seed per region
def fused_contours_to_graph(fused_contours, merge_tol=None):
    """Turn {region_id: mean_contour} into a BoundaryGraph + seed list for region ID.
    Each contour becomes a closed polyline; the region centroid becomes its seed."""
    import boundary_graph as bg
    polylines, seeds = [], []
    for region_id, contour in fused_contours.items():
        closed = np.vstack([contour, contour[0]])
        polylines.append(closed)
        seeds.append((tuple(contour.mean(axis=0)), int(region_id)))
    g = bg.polyline_to_graph(polylines, merge_tol=merge_tol)   # finalizes .nodes itself
    return g, seeds
