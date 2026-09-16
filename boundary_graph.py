"""
boundary_graph.py  --  Node/Element boundary-graph engine
Implements the data model and cleanup operations from Kulkarni (2006), section 3.2:
  - Node  : an (x, y) point.
  - Element: a line segment between two nodes, carrying TWO material codes
             (the regions on either side of the line).
  - Operations:  spline->polyline, Clear (dedupe), Extend (open nodes),
                 Intersect (crossings), Merge (redundant elements),
                 Loop (counter-clockwise marching-line region ID + parity island fix).

This is a 2D, per-slice engine. A "slice atlas" is a graph; a full atlas is a
dict of {slice_index: BoundaryGraph}. Everything here is pure-Python/numpy/scipy
so it runs in Colab with no extra machinery.
"""
import numpy as np
from dataclasses import dataclass, field
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
# 3.1.3 / 3.1.3.3  data model: nodes, plus elements that each record the two regions they separate
class BoundaryGraph:
    """Planar straight-line graph of one slice."""
    nodes: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))   # (Nx2) xy
    # elements: list of [node_i, node_j, matA, matB]; mat codes -1 == unknown
    elements: list = field(default_factory=list)

    _grid: dict = field(default_factory=dict, repr=False, compare=False)
    _pts:  list = field(default_factory=list, repr=False, compare=False)

    def _open_buffer(self, tol):
        """Seed the insert buffer and its grid hash from whatever .nodes currently holds."""
        self._pts = [np.asarray(p, float) for p in self.nodes]
        self._grid = {}
        for j, p in enumerate(self._pts):
            self._grid.setdefault((round(p[0] / tol), round(p[1] / tol)), []).append(j)

    def add_node(self, xy, tol=1e-6):
        """Add a node, reusing an existing one within `tol`. Grid-hashed for O(1) average
        insert. .nodes is NOT rebuilt per insert (that was O(n) per call, O(n^2) per graph):
        call finalize_nodes() once after the insert loop. Replacing .nodes by hand after an
        insert run invalidates the buffer; build a new BoundaryGraph instead."""
        xy = np.asarray(xy, float)
        if not self._pts:
            self._open_buffer(tol)
        cell = (round(float(xy[0]) / tol), round(float(xy[1]) / tol))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in self._grid.get((cell[0] + dx, cell[1] + dy), ()):
                    if np.hypot(*(self._pts[j] - xy)) <= tol:
                        return j
        j = len(self._pts)
        self._pts.append(xy)
        self._grid.setdefault(cell, []).append(j)
        return j

    def finalize_nodes(self):
        """Materialise .nodes from the deferred insert buffer. Call once after a run of
        add_node calls. Idempotent; a no-op when nothing was deferred."""
        if self._pts:
            self.nodes = np.asarray(self._pts, float).reshape(-1, 2)
        elif self.nodes is None or not len(self.nodes):
            self.nodes = np.zeros((0, 2))
        return self.nodes

    def add_element(self, i, j, matA=-1, matB=-1):
        if i == j:
            return
        self.elements.append([i, j, matA, matB])

    def node_degree(self):
        deg = np.zeros(len(self.nodes), int)
        for e in self.elements:
            deg[e[0]] += 1; deg[e[1]] += 1
        return deg


# ---------------------------------------------------------------------------
# Kulkarni 3.2.1  Spline -> polyline approximation
# ---------------------------------------------------------------------------
# 4.4.4  flatten a cubic segment into a polyline
def cubic_spline_to_polyline(coeffs, t_values=(0.0, 0.25, 0.75, 1.0)):
    """Approximate one cubic spline segment by straight lines at the given t.
    `coeffs` = (ax,bx,cx,dx, ay,by,cy,dy) per eq. 3-1. Returns ordered points."""
    ax, bx, cx, dx, ay, by, cy, dy = coeffs
    pts = []
    for t in t_values:
        x = ax*t**3 + bx*t**2 + cx*t + dx
        y = ay*t**3 + by*t**2 + cy*t + dy
        pts.append((x, y))
    return np.array(pts)


# 3.1.3.4  polylines into a graph, merging coincident endpoints
def polyline_to_graph(polylines, merge_tol=None, box_frac=3e-8):
    """Build a BoundaryGraph from a list of polylines (each an (M,2) array).
    Coincident nodes within `merge_tol` are merged (the paper used ~3e-8 * box width)."""
    g = BoundaryGraph()
    all_pts = np.vstack([p for p in polylines]) if polylines else np.zeros((0, 2))
    if merge_tol is None and len(all_pts):
        box_w = (all_pts.max(0) - all_pts.min(0)).max()
        merge_tol = box_frac * box_w if box_w > 0 else 1e-9
    for pl in polylines:
        prev = None
        for xy in pl:
            ni = g.add_node(xy, tol=merge_tol)
            if prev is not None:
                g.add_element(prev, ni)
            prev = ni
    g.finalize_nodes()
    return g


# ---------------------------------------------------------------------------
# Kulkarni 3.2.2  Extend / Intersect / Merge
# ---------------------------------------------------------------------------
def _point_seg_distance(p, a, b):
    """Distance from point p to segment ab, plus the projection parameter s in [0,1]."""
    ab = b - a
    L2 = ab.dot(ab)
    if L2 == 0:
        return np.hypot(*(p - a)), 0.0
    s = np.clip((p - a).dot(ab) / L2, 0.0, 1.0)
    proj = a + s * ab
    return np.hypot(*(p - proj)), s


# 3.1.5  close open arcs: a degree-1 node is snapped onto the nearest element
def extend_open_nodes(g):
    """Kulkarni 3.2.2 Extend: open nodes (degree 1) are snapped to the nearest element if they
    fall inside that element's bounding box, else connected to its nearest endpoint."""
    deg = g.node_degree()
    open_ids = np.where(deg == 1)[0]
    for ni in open_ids:
        p = g.nodes[ni]
        best = (np.inf, None, None)   # (dist, elem_idx, s)
        for ei, e in enumerate(g.elements):
            if ni in (e[0], e[1]):
                continue
            a, b = g.nodes[e[0]], g.nodes[e[1]]
            d, s = _point_seg_distance(p, a, b)
            if d < best[0]:
                best = (d, ei, s)
        if best[1] is None:
            continue
        ei = best[1]; e = g.elements[ei]; a, b = g.nodes[e[0]], g.nodes[e[1]]
        # element bounding box test
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        inside_box = np.all(p >= lo - 1e-9) and np.all(p <= hi + 1e-9)
        if inside_box:
            # snap onto element: split element at the projection point
            s = best[2]
            newxy = a + s * (b - a)
            nj = g.add_node(newxy)
            matA, matB = e[2], e[3]
            g.elements[ei] = [e[0], nj, matA, matB]
            g.add_element(nj, e[1], matA, matB)
            g.add_element(ni, nj)
        else:
            # connect to the closest endpoint of the element
            endpoint = e[0] if np.hypot(*(p - a)) <= np.hypot(*(p - b)) else e[1]
            g.add_element(ni, endpoint)
    return g


def _seg_intersection(p1, p2, p3, p4):
    """Return intersection point of segments p1p2 and p3p4 if they cross (else None)."""
    r = p2 - p1; s = p4 - p3
    denom = r[0]*s[1] - r[1]*s[0]
    if abs(denom) < 1e-12:
        return None
    qp = p3 - p1
    t = (qp[0]*s[1] - qp[1]*s[0]) / denom
    u = (qp[0]*r[1] - qp[1]*r[0]) / denom
    if 1e-9 < t < 1-1e-9 and 1e-9 < u < 1-1e-9:
        return p1 + t * r
    return None


# 5.7.2.1 / 5.7.2.2  insert a shared node where two elements cross and split both
def intersect_elements(g):
    """Kulkarni 3.2.2 Intersect: where two elements cross, insert a shared node and split both
    into 4 elements (so every element separates exactly two materials)."""
    changed = True
    while changed:
        changed = False
        for i in range(len(g.elements)):
            for j in range(i+1, len(g.elements)):
                ei, ej = g.elements[i], g.elements[j]
                if set((ei[0], ei[1])) & set((ej[0], ej[1])):
                    continue  # share a node already
                P = _seg_intersection(g.nodes[ei[0]], g.nodes[ei[1]],
                                      g.nodes[ej[0]], g.nodes[ej[1]])
                if P is not None:
                    nk = g.add_node(P)
                    a, b = ei[0], ei[1]; c, d = ej[0], ej[1]
                    mA = (ei[2], ei[3]); mB = (ej[2], ej[3])
                    g.elements[i] = [a, nk, *mA]
                    g.elements[j] = [c, nk, *mB]
                    g.add_element(nk, b, *mA)
                    g.add_element(nk, d, *mB)
                    changed = True
                    break
            if changed:
                break
    return g


# 5.7.1.2 / 6.1.2.2  collapse duplicate elements between the same node pair
def merge_redundant_elements(g):
    """Kulkarni 3.2.2 Merge: collapse duplicate elements connecting the same node pair."""
    seen = {}
    out = []
    for e in g.elements:
        key = tuple(sorted((e[0], e[1])))
        if key in seen:
            continue
        seen[key] = True
        out.append(e)
    g.elements = out
    return g


# 5.7.1  prepare arcs: extend, intersect, merge
def clear_clean(g, node_tol=1e-6):
    """3 'Clear': remove zero-length elements and exact-duplicate nodes/elements."""
    g.elements = [e for e in g.elements if e[0] != e[1]]
    return merge_redundant_elements(g)


# ---------------------------------------------------------------------------
# Kulkarni 3.2.3  Loop: counter-clockwise marching-line region identification
#
# NOTE ON NESTED REGIONS: the marching loop + parity-island rule (resolve_islands_
# parity) covers the common cases (adjacent regions, a region against background, and
# single isolated islands like 'PS' in Kulkarni Fig 3.12). Deeply nested topologies (region
# inside region inside region) may leave some outer-ring elements with one code
# unresolved (-1); these are reported by audit_unresolved() so the expert can fix the
# handful of affected elements during QA. This matches the paper's own reliance on
# manual seed placement for ambiguous nests.
# ---------------------------------------------------------------------------
def _build_adjacency(g):
    adj = {i: [] for i in range(len(g.nodes))}
    for ei, e in enumerate(g.elements):
        adj[e[0]].append(ei); adj[e[1]].append(ei)
    return adj


def _ray_intersections(g, seed, direction=None):
    """Return [(distance, element_idx)] for elements hit by a ray from seed.
    Uses a parametric ray (origin + t*dir, t>0) rather than a finite far endpoint,
    which avoids numerical underflow for elements close to the seed. The direction is
    slightly irrational to avoid degenerate hits through vertices."""
    hits = []
    o = np.asarray(seed, float)
    d = np.asarray(direction if direction is not None else [1.0, 0.0001732051], float)
    for ei, e in enumerate(g.elements):
        a = g.nodes[e[0]]; b = g.nodes[e[1]]
        s = b - a
        denom = d[0]*(-s[1]) - d[1]*(-s[0])      # cross(d, s)
        if abs(denom) < 1e-12:
            continue
        ao = a - o
        t = (ao[0]*(-s[1]) - ao[1]*(-s[0])) / denom   # ray param (>0 ahead of seed)
        u = (d[0]*ao[1] - d[1]*ao[0]) / denom         # segment param in [0,1]
        if t > 1e-9 and 1e-9 < u < 1 - 1e-9:
            hits.append((t * np.hypot(*d), ei))
    hits.sort()
    return hits


# 3.1.6.2 / 5.7.3  seeds decide which region each element bounds
def identify_regions(g, seeds):
    """Kulkarni 3.2.3: for each seed=(xy, material_code), shoot a ray, take the nearest hit as
    the start element, then march counter-clockwise (steer relative-left at each shared
    node) until the loop closes; assign the seed's material to those elements."""
    adj = _build_adjacency(g)
    for xy, mat in seeds:
        hits = _ray_intersections(g, xy)
        if not hits:
            continue
        start = hits[0][1]
        loop = _march_loop(g, adj, start, xy)
        for ei in loop:
            _assign_material(g.elements[ei], mat)
    return g


def _assign_material(elem, mat):
    """Fill the first free material slot (-1) of an element with `mat`."""
    if elem[2] == -1:
        elem[2] = mat
    elif elem[3] == -1 and elem[2] != mat:
        elem[3] = mat


def _march_loop(g, adj, start_ei, seed_xy, max_steps=100000):
    """Walk the boundary loop using the tail->head, steer-left rule.
    The start element is oriented so the seed point lies to its LEFT, so the
    counter-clockwise march encloses the seed's region."""
    e = g.elements[start_ei]
    a, b = e[0], e[1]
    # choose orientation (a->b vs b->a) so seed is on the left of the directed edge
    v = g.nodes[b] - g.nodes[a]
    to_seed = np.asarray(seed_xy, float) - g.nodes[a]
    cross = v[0]*to_seed[1] - v[1]*to_seed[0]   # >0 => seed left of a->b
    if cross < 0:
        a, b = b, a
    tail, head = a, b
    loop = [start_ei]
    cur_ei = start_ei
    cur_head = head; cur_tail = tail
    for _ in range(max_steps):
        cands = [k for k in adj[cur_head] if k != cur_ei]
        if not cands:
            break
        if len(cands) == 1:
            nxt = cands[0]
        else:
            nxt = _steer_left(g, cur_tail, cur_head, cands)
        e2 = g.elements[nxt]
        nxt_tail = cur_head
        nxt_head = e2[1] if e2[0] == cur_head else e2[0]
        cur_ei, cur_tail, cur_head = nxt, nxt_tail, nxt_head
        if cur_ei == start_ei:
            break
        loop.append(cur_ei)
    return loop


def _steer_left(g, tail, head, cand_elem_ids):
    """Pick the candidate element making the most counter-clockwise (leftmost) turn."""
    v_in = g.nodes[head] - g.nodes[tail]
    a_in = np.arctan2(v_in[1], v_in[0])
    best, best_ang = None, -np.inf
    for ei in cand_elem_ids:
        e = g.elements[ei]
        other = e[1] if e[0] == head else e[0]
        v_out = g.nodes[other] - g.nodes[head]
        a_out = np.arctan2(v_out[1], v_out[0])
        turn = (a_out - a_in + np.pi) % (2*np.pi) - np.pi   # signed turn, left positive
        if turn > best_ang:
            best_ang, best = turn, ei
    return best


# 5.7.3.2  neighbour / parity rule for island and donut pairs
def resolve_islands_parity(g, seeds, background=0):
    """Kulkarni 3.2.3 parity fix: elements still carrying only one material code (isolated
    islands like 'PS') get their 2nd code from the odd-occurrence rule along a ray.
    If a ray to +infinity exits the brain with no further crossings, the missing side
    is `background` (the element is on the outer brain contour)."""
    for ei, e in enumerate(g.elements):
        if e[3] != -1 or e[2] == -1:
            continue
        mid = (g.nodes[e[0]] + g.nodes[e[1]]) / 2.0
        hits = _ray_intersections(g, mid)
        # crossings strictly beyond this element along the ray
        counts = {}
        for dist, hj in hits:
            if hj == ei:
                continue
            for m in (g.elements[hj][2], g.elements[hj][3]):
                if m not in (-1,):
                    counts[m] = counts.get(m, 0) + 1
        odd = [m for m, c in counts.items() if c % 2 == 1 and m != e[2]]
        if odd:
            e[3] = odd[0]
        else:
            # no enclosing region on the other side -> outer contour -> background
            e[3] = background
    return g


# ---------------------------------------------------------------------------
# Kulkarni 3.2.4  Zipper void loops (overlap artifacts)
# ---------------------------------------------------------------------------
# 5.7.2.4  zipper void loops left by overlap artifacts
def zipper_voids(g):
    """Kulkarni 3.2.4: void loops (elements with one material code still -1) are zippered by
    taking the midline between opposing boundary elements. Simplified: assign the
    nearest well-defined neighbour's outer material."""
    deg = g.node_degree()
    for e in g.elements:
        if e[3] == -1 and e[2] != -1:
            mid = (g.nodes[e[0]] + g.nodes[e[1]]) / 2.0
            # find nearest fully-defined element and borrow its non-shared material
            best = (np.inf, None)
            for f in g.elements:
                if f[3] == -1:
                    continue
                fm = (g.nodes[f[0]] + g.nodes[f[1]]) / 2.0
                d = np.hypot(*(mid - fm))
                if d < best[0]:
                    best = (d, f)
            if best[1] is not None:
                cand = [m for m in (best[1][2], best[1][3]) if m != e[2]]
                if cand:
                    e[3] = cand[0]
    return g


# ---------------------------------------------------------------------------
# Kulkarni 3.2.5  Mirror + ASCII output
# ---------------------------------------------------------------------------
# 6.1.1  reflect the graph across the L-R midline
def mirror_across_midline(g, axis_x):
    """Mirror the (right-half) graph across the SI midline at x=axis_x to make full brain."""
    n0 = len(g.nodes)
    mirrored = g.nodes.copy()
    mirrored[:, 0] = 2*axis_x - mirrored[:, 0]
    g.nodes = np.vstack([g.nodes, mirrored])
    new_elems = []
    for e in g.elements:
        new_elems.append([e[0]+n0, e[1]+n0, e[2], e[3]])
    g.elements.extend(new_elems)
    return g


# 3.1.8  ASCII line file: nodes, then elements
def write_ascii(g, path):
    """Kulkarni 3.2.5 output format: nodes (id x y) then elements (n_i n_j matA matB)."""
    with open(path, "w") as f:
        f.write(f"# NODES {len(g.nodes)}\n")
        for i, (x, y) in enumerate(g.nodes):
            f.write(f"N {i} {x:.6f} {y:.6f}\n")
        f.write(f"# ELEMENTS {len(g.elements)}\n")
        for k, e in enumerate(g.elements):
            f.write(f"E {k} {e[0]} {e[1]} {e[2]} {e[3]}\n")
    return path


# 5.1.2  read a boundary graph back from the ASCII line file
def read_ascii(path):
    g = BoundaryGraph()
    nodes = []
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t or t[0].startswith("#"):
                continue
            if t[0] == "N":
                nodes.append((float(t[2]), float(t[3])))
            elif t[0] == "E":
                g.elements.append([int(t[2]), int(t[3]), int(t[4]), int(t[5])])
    g.nodes = np.array(nodes) if nodes else np.zeros((0, 2))
    return g


# ---------------------------------------------------------------------------
# Full per-slice cleanup driver  (the "line-fixing algorithm")
# ---------------------------------------------------------------------------
# 3.1.3 / 5.7.1  per-slice cleanup driver
def fix_slice(g, seeds, midline_x=None):
    """Run the full section-3.2 sequence on one slice graph."""
    g = clear_clean(g)            # Clear: dedupe
    g = extend_open_nodes(g)      # Extend: open segments
    g = intersect_elements(g)     # Intersect: crossings
    g = merge_redundant_elements(g)  # Merge: within tolerance
    g = identify_regions(g, seeds)   # Loop: assign materials
    g = resolve_islands_parity(g, seeds)
    g = zipper_voids(g)
    if midline_x is not None:
        g = mirror_across_midline(g, midline_x)
    return g


# 5.7.3.4  report elements still carrying an unresolved code
def audit_unresolved(g):
    """Report elements still missing a material code after the full pipeline, so the
    expert can resolve the few ambiguous (usually deeply nested) cases manually."""
    bad = [k for k, e in enumerate(g.elements) if e[2] == -1 or e[3] == -1]
    return bad
