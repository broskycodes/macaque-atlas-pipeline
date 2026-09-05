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

TRIMMED (v12): the section-3.2 geometry CLEANUP driver (fix_slice) and everything it chained --
spline->polyline, Extend, Intersect, Merge, the marching-line region identifier (Loop), parity
islands, zipper voids, and the midline mirror -- moved to bg_legacy.py. In THIS pipeline the graph
arrives already built and code-labelled from the tracing cell, and edge_fusion.py does its own
arc/node fusion, so none of those were called. KEPT here (still used by the notebook): the
BoundaryGraph data model, and write_ascii / read_ascii (the on-disk line format CELL 5/6/7 use).
Resurrect the from-scratch region identification with `import bg_legacy`.
"""
import numpy as np
from dataclasses import dataclass, field
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class BoundaryGraph:
    """Planar straight-line graph of one slice."""
    nodes: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))   # (Nx2) xy
    # elements: list of [node_i, node_j, matA, matB]; mat codes -1 == unknown
    elements: list = field(default_factory=list)

    _grid: dict = field(default_factory=dict, repr=False, compare=False)
    _pts:  list = field(default_factory=list, repr=False, compare=False)

    # boundary_graph.py, INSIDE class BoundaryGraph — replace the first 5 lines of add_node
    def add_node(self, xy, tol=1e-6):
        xy = np.asarray(xy, float)
        if not self._pts:                      # was: hasattr(self,"_grid") -- ALWAYS True
            self._grid = {}
            for j, p in enumerate(self.nodes):
                self._grid.setdefault((round(p[0]/tol), round(p[1]/tol)), []).append(j)
            self._pts = [np.asarray(p, float) for p in self.nodes]
        cell = (round(float(xy[0])/tol), round(float(xy[1])/tol))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in self._grid.get((cell[0]+dx, cell[1]+dy), ()):
                    if np.hypot(*(self._pts[j] - xy)) <= tol:
                        return j
        j = len(self._pts)
        self._pts.append(xy)
        self._grid.setdefault(cell, []).append(j)
        return j                                # NOTE: .nodes is now DEFERRED

# ADD this method directly below add_node
    def finalize_nodes(self):
        """Materialise .nodes from the pending list ONCE, after a build loop."""
        if self._pts:
            self.nodes = np.asarray(self._pts, float).reshape(-1, 2)
        return self

    def add_element(self, i, j, matA=-1, matB=-1):
        if i == j:
            return
        self.elements.append([i, j, matA, matB])

    def node_degree(self):
        deg = np.zeros(len(self.nodes), int)
        for e in self.elements:
            deg[e[0]] += 1; deg[e[1]] += 1
        return deg


def write_ascii(g, path):
    """3.2.5 output format: nodes (id x y) then elements (n_i n_j matA matB)."""
    with open(path, "w") as f:
        f.write(f"# NODES {len(g.nodes)}\n")
        for i, (x, y) in enumerate(g.nodes):
            f.write(f"N {i} {x:.6f} {y:.6f}\n")
        f.write(f"# ELEMENTS {len(g.elements)}\n")
        for k, e in enumerate(g.elements):
            f.write(f"E {k} {e[0]} {e[1]} {e[2]} {e[3]}\n")
    return path




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
