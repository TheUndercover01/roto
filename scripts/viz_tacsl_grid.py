"""Standalone visualizer for TacSL's tactile-grid generation on a fingertip mesh.

Replicates ``VisuoTactileSensor._generate_tactile_grid()`` EXACTLY (same
slim-axis heuristic, same ray casting), but offline with trimesh — no Isaac Sim.

Why this matches sim: TacSL reads ``UsdGeom.Mesh.GetPointsAttr()``, which are the
raw mesh points = ``fingertip_v5.stl`` scaled by the URDF's 0.001. The URDF
visual ``<origin xyz="0 0 0.02688">`` becomes a parent Xform and is NOT baked
into the points, so STL-verts * 0.001 reproduces the sim grid one-to-one.

Usage:
    python scripts/viz_tacsl_grid.py                       # current sim mapping (auto slim-axis)
    python scripts/viz_tacsl_grid.py --axis y --side max   # preview grid on a chosen face
    python scripts/viz_tacsl_grid.py --no-show             # print summary only
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import trimesh

_DEFAULT_STL = (
    Path(__file__).resolve().parent.parent
    / "roto/assets/shadow_lite/meshes/touchlab/fingertip_v5.stl"
)
_AXES = {"x": 0, "y": 1, "z": 2}
_AXIS_NAME = ["X", "Y", "Z"]


def build_grid(mesh: trimesh.Trimesh, num_divs, margin, axis_override, side_override):
    """Exact port of TacSL _generate_tactile_grid (with optional axis/side overrides)."""
    points = mesh.vertices
    mesh_bounds = np.array([points.min(axis=0), points.max(axis=0)])
    elastomer_dims = np.diff(mesh_bounds, axis=0).squeeze()

    slim_axis = int(np.argmin(elastomer_dims)) if axis_override is None else axis_override

    if side_override is None:
        com = mesh.center_mass[slim_axis]
        bbox_center = (mesh_bounds[0, slim_axis] + mesh_bounds[1, slim_axis]) / 2.0
        tip_direction_sign = 1.0 if com > bbox_center else -1.0
    else:
        tip_direction_sign = 1.0 if side_override == "max" else -1.0

    axis_idxs = [a for a in range(3) if a != slim_axis]
    div_sz = (elastomer_dims[axis_idxs] - margin * 2.0) / (np.array(num_divs) + 1)
    tactile_points_dx = float(min(div_sz))

    center = (mesh_bounds[0] + mesh_bounds[1]) / 2.0
    planar_grid_points = []
    idx = 0
    for axis_i in range(3):
        if axis_i == slim_axis:
            planar_grid_points.append([tip_direction_sign])
        else:
            g = np.linspace(
                center[axis_i] - tactile_points_dx * (num_divs[idx] + 1.0) / 2.0,
                center[axis_i] + tactile_points_dx * (num_divs[idx] + 1.0) / 2.0,
                num_divs[idx] + 2,
            )
            planar_grid_points.append(g[1:-1])
            idx += 1

    grid_corners = np.array(list(itertools.product(*planar_grid_points)))
    ray_dir = np.zeros(3)
    ray_dir[slim_axis] = -tip_direction_sign

    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    _, index_ray, locations = intersector.intersects_id(
        grid_corners,
        np.tile([ray_dir], (grid_corners.shape[0], 1)),
        return_locations=True,
        multiple_hits=False,
    )
    return dict(
        slim_axis=slim_axis,
        tip_direction_sign=tip_direction_sign,
        ray_dir=ray_dir,
        elastomer_dims=elastomer_dims,
        mesh_bounds=mesh_bounds,
        grid_corners=grid_corners,
        index_ray=index_ray,
        locations=locations,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stl", type=str, default=str(_DEFAULT_STL))
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--margin", type=float, default=0.001)
    ap.add_argument("--axis", choices=["auto", "x", "y", "z"], default="auto",
                    help="Override slim axis (sim default = auto = argmin of bbox).")
    ap.add_argument("--side", choices=["auto", "min", "max"], default="auto",
                    help="Which face along the axis to project onto.")
    ap.add_argument("--no-show", action="store_true", help="Print summary only, no 3D window.")
    args = ap.parse_args()

    mesh = trimesh.load(args.stl, process=False)
    mesh.apply_scale(0.001)  # URDF scale mm -> m; reproduces USD GetPointsAttr()

    axis_override = None if args.axis == "auto" else _AXES[args.axis]
    side_override = None if args.side == "auto" else args.side
    num_divs = [args.rows, args.cols]

    r = build_grid(mesh, num_divs, args.margin, axis_override, side_override)
    sa = r["slim_axis"]
    n_total = r["grid_corners"].shape[0]
    n_hits = r["locations"].shape[0]
    dims_mm = r["elastomer_dims"] * 1000.0

    print("=" * 64)
    print(f"mesh: {args.stl}")
    print(f"bbox dims (mm):  X={dims_mm[0]:.2f}  Y={dims_mm[1]:.2f}  Z={dims_mm[2]:.2f}")
    print(f"slim_axis:       {_AXIS_NAME[sa]} (axis {sa})"
          f"{'  [AUTO=argmin]' if axis_override is None else '  [OVERRIDE]'}")
    print(f"tip_dir_sign:    {r['tip_direction_sign']:+.0f}   ray_dir: {r['ray_dir']}")
    print(f"grid:            {args.rows}x{args.cols} = {n_total} rays, margin={args.margin}")
    print(f"ray hits:        {n_hits}/{n_total}"
          f"{'  <-- MISSES (some corners off the mesh silhouette)' if n_hits < n_total else ''}")
    if n_hits:
        hb = np.array([r["locations"].min(0), r["locations"].max(0)]) * 1000.0
        print(f"hit patch (mm):  X[{hb[0,0]:.1f},{hb[1,0]:.1f}] "
              f"Y[{hb[0,1]:.1f},{hb[1,1]:.1f}] Z[{hb[0,2]:.1f},{hb[1,2]:.1f}]")
        face = "max" if r["tip_direction_sign"] > 0 else "min"
        print(f"landed on:       {_AXIS_NAME[sa]}-{face} face of the fingertip")
    print("=" * 64)

    if args.no_show or n_hits == 0:
        return

    mesh.visual.face_colors = [180, 180, 185, 90]  # translucent gray
    scene_geom = [mesh]

    # Hit points as clearly visible blue spheres.
    for loc in r["locations"]:
        s = trimesh.creation.uv_sphere(radius=0.0009)
        s.apply_translation(loc)
        s.visual.face_colors = [30, 90, 255, 255]
        scene_geom.append(s)

    # Ray segments: from a plane just outside the hit face, to each hit.
    sign = r["tip_direction_sign"]
    plane = r["mesh_bounds"][1 if sign > 0 else 0, sa] + sign * 0.004
    segs = []
    for loc in r["locations"]:
        start = loc.copy()
        start[sa] = plane
        segs.append([start, loc])
    if segs:
        path = trimesh.load_path(np.array(segs))
        path.colors = np.tile([0, 200, 0, 255], (len(segs), 1))
        scene_geom.append(path)

    trimesh.Scene(scene_geom).show()


if __name__ == "__main__":
    main()
