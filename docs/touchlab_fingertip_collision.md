# TouchLab Fingertip Collision: Box → Mesh

## Why this change

`sr_hand_mimic_touchlab.urdf` previously used a **box** primitive as the collision
shape for the four distal links (`rh_ffdistal`, `rh_mfdistal`, `rh_rfdistal`,
`rh_thdistal`) while the *visual* was the real TouchLab `fingertip_v5.stl`.

A box makes contact occur on flat faces and sharp edges instead of the rounded
sensing pad. That distorts contact onset, force magnitude, and force direction —
directly inflating the sim-vs-real tactile gap we are trying to measure.

The original (pre‑TouchLab) `sr_hand_mimic.urdf` did **not** use a box: each distal
link's `<collision>` was the PST fingertip mesh itself, relying on Isaac Lab's default
`collider_type="convex_hull"`. This change restores that exact approach, but with the
TouchLab fingertip mesh in place of both the box and the old PST mesh.

## What changed

### 1. `roto/roto/assets/shadow_lite/sr_hand_mimic_touchlab.urdf`

For every distal link, the `<collision>` block went from a box to the TouchLab mesh,
**identical geometry and origin as the `<visual>`** (mirroring how the original PST
link had `collision == visual`):

**Before**
```xml
<collision>
  <origin xyz="0 0 0.01375"/>
  <geometry>
    <box size="0.021 0.02123 0.0275"/>
  </geometry>
</collision>
```

**After**
```xml
<collision>
  <origin xyz="0 0 0.02688" rpy="0 0 0"/>
  <geometry>
    <mesh filename="meshes/touchlab/fingertip_v5.stl" scale="0.001 0.001 0.001"/>
  </geometry>
</collision>
```

> **Mesh: full-resolution v5, scale 0.001.** Uses `fingertip_v5.stl` (full-res,
> 83.5k tris) — *not* `fingertip_v5_simple.stl`. The two are geometrically
> identical (Δ = 0 on every axis) but exported in different units, so the full-res
> file requires `scale="0.001 0.001 0.001"` (the simple one needed `0.1`). Using
> the wrong scale silently makes the fingertip 10–100× the correct size.
>
> **Mesh path must be relative and in-tree.** The STL is copied to
> `roto/roto/assets/shadow_lite/meshes/touchlab/fingertip_v5.stl` and
> referenced by the **relative** path `meshes/touchlab/fingertip_v5.stl`
> (both visual and collision), mirroring how the PST mesh is referenced. An
> absolute path into the separate `touchlab_description/` repo loads in Isaac
> Lab's converter but **silently fails in standalone URDF viewers**, which
> sandbox file access to the URDF's own directory — that is why the collision
> mesh did not appear in the URDF viewer. Keep the STL inside the asset tree.

- The collision `<origin>` is pinned **equal to the visual `<origin>`**
  (`xyz="0 0 0.02688"`, no rotation). That pose was validated in earlier work: the
  STL's large open mount seats on the FFJ1 joint axis, the rounded tip points
  distally (+z), and the sensing pad faces −y (palmar). Collision now coincides
  exactly with what you see.
- `<visual>`, `<inertial>` (mass `0.009` kg, 9 g box‑approx inertia tensor) and every
  other link/joint are unchanged. Inertia is a separate spec from collision and was
  intentionally left as the 9 g solid‑box approximation.
- The URDF is regenerated from the pristine `sr_hand_mimic.urdf` (the same
  regeneration pattern used throughout this work) so only the four distal-link
  blocks differ from stock.

### 2. `roto/roto/assets/shadow_hand_lite.py`

Added one explicit line to the active `SHADOW_HAND_LITE_CFG` →
`spawn=sim_utils.UrdfFileCfg(...)`:

```python
collider_type="convex_hull",
```

`UrdfConverterCfg.collider_type` (Isaac Lab 2.3.2,
`isaaclab/sim/converters/urdf_converter_cfg.py`) is
`Literal["convex_hull", "convex_decomposition"]`, default `"convex_hull"`. PhysX
cannot use a raw triangle mesh on an articulation link, so the URDF `<mesh>` collision
is convexified at import time. This line does not change behavior (it is already the
default — and exactly what the PST fingertip silently used); it makes the intent
explicit and robust against any future Isaac Lab default change.

**Why convex hull (not convex decomposition):** the TouchLab fingertip's only
concavity is the internal mounting socket, which never contacts anything externally.
The convex hull of the outer sensing surface is therefore geometrically faithful for
contact, while staying stable and fast — identical to the proven PST setup.

### 3. Cached USD deleted

Isaac Lab caches the URDF→USD conversion at
`roto/assets/shadow_lite/sr_hand_touch.usd` (the active config's `usd_file_name`).
A collision change is invisible until that cache is rebuilt, so it was deleted:

```bash
rm -f roto/roto/assets/shadow_lite/sr_hand_touch.usd
```

Isaac Lab regenerates it automatically on the next launch. **Any time you edit the
URDF you must delete this USD again**, otherwise the stale cache is loaded.

## How to verify

1. **Structure** — parse the URDF; every `rh_(ff|mf|rf|th)distal` has
   `collision/geometry/mesh` (no `box`), `scale="0.001 0.001 0.001"`, and collision
   `<origin> xyz` equals the visual `<origin> xyz` (`0 0 0.02688`). (Confirmed.)
2. **Reconversion** — with the USD deleted, run
   `python scripts/collect_sim2real_data.py --with_ball --seed 0 --no_save`; Isaac Lab
   should log a fresh URDF→USD conversion with no collision/convex-hull import errors.
3. **Visual = collision** — in the Isaac Sim viewport, enable collision-mesh display;
   the convex hull should hug the rounded fingertip, not a box.
4. **Contact behavior** — run the live monitor
   (`--live_port 9870` + `python scripts/live_tactile_monitor.py --port 9870`);
   closing on the ball should register contact on the rounded sensing surface with a
   smoother onset than the box's edge/face hits, and `--no_ball` free-air stays flat.

## Re-collect after this change

Because the collision geometry changed, regenerate the sim NPZs before any new
sim-vs-real comparison (otherwise you compare hardware against stale box-shaped sim
tactile):

```bash
bash scripts/run_tactile_collection.sh
```
