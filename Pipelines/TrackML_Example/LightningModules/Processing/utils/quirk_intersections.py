"""
Quirk hit generation via exact segment-plane intersection against real detector modules.

Each TrackML module is treated as a finite flat surface described by:
  - center (cx, cy, cz) from detectors.csv
  - normal n  = (rot_xw, rot_yw, rot_zw)
  - u-axis    = (rot_xu, rot_yu, rot_zu)   [along module width]
  - v-axis    = (rot_xv, rot_yv, rot_zv)   [along module height]
  - half-widths: module_maxhu (u), module_hv (v)
  - corners X1-X4 (one active face)

For every consecutive trajectory segment A->B the intersection parameter t is
computed analytically:

    denom = dot(n, B-A)
    t     = dot(n, C-A) / denom       if |denom| > eps
    F     = A + t * (B-A)             if 0 <= t <= 1

F is accepted as a hit only when it lies strictly inside the module boundary:

    |local_u| = |dot(F-C, u)| <= module_maxhu
    |local_v| = |dot(F-C, v)| <= module_hv

No tolerance/proximity fudge is used.  SM logic is completely unchanged.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_column(frame, candidates):
    for col in candidates:
        if col in frame.columns:
            return col
    return None


def estimate_module_centers_from_hits(input_dir, max_events=20):
    """Build module centers from TrackML hit CSV files."""
    base = Path(input_dir)
    hits_files = sorted(base.glob("*-hits.csv"))[: int(max(1, max_events))]
    if not hits_files:
        raise FileNotFoundError(
            f"No '*-hits.csv' files found in {input_dir}; cannot estimate module centers."
        )
    frames = []
    for file_path in hits_files:
        frame = pd.read_csv(
            file_path,
            usecols=["x", "y", "z", "volume_id", "layer_id", "module_id"],
        )
        frames.append(frame)
    hits = pd.concat(frames, axis=0, ignore_index=True)
    grouped = (
        hits.groupby(["volume_id", "layer_id", "module_id"], as_index=False)
        .agg(
            cx=("x", "mean"),
            cy=("y", "mean"),
            cz=("z", "mean"),
            spread_x=("x", "std"),
            spread_y=("y", "std"),
            spread_z=("z", "std"),
            n_hits=("x", "size"),
        )
        .fillna(0.0)
    )
    return grouped


def build_detector_module_table(detector_df, module_centers_df=None):
    """
    Build a module-geometry table from detectors.csv.

    Columns produced:
        volume_id, layer_id, module_id, cx, cy, cz,
        ux/uy/uz  (u-axis, normalized),
        vx/vy/vz  (v-axis, normalized),
        nx/ny/nz  (normal, normalized),
        half_u, half_v  (active-surface half-widths in mm),
        module_index
    """
    if module_centers_df is not None:
        table = detector_df.merge(
            module_centers_df[
                ["volume_id", "layer_id", "module_id",
                 "cx", "cy", "cz", "spread_x", "spread_y", "spread_z"]
            ],
            on=["volume_id", "layer_id", "module_id"],
            how="left",
            suffixes=("", "_hits"),
        )
        for coord in ["cx", "cy", "cz"]:
            hit_col = f"{coord}_hits"
            if coord not in table.columns and hit_col in table.columns:
                table[coord] = table[hit_col]
            elif coord in table.columns and hit_col in table.columns:
                table[coord] = table[coord].fillna(table[hit_col])
    else:
        table = detector_df.copy()

    x_col = _pick_column(table, ["cx", "cx_x", "cx_y", "module_x", "x"])
    y_col = _pick_column(table, ["cy", "cy_x", "cy_y", "module_y", "y"])
    z_col = _pick_column(table, ["cz", "cz_x", "cz_y", "module_z", "z"])
    if any(col is None for col in [x_col, y_col, z_col]):
        raise ValueError("Could not resolve detector module centers (cx, cy, cz).")

    keep_cols = ["volume_id", "layer_id", "module_id", x_col, y_col, z_col]
    for extra in [
        "spread_x", "spread_y", "spread_z",
        "rot_xu", "rot_yu", "rot_zu",
        "rot_xv", "rot_yv", "rot_zv",
        "rot_xw", "rot_yw", "rot_zw",
        "module_minhu", "module_maxhu", "module_hv",
    ]:
        if extra in table.columns:
            keep_cols.append(extra)

    table = table[keep_cols].copy()
    table = table.rename(columns={x_col: "cx", y_col: "cy", z_col: "cz"})
    table = table.reset_index(drop=True)
    table["module_index"] = table.index.astype(np.int64)
    table = table.dropna(subset=["cx", "cy", "cz"]).reset_index(drop=True)
    table["module_index"] = table.index.astype(np.int64)

    # --- local frame axes ---
    if {"rot_xu", "rot_yu", "rot_zu"}.issubset(table.columns):
        u = table[["rot_xu", "rot_yu", "rot_zu"]].to_numpy(dtype=np.float64)
    else:
        phi = np.arctan2(table["cy"].to_numpy(), table["cx"].to_numpy())
        u = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=1)

    if {"rot_xv", "rot_yv", "rot_zv"}.issubset(table.columns):
        v = table[["rot_xv", "rot_yv", "rot_zv"]].to_numpy(dtype=np.float64)
    else:
        v = np.tile(np.array([0.0, 0.0, 1.0]), (len(table), 1))

    if {"rot_xw", "rot_yw", "rot_zw"}.issubset(table.columns):
        n = table[["rot_xw", "rot_yw", "rot_zw"]].to_numpy(dtype=np.float64)
    else:
        c = table[["cx", "cy", "cz"]].to_numpy(dtype=np.float64)
        n = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)

    def _norm_rows(arr):
        return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)

    u = _norm_rows(u)
    v = _norm_rows(v)
    n = _norm_rows(n)

    table["ux"], table["uy"], table["uz"] = u[:, 0], u[:, 1], u[:, 2]
    table["vx"], table["vy"], table["vz"] = v[:, 0], v[:, 1], v[:, 2]
    table["nx"], table["ny"], table["nz"] = n[:, 0], n[:, 1], n[:, 2]

    # --- half-widths ---
    if {"module_minhu", "module_maxhu"}.issubset(table.columns):
        half_u_arr = np.maximum(
            np.abs(table["module_minhu"].to_numpy(dtype=np.float64)),
            np.abs(table["module_maxhu"].to_numpy(dtype=np.float64)),
        )
    else:
        spread_x = table["spread_x"].to_numpy(dtype=np.float64) if "spread_x" in table.columns else np.zeros(len(table))
        spread_y = table["spread_y"].to_numpy(dtype=np.float64) if "spread_y" in table.columns else np.zeros(len(table))
        half_u_arr = np.maximum(5.0, np.sqrt(spread_x**2 + spread_y**2) * 2.0)

    if "module_hv" in table.columns:
        half_v_arr = np.abs(table["module_hv"].to_numpy(dtype=np.float64))
    else:
        spread_z = table["spread_z"].to_numpy(dtype=np.float64) if "spread_z" in table.columns else np.zeros(len(table))
        half_v_arr = np.maximum(5.0, np.abs(spread_z) * 2.0)

    table["half_u"] = np.clip(half_u_arr, 2.0, 200.0)
    table["half_v"] = np.clip(half_v_arr, 2.0, 400.0)
    return table


def make_synthetic_detector_modules(
    radii_mm=(220.0, 340.0, 460.0, 620.0, 780.0, 960.0, 1160.0, 1400.0),
    z_layers_mm=(-900.0, -600.0, -300.0, 0.0, 300.0, 600.0, 900.0),
    n_phi=96,
):
    """Minimal synthetic detector for testing without detectors.csv."""
    rows = []
    module_counter = 0
    n_phi = int(max(8, n_phi))
    for layer_i, radius in enumerate(radii_mm):
        for z_i, z in enumerate(z_layers_mm):
            for phi_i in range(n_phi):
                phi = 2.0 * np.pi * (phi_i / n_phi)
                cx = float(radius * np.cos(phi))
                cy = float(radius * np.sin(phi))
                rows.append({
                    "volume_id":  int(8 + (layer_i % 2) * 5),
                    "layer_id":   int(2 + 2 * layer_i),
                    "module_id":  int(module_counter),
                    "cx": cx, "cy": cy, "cz": float(z),
                })
                module_counter += 1
    return build_detector_module_table(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Core intersection function
# ---------------------------------------------------------------------------

def intersect_track_with_modules(
    track_xyz,
    detector_modules,
    tolerance_mm=30.0,    # retained for API compatibility — not used in geometry
    max_hits=64,
    sample_points=600,    # retained for API compatibility — not used
    diagnostic_log=None,  # list; if given, first 20 candidates appended as dicts
):
    """
    Find quirk hits by exact segment-plane intersection against real module geometry.

    Algorithm
    ---------
    For each consecutive segment A -> B in track_xyz:
      1. KD-tree query finds modules whose centers lie within
            search_r = |AB|/2 + max_module_diagonal
         of the segment midpoint.
      2. For each candidate module (vectorised over all candidates per segment):
            denom = dot(n, B-A)
            t     = dot(n, C-A) / denom       [skip if |denom| < 1e-10]
            F     = A + t*(B-A)               [skip if t outside [0, 1]]
      3. F is projected into the module local frame:
            local_u = dot(F-C, u_axis)
            local_v = dot(F-C, v_axis)
         Hit accepted iff |local_u| <= half_u AND |local_v| <= half_v.
      4. The hit position recorded is F (exact plane crossing), not a
         sampled trajectory point.
      5. trajectory_step = segment index i, used for time-ordered true edges.

    Parameters
    ----------
    track_xyz : array-like, shape (N, 3)  [mm]
    detector_modules : DataFrame from build_detector_module_table()
    tolerance_mm : ignored (kept for API compatibility with event_utils.py)
    max_hits : maximum hits to record per track
    sample_points : ignored (kept for API compatibility)
    diagnostic_log : if a list is supplied, up to 20 dicts are appended with
        keys: module_index, volume_id, layer_id, module_id, X1..X4, t, F,
        local_u_mm, local_v_mm, half_u_mm, half_v_mm, inside, trajectory_step.
        Both accepted and rejected candidates are logged (inside=True/False).

    Returns
    -------
    pd.DataFrame with columns:
        hit_id, trajectory_step, x, y, z, r, phi,
        volume_id, layer_id, module_id, module_index,
        local_u_mm, local_v_mm, distance_to_module (=0 for exact intersections)
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        raise ImportError(
            "scipy is required for geometry-based hit finding. "
            "Install with: conda install scipy  or  pip install scipy"
        )

    if len(track_xyz) < 2:
        return pd.DataFrame()

    track_xyz = np.asarray(track_xyz, dtype=np.float64)
    n_steps   = len(track_xyz)
    max_hits  = int(max_hits)

    # Extract geometry arrays once
    centers = detector_modules[["cx", "cy", "cz"]].to_numpy(dtype=np.float64)
    normals = detector_modules[["nx", "ny", "nz"]].to_numpy(dtype=np.float64)
    uvec    = detector_modules[["ux", "uy", "uz"]].to_numpy(dtype=np.float64)
    vvec    = detector_modules[["vx", "vy", "vz"]].to_numpy(dtype=np.float64)
    half_u  = detector_modules["half_u"].to_numpy(dtype=np.float64)
    half_v  = detector_modules["half_v"].to_numpy(dtype=np.float64)

    # Spatial index on module centers (built once per call; cache externally if hot)
    tree            = cKDTree(centers)
    max_module_diag = float(np.sqrt(half_u.max()**2 + half_v.max()**2)) + 5.0

    rows          = []
    n_diag_logged = 0
    last_hit_step = {}  # mod_i -> last trajectory step that produced a hit
    min_gap       = 5   # steps: prevents double-counting a single crossing

    for i in range(n_steps - 1):
        A  = track_xyz[i]
        B  = track_xyz[i + 1]
        # Skip segments with NaN/inf coordinates (numerical overflow in simulator tail)
        if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))):
            continue
        AB = B - A
        seg_len = float(np.linalg.norm(AB))
        if seg_len < 1e-12:
            continue

        # --- candidate modules near this segment ---
        mid      = 0.5 * (A + B)
        search_r = 0.5 * seg_len + max_module_diag
        cand_ids = np.asarray(tree.query_ball_point(mid, r=search_r), dtype=np.int64)
        if cand_ids.size == 0:
            continue

        # --- vectorised segment-plane intersection for all candidates ---
        n_c    = normals[cand_ids]         # (nc, 3)
        C_c    = centers[cand_ids]         # (nc, 3)
        denoms = n_c @ AB                  # (nc,)  dot(n, B-A)
        valid  = np.abs(denoms) > 1e-10
        if not np.any(valid):
            continue

        CA     = C_c - A[None, :]          # (nc, 3)
        numer  = np.einsum('ij,ij->i', n_c, CA)
        t_vals = np.where(valid, numer / np.where(valid, denoms, 1.0), -1.0)
        in_seg = valid & (t_vals >= 0.0) & (t_vals <= 1.0)
        if not np.any(in_seg):
            continue

        hit_ids = cand_ids[in_seg]         # module indices with valid t
        hit_t   = t_vals[in_seg]
        F_all   = A[None, :] + hit_t[:, None] * AB[None, :]  # (nh, 3)

        # --- local-frame projection and boundary check ---
        d_all  = F_all - centers[hit_ids]
        lu_all = np.einsum('ij,ij->i', d_all, uvec[hit_ids])
        lv_all = np.einsum('ij,ij->i', d_all, vvec[hit_ids])
        inside = (
            (np.abs(lu_all) <= half_u[hit_ids]) &
            (np.abs(lv_all) <= half_v[hit_ids])
        )

        # --- diagnostics (before boundary filter) ---
        if diagnostic_log is not None and n_diag_logged < 20:
            for k in range(len(hit_ids)):
                mod_i = int(hit_ids[k])
                hu    = float(half_u[mod_i])
                hv    = float(half_v[mod_i])
                u_ax  = uvec[mod_i]
                v_ax  = vvec[mod_i]
                Cm    = centers[mod_i]
                diagnostic_log.append({
                    "module_index":  mod_i,
                    "volume_id":     int(detector_modules.iloc[mod_i].volume_id),
                    "layer_id":      int(detector_modules.iloc[mod_i].layer_id),
                    "module_id":     int(detector_modules.iloc[mod_i].module_id),
                    # corners of active surface
                    "X1": (Cm + hu * u_ax + hv * v_ax).tolist(),
                    "X2": (Cm - hu * u_ax + hv * v_ax).tolist(),
                    "X3": (Cm - hu * u_ax - hv * v_ax).tolist(),
                    "X4": (Cm + hu * u_ax - hv * v_ax).tolist(),
                    "t":             float(hit_t[k]),
                    "F":             F_all[k].tolist(),
                    "local_u_mm":    float(lu_all[k]),
                    "local_v_mm":    float(lv_all[k]),
                    "half_u_mm":     hu,
                    "half_v_mm":     hv,
                    "inside":        bool(inside[k]),
                    "trajectory_step": i,
                })
                n_diag_logged += 1
                if n_diag_logged >= 20:
                    break

        # --- record accepted hits ---
        for k in np.where(inside)[0]:
            mod_i     = int(hit_ids[k])
            last_step = last_hit_step.get(mod_i, -9999)
            if i - last_step < min_gap:
                continue  # same crossing, deduplicate
            last_hit_step[mod_i] = i

            F   = F_all[k]
            x, y, z = float(F[0]), float(F[1]), float(F[2])
            r   = float(np.sqrt(x * x + y * y))
            phi = float(np.arctan2(y, x))
            mod = detector_modules.iloc[mod_i]

            rows.append({
                "hit_id":            len(rows),
                "trajectory_step":   int(i),
                "x": x, "y": y, "z": z,
                "r": r, "phi": phi,
                "volume_id":         int(mod.volume_id),
                "layer_id":          int(mod.layer_id),
                "module_id":         int(mod.module_id),
                "module_index":      int(mod.module_index),
                "local_u_mm":        float(lu_all[k]),
                "local_v_mm":        float(lv_all[k]),
                "distance_to_module": 0.0,   # exact intersection, not proximity
            })

            if len(rows) >= max_hits:
                break

        if len(rows) >= max_hits:
            break

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Diagnostic utilities
# ---------------------------------------------------------------------------

def print_diagnostic_log(diag_log, max_entries=10):
    """
    Pretty-print diagnostic records from intersect_track_with_modules.

    Shows module corners, intersection point F, local coordinates, and
    whether F is inside the module boundary.
    """
    print(f"\n{'='*70}")
    print(f"  Intersection diagnostics  ({min(len(diag_log), max_entries)} of {len(diag_log)} candidates)")
    print(f"{'='*70}")
    for rec in diag_log[:max_entries]:
        status = "INSIDE  [HIT]" if rec["inside"] else "OUTSIDE [rejected]"
        print(
            f"\nModule {rec['module_index']:5d}  "
            f"vol={rec['volume_id']}  lay={rec['layer_id']}  mod={rec['module_id']}"
        )
        print(f"  Corners X1-X4 (active surface):")
        for lbl, key in [("X1", "X1"), ("X2", "X2"), ("X3", "X3"), ("X4", "X4")]:
            pt = rec[key]
            print(f"    {lbl}: ({pt[0]:8.2f}, {pt[1]:8.2f}, {pt[2]:8.2f})  mm")
        F = rec["F"]
        print(f"  Segment param  t     = {rec['t']:.4f}  (0=start, 1=end)")
        print(f"  Intersection   F     = ({F[0]:8.2f}, {F[1]:8.2f}, {F[2]:8.2f})  mm")
        print(f"  local_u        = {rec['local_u_mm']:8.2f} mm   (limit ±{rec['half_u_mm']:.2f} mm)")
        print(f"  local_v        = {rec['local_v_mm']:8.2f} mm   (limit ±{rec['half_v_mm']:.2f} mm)")
        print(f"  --> {status}")
    print(f"\n{'='*70}\n")


def compute_before_after_stats(track_xyz, detector_modules,
                                max_hits=150, sample_points=800,
                                tolerance_mm=1000.0):
    """
    Run both the old proximity-based and new exact-intersection methods on a
    single track and return a summary dict for reporting.

    Used by the diagnostic script to generate before/after statistics across
    multiple events and files.
    """
    n_steps = len(track_xyz)

    # --- OLD method (proximity, sampled points) ---
    # Inline reproduction of the old sampling logic to avoid needing the old file.
    track_xyz_arr = np.asarray(track_xyz, dtype=np.float64)
    centers = detector_modules[["cx", "cy", "cz"]].to_numpy(dtype=np.float64)
    normals = detector_modules[["nx", "ny", "nz"]].to_numpy(dtype=np.float64)
    uvec_   = detector_modules[["ux", "uy", "uz"]].to_numpy(dtype=np.float64)
    vvec_   = detector_modules[["vx", "vy", "vz"]].to_numpy(dtype=np.float64)
    half_u_ = detector_modules["half_u"].to_numpy(dtype=np.float64)
    half_v_ = detector_modules["half_v"].to_numpy(dtype=np.float64)

    sp = int(max(10, min(sample_points, n_steps)))
    sample_idx   = np.linspace(0, n_steps - 1, sp).astype(np.int64)
    sampled_xyz  = track_xyz_arr[sample_idx]
    tol          = float(tolerance_mm)
    coarse_tol2  = (4.0 * tol) ** 2

    old_hits = 0
    last_mod = None
    for p in sampled_xyz:
        delta = p[None, :] - centers
        d2    = np.sum(delta * delta, axis=1)
        cands = np.where(d2 <= coarse_tol2)[0]
        if len(cands) == 0:
            cands = np.array([int(np.argmin(d2))], dtype=np.int64)
        best = None
        best_score = None
        for mod_i in cands:
            d = delta[mod_i]
            pd_ = abs(float(np.dot(d, normals[mod_i])))
            if pd_ > tol:
                continue
            du = abs(float(np.dot(d, uvec_[mod_i])))
            dv = abs(float(np.dot(d, vvec_[mod_i])))
            if du > float(half_u_[mod_i] + tol) or dv > float(half_v_[mod_i] + tol):
                continue
            score = pd_ + 0.01 * (du + dv)
            if best_score is None or score < best_score:
                best_score = score
                best = int(mod_i)
        if best is None:
            continue
        if last_mod == best:
            continue
        old_hits += 1
        last_mod  = best
        if old_hits >= max_hits:
            break

    # --- NEW method (exact geometry) ---
    diag_log = []
    new_df   = intersect_track_with_modules(
        track_xyz_arr, detector_modules,
        max_hits=max_hits, diagnostic_log=diag_log,
    )
    new_hits         = len(new_df)
    n_candidates     = len(diag_log)
    n_inside         = sum(1 for r in diag_log if r["inside"])
    n_outside        = n_candidates - n_inside

    return {
        "n_trajectory_steps":   n_steps,
        "n_segments":           max(0, n_steps - 1),
        "old_sample_points":    sp,
        "old_hits":             old_hits,
        "new_hits":             new_hits,
        "n_diag_candidates":    n_candidates,
        "n_inside_boundary":    n_inside,
        "n_outside_boundary":   n_outside,
        "diag_log":             diag_log,
    }
