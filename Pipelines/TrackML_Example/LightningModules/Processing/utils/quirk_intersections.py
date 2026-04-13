import numpy as np
import pandas as pd
from pathlib import Path


def _pick_column(frame, candidates):
    for col in candidates:
        if col in frame.columns:
            return col
    return None


def estimate_module_centers_from_hits(input_dir, max_events=20):
    """
    Build module centers from TrackML hit CSV files.
    """
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
    Build a module-center table from detector geometry.
    """
    if module_centers_df is not None:
        table = detector_df.merge(
            module_centers_df[
                ["volume_id", "layer_id", "module_id", "cx", "cy", "cz", "spread_x", "spread_y", "spread_z"]
            ],
            on=["volume_id", "layer_id", "module_id"],
            how="left",
            suffixes=("", "_hits"),
        )
        # Prefer detector geometry centers when available; fallback to hit-derived centers.
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
        "spread_x",
        "spread_y",
        "spread_z",
        "rot_xu",
        "rot_yu",
        "rot_zu",
        "rot_xv",
        "rot_yv",
        "rot_zv",
        "rot_xw",
        "rot_yw",
        "rot_zw",
        "module_minhu",
        "module_maxhu",
        "module_hv",
    ]:
        if extra in table.columns:
            keep_cols.append(extra)

    table = table[keep_cols].copy()
    table = table.rename(columns={x_col: "cx", y_col: "cy", z_col: "cz"})
    table = table.reset_index(drop=True)
    table["module_index"] = table.index.astype(np.int64)
    table = table.dropna(subset=["cx", "cy", "cz"]).reset_index(drop=True)
    table["module_index"] = table.index.astype(np.int64)

    # Build local frame and module dimensions.
    if {"rot_xu", "rot_yu", "rot_zu"}.issubset(table.columns):
        u = table[["rot_xu", "rot_yu", "rot_zu"]].to_numpy(dtype=np.float32)
    else:
        # Tangential approximation in transverse plane.
        phi = np.arctan2(table["cy"].to_numpy(), table["cx"].to_numpy())
        u = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=1).astype(np.float32)

    if {"rot_xv", "rot_yv", "rot_zv"}.issubset(table.columns):
        v = table[["rot_xv", "rot_yv", "rot_zv"]].to_numpy(dtype=np.float32)
    else:
        v = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(table), 1))

    if {"rot_xw", "rot_yw", "rot_zw"}.issubset(table.columns):
        n = table[["rot_xw", "rot_yw", "rot_zw"]].to_numpy(dtype=np.float32)
    else:
        c = table[["cx", "cy", "cz"]].to_numpy(dtype=np.float32)
        n = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)

    def _norm_rows(arr):
        return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)

    u = _norm_rows(u)
    v = _norm_rows(v)
    n = _norm_rows(n)

    table["ux"], table["uy"], table["uz"] = u[:, 0], u[:, 1], u[:, 2]
    table["vx"], table["vy"], table["vz"] = v[:, 0], v[:, 1], v[:, 2]
    table["nx"], table["ny"], table["nz"] = n[:, 0], n[:, 1], n[:, 2]

    if {"module_minhu", "module_maxhu"}.issubset(table.columns):
        half_u = np.maximum(
            np.abs(table["module_minhu"].to_numpy(dtype=np.float32)),
            np.abs(table["module_maxhu"].to_numpy(dtype=np.float32)),
        )
    else:
        spread_x = (
            table["spread_x"].to_numpy(dtype=np.float32)
            if "spread_x" in table.columns
            else np.zeros(len(table), dtype=np.float32)
        )
        spread_y = (
            table["spread_y"].to_numpy(dtype=np.float32)
            if "spread_y" in table.columns
            else np.zeros(len(table), dtype=np.float32)
        )
        half_u = np.maximum(5.0, np.sqrt(spread_x**2 + spread_y**2) * 2.0)

    if "module_hv" in table.columns:
        half_v = np.abs(table["module_hv"].to_numpy(dtype=np.float32))
    else:
        spread_z = (
            table["spread_z"].to_numpy(dtype=np.float32)
            if "spread_z" in table.columns
            else np.zeros(len(table), dtype=np.float32)
        )
        half_v = np.maximum(5.0, np.abs(spread_z) * 2.0)

    table["half_u"] = np.clip(half_u, 2.0, 200.0)
    table["half_v"] = np.clip(half_v, 2.0, 400.0)
    return table


def make_synthetic_detector_modules(
    radii_mm=(220.0, 340.0, 460.0, 620.0, 780.0, 960.0, 1160.0, 1400.0),
    z_layers_mm=(-900.0, -600.0, -300.0, 0.0, 300.0, 600.0, 900.0),
    n_phi=96,
):
    """
    Build a simple synthetic detector module table when no detector CSV is available.
    """
    rows = []
    module_counter = 0
    n_phi = int(max(8, n_phi))
    for layer_i, radius in enumerate(radii_mm):
        for z_i, z in enumerate(z_layers_mm):
            for phi_i in range(n_phi):
                phi = 2.0 * np.pi * (phi_i / n_phi)
                cx = float(radius * np.cos(phi))
                cy = float(radius * np.sin(phi))
                rows.append(
                    {
                        "volume_id": int(8 + (layer_i % 2) * 5),
                        "layer_id": int(2 + 2 * layer_i),
                        "module_id": int(module_counter),
                        "cx": cx,
                        "cy": cy,
                        "cz": float(z),
                    }
                )
                module_counter += 1
    return build_detector_module_table(pd.DataFrame(rows))


def intersect_track_with_modules(
    track_xyz,
    detector_modules,
    tolerance_mm=30.0,
    max_hits=64,
    sample_points=600,
):
    """
    Approximate intersections against module planes and finite module bounds.
    """
    if len(track_xyz) == 0:
        return pd.DataFrame()

    n_track = len(track_xyz)
    sample_points = int(max(10, min(sample_points, n_track)))
    sample_idx = np.linspace(0, n_track - 1, sample_points).astype(np.int64)
    sampled_xyz = track_xyz[sample_idx]

    centers = detector_modules[["cx", "cy", "cz"]].to_numpy(dtype=np.float32)
    normals = detector_modules[["nx", "ny", "nz"]].to_numpy(dtype=np.float32)
    uvec = detector_modules[["ux", "uy", "uz"]].to_numpy(dtype=np.float32)
    vvec = detector_modules[["vx", "vy", "vz"]].to_numpy(dtype=np.float32)
    half_u = detector_modules["half_u"].to_numpy(dtype=np.float32)
    half_v = detector_modules["half_v"].to_numpy(dtype=np.float32)

    selected = []
    last_module = None
    max_hits = int(max_hits)
    tol = float(tolerance_mm)
    coarse_tol2 = (4.0 * tol) ** 2

    for idx, p in zip(sample_idx, sampled_xyz):
        delta = p[None, :] - centers
        d2 = np.sum(delta * delta, axis=1)
        candidates = np.where(d2 <= coarse_tol2)[0]
        if len(candidates) == 0:
            candidates = np.array([int(np.argmin(d2))], dtype=np.int64)

        best = None
        best_score = None
        for mod_i in candidates:
            d = delta[mod_i]
            plane_dist = abs(float(np.dot(d, normals[mod_i])))
            if plane_dist > tol:
                continue
            du = abs(float(np.dot(d, uvec[mod_i])))
            dv = abs(float(np.dot(d, vvec[mod_i])))
            if du > float(half_u[mod_i] + tol) or dv > float(half_v[mod_i] + tol):
                continue
            score = plane_dist + 0.01 * (du + dv)
            if best_score is None or score < best_score:
                best_score = score
                best = (int(mod_i), plane_dist)

        if best is None:
            continue

        mod_i, dist = best
        if last_module == mod_i:
            continue
        selected.append((int(idx), mod_i, float(dist)))
        last_module = mod_i
        if len(selected) >= max_hits:
            break

    if len(selected) < 2:
        return pd.DataFrame()

    rows = []
    for hit_id, (track_i, module_i, dist) in enumerate(selected):
        mod = detector_modules.iloc[module_i]
        x, y, z = track_xyz[track_i]
        r = float(np.sqrt(x * x + y * y))
        phi = float(np.arctan2(y, x))
        rows.append(
            {
                "hit_id": int(hit_id),
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "r": r,
                "phi": phi,
                "volume_id": int(mod.volume_id),
                "layer_id": int(mod.layer_id),
                "module_id": int(mod.module_id),
                "module_index": int(mod.module_index),
                "distance_to_module": float(dist),
            }
        )
    return pd.DataFrame(rows)
