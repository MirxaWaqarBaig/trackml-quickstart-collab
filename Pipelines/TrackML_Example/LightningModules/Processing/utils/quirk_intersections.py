import numpy as np
import pandas as pd


def _pick_column(frame, candidates):
    for col in candidates:
        if col in frame.columns:
            return col
    return None


def build_detector_module_table(detector_df):
    """
    Build a module-center table from detector geometry.
    """
    x_col = _pick_column(detector_df, ["cx", "module_x", "x"])
    y_col = _pick_column(detector_df, ["cy", "module_y", "y"])
    z_col = _pick_column(detector_df, ["cz", "module_z", "z"])

    required = ["volume_id", "layer_id", "module_id", x_col, y_col, z_col]
    if any(col is None for col in [x_col, y_col, z_col]):
        raise ValueError(
            "Detector CSV must contain module center columns (cx, cy, cz) or equivalents."
        )

    table = detector_df[required].copy()
    table = table.rename(columns={x_col: "cx", y_col: "cy", z_col: "cz"})
    table = table.reset_index(drop=True)
    table["module_index"] = table.index.astype(np.int64)
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
    Approximate intersections by snapping sampled trajectory points to nearest
    detector module centers within `tolerance_mm`.
    """
    if len(track_xyz) == 0:
        return pd.DataFrame()

    n_track = len(track_xyz)
    sample_points = int(max(10, min(sample_points, n_track)))
    sample_idx = np.linspace(0, n_track - 1, sample_points).astype(np.int64)
    sampled_xyz = track_xyz[sample_idx]

    centers = detector_modules[["cx", "cy", "cz"]].to_numpy(dtype=np.float32)
    delta = sampled_xyz[:, None, :] - centers[None, :, :]
    d2 = np.sum(delta * delta, axis=2)

    nearest_mod = np.argmin(d2, axis=1)
    nearest_d = np.sqrt(d2[np.arange(sampled_xyz.shape[0]), nearest_mod])
    keep = nearest_d <= float(tolerance_mm)

    if not np.any(keep):
        return pd.DataFrame()

    selected = []
    last_module = None
    for idx, mod_i, dist in zip(sample_idx[keep], nearest_mod[keep], nearest_d[keep]):
        if last_module == int(mod_i):
            continue
        selected.append((int(idx), int(mod_i), float(dist)))
        last_module = int(mod_i)
        if len(selected) >= int(max_hits):
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
