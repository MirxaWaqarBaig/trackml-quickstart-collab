"""Utilities for processing the overall event.

The module contains useful functions for handling data at the event level. More fine-grained utilities are 
reserved for `detector_utils` and `cell_utils`.
    
Todo:
    * Pull module IDs out into a csv file for readability """

# System
import os
import argparse
import logging
import multiprocessing as mp
from functools import partial

# Externals
import yaml
import numpy as np
import pandas as pd
try:
    import trackml.dataset as trackml_dataset
except ImportError:
    trackml_dataset = None

import torch
from torch_geometric.data import Data

from itertools import permutations
import itertools

# Locals
from .quirk_trajectory import simulate_quirk_pair_tracks
from .quirk_intersections import (
    build_detector_module_table,
    estimate_module_centers_from_hits,
    intersect_track_with_modules,
)

_MODULE_TABLE_CACHE = {}


def get_cell_information(
    data, cell_features, detector_orig, detector_proc, endcaps, noise
):
    # Import lazily so quirk-only generation does not require trackml package.
    from .cell_utils import get_one_event

    event_file = data.event_file
    angles = get_one_event(event_file, detector_orig, detector_proc)
    logging.info("Angles: {}".format(angles))
    hid = pd.DataFrame(data.hid.numpy(), columns=["hit_id"])
    cell_data = torch.from_numpy(
        (hid.merge(angles, on="hit_id")[cell_features]).to_numpy()
    ).float()
    logging.info("DF merged")
    data.cell_data = cell_data

    return data


def get_layerwise_edges(hits):

    hits = hits.assign(
        R=np.sqrt(
            (hits.x - hits.vx) ** 2 + (hits.y - hits.vy) ** 2 + (hits.z - hits.vz) ** 2
        )
    )
    hits = hits.sort_values("R").reset_index(drop=True).reset_index(drop=False)
    hits.loc[hits["particle_id"] == 0, "particle_id"] = np.nan
    hit_list = (
        hits.groupby(["particle_id", "layer"], sort=False)["index"]
        .agg(lambda x: list(x))
        .groupby(level=0)
        .agg(lambda x: list(x))
    )

    true_edges = []
    for row in hit_list.values:
        for i, j in zip(row[0:-1], row[1:]):
            true_edges.extend(list(itertools.product(i, j)))
    true_edges = np.array(true_edges).T

    return true_edges, hits


def get_modulewise_edges(hits):

    signal = hits[
        ((~hits.particle_id.isna()) & (hits.particle_id != 0)) & (~hits.vx.isna())
    ]
    signal = signal.drop_duplicates(
        subset=["particle_id", "volume_id", "layer_id", "module_id"]
    )

    # Sort by increasing distance from production
    signal = signal.assign(
        R=np.sqrt(
            (signal.x - signal.vx) ** 2
            + (signal.y - signal.vy) ** 2
            + (signal.z - signal.vz) ** 2
        )
    )
    signal = signal.sort_values("R").reset_index(drop=False)

    # Handle re-indexing
    signal = signal.rename(columns={"index": "unsorted_index"}).reset_index(drop=False)
    signal.loc[signal["particle_id"] == 0, "particle_id"] = np.nan

    # Group by particle ID
    signal_list = signal.groupby(["particle_id"], sort=False)["index"].agg(
        lambda x: list(x)
    )

    true_edges = []
    for row in signal_list.values:
        for i, j in zip(row[:-1], row[1:]):
            true_edges.append([i, j])

    true_edges = np.array(true_edges).T

    true_edges = signal.unsorted_index.values[true_edges]

    return true_edges


def select_hits(hits, truth, particles, endcaps=False, noise=False, min_pt=None):
    # Barrel volume and layer ids
    if endcaps:
        vlids = [
            (7, 2),
            (7, 4),
            (7, 6),
            (7, 8),
            (7, 10),
            (7, 12),
            (7, 14),
            (8, 2),
            (8, 4),
            (8, 6),
            (8, 8),
            (9, 2),
            (9, 4),
            (9, 6),
            (9, 8),
            (9, 10),
            (9, 12),
            (9, 14),
            (12, 2),
            (12, 4),
            (12, 6),
            (12, 8),
            (12, 10),
            (12, 12),
            (13, 2),
            (13, 4),
            (13, 6),
            (13, 8),
            (14, 2),
            (14, 4),
            (14, 6),
            (14, 8),
            (14, 10),
            (14, 12),
            (16, 2),
            (16, 4),
            (16, 6),
            (16, 8),
            (16, 10),
            (16, 12),
            (17, 2),
            (17, 4),
            (18, 2),
            (18, 4),
            (18, 6),
            (18, 8),
            (18, 10),
            (18, 12),
        ]
    else:
        vlids = [
            (8, 2),
            (8, 4),
            (8, 6),
            (8, 8),
            (13, 2),
            (13, 4),
            (13, 6),
            (13, 8),
            (17, 2),
            (17, 4),
        ]
    n_det_layers = len(vlids)
    # Select barrel layers and assign convenient layer number [0-9]
    vlid_groups = hits.groupby(["volume_id", "layer_id"])
    hits = pd.concat(
        [vlid_groups.get_group(vlids[i]).assign(layer=i) for i in range(n_det_layers)]
    )

    if noise:
        truth = truth.merge(
            particles[["particle_id", "vx", "vy", "vz"]], on="particle_id", how="left"
        )
    else:
        truth = truth.merge(
            particles[["particle_id", "vx", "vy", "vz"]], on="particle_id", how="inner"
        )

    truth = truth.assign(pt=np.sqrt(truth.tpx**2 + truth.tpy**2))

    if min_pt:
        truth = truth[truth.pt > min_pt]

    # Calculate derived hits variables
    r = np.sqrt(hits.x**2 + hits.y**2)
    phi = np.arctan2(hits.y, hits.x)
    # Select the data columns we need
    hits = hits.assign(r=r, phi=phi).merge(truth, on="hit_id")

    return hits


def build_event(
    event_file,
    feature_scale,
    endcaps=False,
    modulewise=True,
    layerwise=True,
    noise=False,
    min_pt=None,
    detector=None,
):
    if trackml_dataset is None:
        raise ImportError(
            "trackml.dataset is required for TrackML feature-store mode. "
            "Install the trackml package or run dataset_mode=quirk."
        )
    # Get true edge list using the ordering by R' = distance from production vertex of each particle
    hits, particles, truth = trackml_dataset.load_event(
        event_file, parts=["hits", "particles", "truth"]
    )
    hits = select_hits(hits, truth, particles, endcaps=endcaps, noise=noise, min_pt=min_pt).assign(
        evtid=int(event_file[-9:])
    )
    
    # Make a unique module ID and attach to hits
    if detector is not None:
        module_lookup = detector.reset_index()[["index", "volume_id", "layer_id", "module_id"]].rename(columns={"index": "module_index"})
        hits = hits.merge(module_lookup, on=["volume_id", "layer_id", "module_id"], how="left")
        module_id = hits.module_index.to_numpy()
    else:
        module_id = None

    layer_id = hits.layer.to_numpy()

    # Handle which truth graph(s) are being produced
    modulewise_true_edges, layerwise_true_edges = None, None

    if layerwise:
        layerwise_true_edges, hits = get_layerwise_edges(hits)
        logging.info(
            "Layerwise truth graph built for {} with size {}".format(
                event_file, layerwise_true_edges.shape
            )
        )

    if modulewise:
        modulewise_true_edges = get_modulewise_edges(hits)
        logging.info(
            "Modulewise truth graph built for {} with size {}".format(
                event_file, modulewise_true_edges.shape
            )
        )

    edge_weights = (
        hits.weight.to_numpy()[modulewise_true_edges]
        if modulewise
        else hits.weight.to_numpy()[layerwise_true_edges]
    )
    edge_weight_average = (edge_weights[0] + edge_weights[1]) / 2
    edge_weight_norm = edge_weight_average / edge_weight_average.mean()

    logging.info("Weights constructed")

    return (
        hits[["r", "phi", "z"]].to_numpy() / feature_scale,
        hits.particle_id.to_numpy(),
        module_id,
        modulewise_true_edges,
        layerwise_true_edges,
        hits["hit_id"].to_numpy(),
        hits.pt.to_numpy(),
        hits.weight.to_numpy(),
        edge_weight_norm,
    )


def prepare_event(
    event_file,
    detector_orig,
    detector_proc,
    cell_features,
    output_dir=None,
    endcaps=False,
    modulewise=True,
    layerwise=True,
    noise=False,
    min_pt=None,
    cell_information=True,
    overwrite=False,
    **kwargs
):

    try:
        evtid = int(event_file[-9:])
        filename = os.path.join(output_dir, str(evtid))

        if not os.path.exists(filename) or overwrite:
            logging.info("Preparing event {}".format(evtid))
            feature_scale = [1000, np.pi, 1000]

            (
                X,
                pid,
                module_id,
                modulewise_true_edges,
                layerwise_true_edges,
                hid,
                pt,
                hit_weights,
                edge_weights
            ) = build_event(
                event_file,
                feature_scale,
                endcaps=endcaps,
                modulewise=modulewise,
                layerwise=layerwise,
                noise=noise,
                min_pt=min_pt,
                detector=detector_orig
            )

            data = Data(
                x=torch.from_numpy(X).float(),
                pid=torch.from_numpy(pid),
                modules=torch.from_numpy(module_id),
                event_file=event_file,
                hid=torch.from_numpy(hid),
                pt=torch.from_numpy(pt),
                hit_weights=torch.from_numpy(hit_weights),
                edge_weights=torch.from_numpy(edge_weights),
            )
            if modulewise_true_edges is not None:
                data.modulewise_true_edges = torch.from_numpy(modulewise_true_edges)
            if layerwise_true_edges is not None:
                data.layerwise_true_edges = torch.from_numpy(layerwise_true_edges)
            logging.info("Getting cell info")

            if cell_information:
                data = get_cell_information(
                    data, cell_features, detector_orig, detector_proc, endcaps, noise
                )

            with open(filename, "wb") as pickle_file:
                torch.save(data, pickle_file)

        else:
            logging.info("{} already exists".format(evtid))
    except Exception as inst:
        print("File:", event_file, "had exception", inst)


def _parse_event_id(event_ref):
    if isinstance(event_ref, (int, np.integer)):
        return int(event_ref)
    text = str(event_ref)
    digits = "".join([ch for ch in text if ch.isdigit()])
    if len(digits) >= 1:
        return int(digits[-9:])
    return int(abs(hash(text)) % 1_000_000_000)


def _trackml_event_prefixes(input_dir):
    base = str(input_dir)
    if not base:
        return []
    candidates = sorted(
        {
            path[: -len("-hits.csv")]
            for path in [
                os.path.join(base, filename)
                for filename in os.listdir(base)
                if filename.endswith("-hits.csv")
            ]
        }
    )
    return candidates


def _load_trackml_background_hits(
    event_prefix,
    module_table,
    sm_pt_min=0.1,
    sm_max_hits=4000,
):
    hits = pd.read_csv(event_prefix + "-hits.csv")
    truth = pd.read_csv(event_prefix + "-truth.csv")
    particles = pd.read_csv(event_prefix + "-particles.csv")

    truth = truth.assign(pt=np.sqrt(truth.tpx**2 + truth.tpy**2))
    truth = truth[truth.pt >= float(sm_pt_min)]
    truth = truth[truth.particle_id != 0]
    if len(truth) == 0:
        return pd.DataFrame()

    vertex = particles[["particle_id", "vx", "vy", "vz"]]
    truth = truth.merge(vertex, on="particle_id", how="left")

    merged = hits.merge(
        truth[["hit_id", "particle_id", "pt", "vx", "vy", "vz"]],
        on="hit_id",
        how="inner",
    )
    if len(merged) == 0:
        return pd.DataFrame()

    merged = merged.merge(
        module_table[["volume_id", "layer_id", "module_id", "module_index"]],
        on=["volume_id", "layer_id", "module_id"],
        how="inner",
    )
    if len(merged) == 0:
        return pd.DataFrame()

    merged = merged.assign(
        r=np.sqrt(merged.x**2 + merged.y**2),
        phi=np.arctan2(merged.y, merged.x),
    )
    merged = merged.assign(
        order_key=np.sqrt(
            (merged.x - merged.vx) ** 2 + (merged.y - merged.vy) ** 2 + (merged.z - merged.vz) ** 2
        )
    )
    merged = merged.sort_values(["particle_id", "order_key"]).reset_index(drop=True)

    if len(merged) > int(sm_max_hits):
        merged = merged.iloc[: int(sm_max_hits)].copy()

    return merged


def _build_edges_from_group_order(hits_df):
    edge_src = []
    edge_dst = []
    for _, grp in hits_df.groupby("particle_id", sort=False):
        idx = grp.index.to_numpy(dtype=np.int64)
        if len(idx) < 2:
            continue
        edge_src.extend(idx[:-1].tolist())
        edge_dst.extend(idx[1:].tolist())
    if len(edge_src) == 0:
        return np.empty((2, 0), dtype=np.int64)
    return np.vstack([np.array(edge_src, dtype=np.int64), np.array(edge_dst, dtype=np.int64)])


def _get_module_table(detector, input_dir):
    cache_key = (id(detector), str(input_dir))
    if cache_key in _MODULE_TABLE_CACHE:
        return _MODULE_TABLE_CACHE[cache_key]
    module_centers = estimate_module_centers_from_hits(input_dir)
    module_table = build_detector_module_table(detector, module_centers_df=module_centers)
    _MODULE_TABLE_CACHE[cache_key] = module_table
    return module_table


def _fallback_quirk_hits_from_modules(
    event_id,
    module_table,
    min_hits_per_track,
    pair_pt,
):
    """
    Build a minimal quirk/anti-quirk hit set directly from detector module centers.
    Used only when trajectory intersections fail after retries.
    """
    min_hits_per_track = int(max(2, min_hits_per_track))
    if len(module_table) < 2 * min_hits_per_track:
        return pd.DataFrame()

    centers = module_table.copy()
    centers = centers.assign(
        r=np.sqrt(centers.cx**2 + centers.cy**2),
        phi=np.arctan2(centers.cy, centers.cx),
    )
    centers = centers.sort_values(["r", "phi"]).reset_index(drop=True)
    stride = max(1, len(centers) // (4 * min_hits_per_track))
    idx_q = np.arange(0, stride * min_hits_per_track, stride, dtype=np.int64)
    idx_aq = np.arange(
        len(centers) // 2,
        len(centers) // 2 + stride * min_hits_per_track,
        stride,
        dtype=np.int64,
    )
    idx_q = np.clip(idx_q, 0, len(centers) - 1)
    idx_aq = np.clip(idx_aq, 0, len(centers) - 1)

    q = centers.iloc[idx_q].copy().reset_index(drop=True)
    aq = centers.iloc[idx_aq].copy().reset_index(drop=True)

    q = q.assign(
        x=q.cx.astype(np.float32),
        y=q.cy.astype(np.float32),
        z=q.cz.astype(np.float32),
        volume_id=q.volume_id.astype(np.int64),
        layer_id=q.layer_id.astype(np.int64),
        module_id=q.module_id.astype(np.int64),
        module_index=q.module_index.astype(np.int64),
        distance_to_module=0.0,
        particle_id=np.int64(event_id * 100000 + 90001),
        q_label="quirk",
        source_label=np.int64(1),
        pt=float(max(1e-4, 0.5 * pair_pt)),
        order_key=np.linspace(0.0, 1.0, len(q), dtype=np.float32),
    )
    aq = aq.assign(
        x=aq.cx.astype(np.float32),
        y=aq.cy.astype(np.float32),
        z=aq.cz.astype(np.float32),
        volume_id=aq.volume_id.astype(np.int64),
        layer_id=aq.layer_id.astype(np.int64),
        module_id=aq.module_id.astype(np.int64),
        module_index=aq.module_index.astype(np.int64),
        distance_to_module=0.0,
        particle_id=np.int64(event_id * 100000 + 90002),
        q_label="anti_quirk",
        source_label=np.int64(2),
        pt=float(max(1e-4, 0.5 * pair_pt)),
        order_key=np.linspace(0.0, 1.0, len(aq), dtype=np.float32),
    )
    return pd.concat([q, aq], ignore_index=True, sort=False)


def build_quirk_event(
    event_id,
    detector,
    input_dir,
    feature_scale,
    quirk_n_steps=6000,
    quirk_t_max=8.0,
    quirk_b_field=2.0,
    quirk_charge=1.0,
    quirk_mass=1000.0,
    quirk_pair_pt=5.0,
    quirk_pair_pz=1.0,
    quirk_opening_angle=np.pi / 2.0,
    quirk_phi0=0.0,
    quirk_x0=0.0,
    quirk_y0=0.0,
    quirk_z0=0.0,
    quirk_string_tension=0.02,
    quirk_oscillation_jitter=0.0,
    quirk_velocity_scale=1500.0,
    quirk_tolerance_mm=30.0,
    quirk_max_hits=96,
    quirk_sample_points=600,
    include_sm_background=True,
    sm_background_pt_min=0.5,
    sm_background_max_hits=3000,
    quirk_pairs_per_event=1,
    quirk_pair_pt_jitter_frac=0.15,
    quirk_pair_phi_spread=0.35,
    quirk_opening_angle_jitter=0.25,
    quirk_origin_jitter_mm=30.0,
    min_quirk_hits_per_track=8,
    min_total_quirk_hits=20,
    quirk_event_max_retries=5,
):
    if detector is None:
        raise ValueError(
            "dataset_mode=quirk now requires real TrackML detector geometry. "
            "Provide detector_path and input_dir with TrackML hit CSV files."
        )

    module_table = _get_module_table(detector, input_dir)
    quirk_pairs_per_event = int(max(1, quirk_pairs_per_event))
    quirk_event_max_retries = int(max(1, quirk_event_max_retries))
    quirk_pid_base = np.int64(event_id) * np.int64(100000)
    hits = pd.DataFrame()
    last_quirk_hits = 0

    for retry in range(quirk_event_max_retries):
        rng = np.random.default_rng(int(event_id) * 1000 + retry)
        quirk_frames = []
        tol_retry = float(quirk_tolerance_mm) * (1.0 + 0.2 * retry)

        for pair_idx in range(quirk_pairs_per_event):
            phi_delta = float(rng.uniform(-quirk_pair_phi_spread, quirk_pair_phi_spread))
            opening_delta = float(
                rng.uniform(-quirk_opening_angle_jitter, quirk_opening_angle_jitter)
            )
            pt_jitter = 1.0 + float(
                rng.uniform(-quirk_pair_pt_jitter_frac, quirk_pair_pt_jitter_frac)
            )
            pair_pt_i = float(max(1e-4, quirk_pair_pt * pt_jitter))
            x0_i = float(quirk_x0 + rng.uniform(-quirk_origin_jitter_mm, quirk_origin_jitter_mm))
            y0_i = float(quirk_y0 + rng.uniform(-quirk_origin_jitter_mm, quirk_origin_jitter_mm))
            z0_i = float(
                quirk_z0 + rng.uniform(-0.5 * quirk_origin_jitter_mm, 0.5 * quirk_origin_jitter_mm)
            )
            opening_i = float(
                np.clip(
                    quirk_opening_angle + opening_delta,
                    1e-3,
                    np.pi - 1e-3,
                )
            )

            pair = simulate_quirk_pair_tracks(
                event_id=event_id * 100 + pair_idx + 10000 * retry,
                n_steps=quirk_n_steps,
                t_max=quirk_t_max,
                b_field=quirk_b_field,
                charge=quirk_charge,
                quirk_mass=quirk_mass,
                pair_pt=pair_pt_i,
                pair_pz=quirk_pair_pz,
                opening_angle=opening_i,
                phi0=quirk_phi0 + phi_delta,
                x0=x0_i,
                y0=y0_i,
                z0=z0_i,
                string_tension=quirk_string_tension,
                oscillation_jitter=quirk_oscillation_jitter,
                velocity_scale=quirk_velocity_scale,
            )

            hits_q = intersect_track_with_modules(
                pair["xyz_q"],
                module_table,
                tolerance_mm=tol_retry,
                max_hits=quirk_max_hits,
                sample_points=quirk_sample_points,
            )
            hits_aq = intersect_track_with_modules(
                pair["xyz_aq"],
                module_table,
                tolerance_mm=tol_retry,
                max_hits=quirk_max_hits,
                sample_points=quirk_sample_points,
            )

            # Enforce per-track minimum for signal quality in mixed events.
            if len(hits_q) < int(min_quirk_hits_per_track) or len(hits_aq) < int(
                min_quirk_hits_per_track
            ):
                continue

            pid_q = quirk_pid_base + np.int64(2 * pair_idx + 1)
            pid_aq = quirk_pid_base + np.int64(2 * pair_idx + 2)

            hits_q["particle_id"] = pid_q
            hits_q["q_label"] = "quirk"
            hits_q["source_label"] = np.int64(1)
            hits_q["pt"] = float(max(1e-4, 0.5 * pair_pt_i))
            # Use trajectory_step (simulator time index) for correct time-ordered edges.
            # Quirks oscillate back toward origin so distance-from-vertex ordering is wrong.
            if "trajectory_step" in hits_q.columns:
                hits_q["order_key"] = hits_q["trajectory_step"].astype(np.float32)
            else:
                hits_q["order_key"] = np.arange(len(hits_q), dtype=np.float32)

            hits_aq["particle_id"] = pid_aq
            hits_aq["q_label"] = "anti_quirk"
            hits_aq["source_label"] = np.int64(2)
            hits_aq["pt"] = float(max(1e-4, 0.5 * pair_pt_i))
            if "trajectory_step" in hits_aq.columns:
                hits_aq["order_key"] = hits_aq["trajectory_step"].astype(np.float32)
            else:
                hits_aq["order_key"] = np.arange(len(hits_aq), dtype=np.float32)
            quirk_frames.extend([hits_q, hits_aq])

        if quirk_frames:
            hits = pd.concat(quirk_frames, ignore_index=True)
            last_quirk_hits = int((hits["source_label"] > 0).sum())
            if last_quirk_hits >= int(min_total_quirk_hits):
                break
        else:
            last_quirk_hits = 0

    # If retry loop still cannot build enough quirk intersections, add a deterministic
    # fallback from module centers to avoid dropping entire events.
    if (hits.empty or int((hits["source_label"] > 0).sum()) < int(min_total_quirk_hits)):
        fallback_hits = _fallback_quirk_hits_from_modules(
            event_id=event_id,
            module_table=module_table,
            min_hits_per_track=min_quirk_hits_per_track,
            pair_pt=quirk_pair_pt,
        )
        if len(fallback_hits) > 0:
            hits = pd.concat([hits, fallback_hits], ignore_index=True, sort=False)

    if include_sm_background:
        prefixes = _trackml_event_prefixes(input_dir)
        if prefixes:
            bg_prefix = prefixes[event_id % len(prefixes)]
            bg_hits = _load_trackml_background_hits(
                bg_prefix,
                module_table,
                sm_pt_min=sm_background_pt_min,
                sm_max_hits=sm_background_max_hits,
            )
            if len(bg_hits) > 0:
                bg_hits = bg_hits.assign(q_label="sm")
                bg_hits = bg_hits.assign(source_label=np.int64(0))
                hits = pd.concat([hits, bg_hits], ignore_index=True, sort=False)

    if hits.empty or len(hits) < 4:
        raise ValueError(
            f"No usable module intersections for quirk event {event_id}. "
            "Try larger quirk_tolerance_mm or different trajectory params."
        )

    total_quirk_hits = int((hits["source_label"] > 0).sum()) if "source_label" in hits else 0
    if total_quirk_hits < int(min_total_quirk_hits):
        raise ValueError(
            f"Insufficient quirk signal for event {event_id}: only {total_quirk_hits} "
            f"hits (required >= {int(min_total_quirk_hits)}). "
            "Increase quirk_tolerance_mm, quirk_pairs_per_event, or lower min thresholds."
        )

    # Re-index and build sequential true edges per particle.
    hits = hits.sort_values(["particle_id", "order_key"]).reset_index(drop=True)
    hits["hit_id"] = np.arange(len(hits), dtype=np.int64)
    modulewise_true_edges = _build_edges_from_group_order(hits)
    layerwise_true_edges = modulewise_true_edges.copy()
    if modulewise_true_edges.shape[1] == 0:
        raise ValueError(
            f"No valid truth edges formed for quirk event {event_id}. "
            "Check quirk/SM generation settings."
        )

    pid = hits["particle_id"].to_numpy(dtype=np.int64)
    source_label = hits["source_label"].to_numpy(dtype=np.int64)
    pt = hits["pt"].to_numpy(dtype=np.float32)
    hid = hits["hit_id"].to_numpy(dtype=np.int64)
    modules = hits["module_index"].to_numpy(dtype=np.int64)
    hit_weights = np.ones(len(hits), dtype=np.float32)
    edge_weights = np.ones(modulewise_true_edges.shape[1], dtype=np.float32)

    X = hits[["r", "phi", "z"]].to_numpy(dtype=np.float32) / np.array(
        feature_scale, dtype=np.float32
    )

    return (
        X,
        pid,
        modules,
        source_label,
        modulewise_true_edges,
        layerwise_true_edges,
        hid,
        pt,
        hit_weights,
        edge_weights,
    )


def prepare_quirk_event(
    event_ref,
    detector_orig,
    detector_proc,
    cell_features,
    output_dir=None,
    overwrite=False,
    cell_information=False,
    **kwargs,
):
    try:
        evtid = _parse_event_id(event_ref)
        filename = os.path.join(output_dir, str(evtid))

        if not os.path.exists(filename) or overwrite:
            logging.info("Preparing quirk event %s", evtid)
            feature_scale = [1000, np.pi, 1000]

            (
                X,
                pid,
                module_id,
                source_label,
                modulewise_true_edges,
                layerwise_true_edges,
                hid,
                pt,
                hit_weights,
                edge_weights,
            ) = build_quirk_event(
                evtid,
                detector=detector_orig,
                input_dir=kwargs.get("input_dir", ""),
                feature_scale=feature_scale,
                **{
                    k: kwargs[k]
                    for k in [
                        "quirk_n_steps",
                        "quirk_t_max",
                        "quirk_b_field",
                        "quirk_charge",
                        "quirk_mass",
                        "quirk_pair_pt",
                        "quirk_pair_pz",
                        "quirk_opening_angle",
                        "quirk_phi0",
                        "quirk_x0",
                        "quirk_y0",
                        "quirk_z0",
                        "quirk_string_tension",
                        "quirk_oscillation_jitter",
                        "quirk_velocity_scale",
                        "quirk_tolerance_mm",
                        "quirk_max_hits",
                        "quirk_sample_points",
                        "include_sm_background",
                        "sm_background_pt_min",
                        "sm_background_max_hits",
                        "quirk_pairs_per_event",
                        "quirk_pair_pt_jitter_frac",
                        "quirk_pair_phi_spread",
                        "quirk_opening_angle_jitter",
                        "quirk_origin_jitter_mm",
                        "min_quirk_hits_per_track",
                        "min_total_quirk_hits",
                        "quirk_event_max_retries",
                    ]
                    if k in kwargs
                },
            )

            data = Data(
                x=torch.from_numpy(X).float(),
                pid=torch.from_numpy(pid),
                source_label=torch.from_numpy(source_label),
                modules=torch.from_numpy(module_id),
                event_file=f"quirk_event_{evtid:09d}",
                hid=torch.from_numpy(hid),
                pt=torch.from_numpy(pt),
                hit_weights=torch.from_numpy(hit_weights),
                edge_weights=torch.from_numpy(edge_weights),
            )
            if modulewise_true_edges is not None:
                data.modulewise_true_edges = torch.from_numpy(modulewise_true_edges)
            if layerwise_true_edges is not None:
                data.layerwise_true_edges = torch.from_numpy(layerwise_true_edges)

            if cell_information:
                logging.warning(
                    "cell_information=True requested for quirk event %s, "
                    "but no raw cell file exists. Skipping cell feature build.",
                    evtid,
                )

            with open(filename, "wb") as pickle_file:
                torch.save(data, pickle_file)
        else:
            logging.info("Quirk event %s already exists", evtid)
    except Exception as inst:
        print("File:", event_ref, "had exception", inst)
