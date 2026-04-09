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
from .quirk_trajectory import simulate_quirk_track
from .quirk_intersections import (
    build_detector_module_table,
    intersect_track_with_modules,
    make_synthetic_detector_modules,
)


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


def build_quirk_event(
    event_id,
    detector,
    feature_scale,
    quirk_n_steps=6000,
    quirk_t_max=8.0,
    quirk_b_field=2.0,
    quirk_charge=1.0,
    quirk_pt=5.0,
    quirk_pz=1.0,
    quirk_phi0=0.0,
    quirk_x0=0.0,
    quirk_y0=0.0,
    quirk_z0=0.0,
    quirk_radial_amplitude=0.0,
    quirk_radial_frequency=3.0,
    quirk_radius_mm=900.0,
    quirk_tolerance_mm=30.0,
    quirk_max_hits=64,
    quirk_sample_points=600,
):
    track = simulate_quirk_track(
        event_id=event_id,
        n_steps=quirk_n_steps,
        t_max=quirk_t_max,
        b_field=quirk_b_field,
        charge=quirk_charge,
        pt=quirk_pt,
        pz=quirk_pz,
        phi0=quirk_phi0,
        x0=quirk_x0,
        y0=quirk_y0,
        z0=quirk_z0,
        quirky_amplitude=quirk_radial_amplitude,
        quirky_frequency=quirk_radial_frequency,
        radius_mm=quirk_radius_mm,
    )

    module_table = (
        build_detector_module_table(detector)
        if detector is not None
        else make_synthetic_detector_modules()
    )
    hits = intersect_track_with_modules(
        track["xyz"],
        module_table,
        tolerance_mm=quirk_tolerance_mm,
        max_hits=quirk_max_hits,
        sample_points=quirk_sample_points,
    )

    if hits.empty or len(hits) < 2:
        raise ValueError(
            f"No usable module intersections for quirk event {event_id}. "
            "Try larger quirk_tolerance_mm or different trajectory params."
        )

    pid = np.full(len(hits), event_id + 1, dtype=np.int64)
    pt = np.full(len(hits), quirk_pt, dtype=np.float32)
    hid = hits["hit_id"].to_numpy(dtype=np.int64)
    modules = hits["module_index"].to_numpy(dtype=np.int64)
    hit_weights = np.ones(len(hits), dtype=np.float32)

    # Sequential edges along trajectory order.
    edge_src = np.arange(len(hits) - 1, dtype=np.int64)
    edge_dst = np.arange(1, len(hits), dtype=np.int64)
    modulewise_true_edges = np.vstack([edge_src, edge_dst]).astype(np.int64)
    layerwise_true_edges = modulewise_true_edges.copy()

    edge_weights = np.ones(modulewise_true_edges.shape[1], dtype=np.float32)

    X = hits[["r", "phi", "z"]].to_numpy(dtype=np.float32) / np.array(
        feature_scale, dtype=np.float32
    )

    return (
        X,
        pid,
        modules,
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
                modulewise_true_edges,
                layerwise_true_edges,
                hid,
                pt,
                hit_weights,
                edge_weights,
            ) = build_quirk_event(
                evtid,
                detector=detector_orig,
                feature_scale=feature_scale,
                **{
                    k: kwargs[k]
                    for k in [
                        "quirk_n_steps",
                        "quirk_t_max",
                        "quirk_b_field",
                        "quirk_charge",
                        "quirk_pt",
                        "quirk_pz",
                        "quirk_phi0",
                        "quirk_x0",
                        "quirk_y0",
                        "quirk_z0",
                        "quirk_radial_amplitude",
                        "quirk_radial_frequency",
                        "quirk_radius_mm",
                        "quirk_tolerance_mm",
                        "quirk_max_hits",
                        "quirk_sample_points",
                    ]
                    if k in kwargs
                },
            )

            data = Data(
                x=torch.from_numpy(X).float(),
                pid=torch.from_numpy(pid),
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
