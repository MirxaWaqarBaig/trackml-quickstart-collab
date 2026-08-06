"""Reproducible coplanarity study at mQ=500 GeV and Lambda=20 eV.

The simulator units and baseline kinematics follow the prepared YAML on the
``quirk-module-intersection-cylinder`` branch.  Track intersections, barrel
layers, and module assignments are derived from the supplied detectors.csv.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Pipelines.TrackML_Example.LightningModules.Processing.utils.coplanarity import (
    PAPER_DPHI_SEED,
    PAPER_DS_FINAL_MM,
    PAPER_DS_ITER_MM,
    PAPER_DS_SEED_MM,
    PAPER_DW_FINAL_MM,
    PAPER_DW_ITER_MM,
    PAPER_DW_SEED_MM,
    compute_coplanarity_score,
    compute_T_tensor,
    evaluate_classifier,
    paper_plane_finding,
    roc_curve_points,
)
from Pipelines.TrackML_Example.LightningModules.Processing.utils.quirk_intersections import (
    build_barrel_layer_catalog,
    build_detector_module_table,
    intersect_track_with_cylinders_and_modules,
)
from Pipelines.TrackML_Example.LightningModules.Processing.utils.quirk_trajectory import (
    simulate_quirk_pair_tracks,
)


PHYSICS = {
    "mass_gev": 500.0,
    "lambda_ev": 20.0,
    "n_steps": 70000,
    "t_max_ns": 80.0,
    "b_field_t": 2.0,
    "charge_e": 1.0,
    "pair_pt_gev": 50.0,
    "pair_pz_gev": 10.0,
    "opening_angle_rad": 1.3,
    "pair_pt_jitter_frac": 0.05,
    "opening_angle_jitter_rad": 0.10,
    "primary_vertex_mm": [0.0, 0.0, 0.0],
}

PAPER_CUTS = {
    "ds_seed": PAPER_DS_SEED_MM,
    "dw_seed": PAPER_DW_SEED_MM,
    "ds_iter": PAPER_DS_ITER_MM,
    "dw_iter": PAPER_DW_ITER_MM,
    "ds_final": PAPER_DS_FINAL_MM,
    "dw_final": PAPER_DW_FINAL_MM,
    "dphi_seed": PAPER_DPHI_SEED,
    "dz_seed_mm": 20.0,
    "grow_factor": 3.0,
    "min_hits_per_layer": 2,
}

_WORKER_MODULES = None
_WORKER_LAYERS = None


def _init_worker(detector_path):
    global _WORKER_MODULES, _WORKER_LAYERS
    detector = pd.read_csv(detector_path)
    _WORKER_MODULES = build_detector_module_table(detector)
    _WORKER_LAYERS = build_barrel_layer_catalog(_WORKER_MODULES)


def _simulate_signal(event_id):
    rng = np.random.default_rng(20260806 + int(event_id))
    pt = PHYSICS["pair_pt_gev"] * (
        1.0 + rng.uniform(-PHYSICS["pair_pt_jitter_frac"],
                          PHYSICS["pair_pt_jitter_frac"])
    )
    opening = PHYSICS["opening_angle_rad"] + rng.uniform(
        -PHYSICS["opening_angle_jitter_rad"],
        PHYSICS["opening_angle_jitter_rad"],
    )
    phi0 = rng.uniform(-np.pi, np.pi)
    tracks = simulate_quirk_pair_tracks(
        event_id=event_id,
        n_steps=PHYSICS["n_steps"],
        t_max=PHYSICS["t_max_ns"],
        b_field=PHYSICS["b_field_t"],
        charge=PHYSICS["charge_e"],
        quirk_mass=PHYSICS["mass_gev"],
        pair_pt=pt,
        pair_pz=PHYSICS["pair_pz_gev"],
        opening_angle=opening,
        phi0=phi0,
        x0=0.0, y0=0.0, z0=0.0,
        string_tension=PHYSICS["lambda_ev"],
        oscillation_jitter=0.0,
        velocity_scale=1.0,
    )
    frames = []
    max_radii = []
    for source, key in ((1, "xyz_q"), (2, "xyz_aq")):
        trajectory = np.asarray(tracks[key], dtype=np.float64)
        max_radii.append(float(np.max(np.hypot(trajectory[:, 0], trajectory[:, 1]))))
        hits = intersect_track_with_cylinders_and_modules(
            trajectory, _WORKER_MODULES, barrel_layers=_WORKER_LAYERS,
            max_hits=200,
        )
        if len(hits):
            hits["source_label"] = source
            frames.append(hits)
    hits = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return {
        "event_id": int(event_id),
        "hits": hits.to_dict("list"),
        "max_r_q_mm": max_radii[0],
        "max_r_aq_mm": max_radii[1],
        "pair_pt_gev": float(pt),
        "opening_angle_rad": float(opening),
    }


def _barrel_background(path, layer_lookup, max_hits, seed):
    hits = pd.read_csv(path, usecols=["x", "y", "z", "volume_id", "layer_id"])
    keys = list(zip(hits.volume_id.astype(int), hits.layer_id.astype(int)))
    analysis_layer = np.array([layer_lookup.get(key, 0) for key in keys], dtype=np.int32)
    hits = hits.loc[analysis_layer > 0].copy()
    hits["analysis_layer"] = analysis_layer[analysis_layer > 0]
    if len(hits) > max_hits:
        hits = hits.sample(n=max_hits, random_state=int(seed), replace=False)
    hits["source_label"] = 0
    return hits.reset_index(drop=True)


def _score_event(hits, n_layers):
    xyz = hits[["x", "y", "z"]].to_numpy(dtype=np.float64)
    layers = hits["analysis_layer"].to_numpy(dtype=np.int32)
    simple_score = compute_coplanarity_score(xyz)[0]
    result = paper_plane_finding(
        xyz, layer_ids=layers,
        min_layers_with_2hits=max(1, n_layers - 1),
        **PAPER_CUTS,
    )
    paper_score = float(result["delta_s"]) if np.isfinite(result["delta_s"]) else 1.0e6
    return simple_score, paper_score, result


def _auc(signal, background):
    _, _, _, value = roc_curve_points(np.asarray(signal), np.asarray(background), 500)
    return value


def _plot_results(records, output_dir, n_layers):
    signal = [r for r in records if r["class"] == "signal"]
    background = [r for r in records if r["class"] == "background"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist([r["simple_score"] for r in signal], 30, alpha=.65, label="signal", density=True)
    ax.hist([r["simple_score"] for r in background], 30, alpha=.65, label="background", density=True)
    ax.set(xlabel="Centered-SVD coplanarity score", ylabel="Density",
           title="Coplanarity score distributions")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(output_dir / "coplanarity_score_distributions.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, field, label in zip(axes, ("delta_s_mm", "delta_w_mm"),
                                ("Delta s [mm]", "Delta w [mm]")):
        for subset, name in ((signal, "signal"), (background, "background")):
            vals = [r[field] for r in subset if np.isfinite(r[field])]
            if vals:
                ax.hist(vals, 25, alpha=.65, label=name)
        truth_field = "truth_quirk_ds_mm" if field == "delta_s_mm" else "truth_quirk_dw_mm"
        truth_vals = [r[truth_field] for r in signal if np.isfinite(r[truth_field])]
        if truth_vals:
            ax.hist(truth_vals, 25, histtype="step", linewidth=2,
                    label="quirk-only hits (truth diagnostic)")
        ax.set_xlabel(label); ax.set_ylabel("Events"); ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle("Plane-width diagnostic distributions")
    fig.tight_layout(); fig.savefig(output_dir / "delta_s_delta_w_distributions.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for field, label in (("simple_score", "Centered SVD"), ("paper_score", "Paper candidate")):
        sig = np.array([r[field] for r in signal], dtype=float)
        bg = np.array([r[field] for r in background], dtype=float)
        _, tpr, fpr, auc = roc_curve_points(sig, bg, 500)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=.5)
    ax.set(xlabel="False-positive rate", ylabel="Signal efficiency", title="ROC curves")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(output_dir / "roc_curves.png", dpi=160)
    plt.close(fig)

    sig_counts = np.array([r["signal_layer_counts"] for r in signal], dtype=float)
    bg_counts = np.array([r["background_layer_counts"] for r in signal], dtype=float)
    x = np.arange(1, n_layers + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - .2, sig_counts.mean(axis=0), .4, label="quirk hits in mixed signal")
    ax.bar(x + .2, bg_counts.mean(axis=0), .4, label="SM hits in mixed signal")
    ax.set_xticks(x); ax.set_xlabel("Geometry-derived barrel layer (inside-out)")
    ax.set_ylabel("Mean hits/event"); ax.set_title("Detector layer hit distributions")
    ax.legend(); ax.grid(axis="y", alpha=.25); fig.tight_layout()
    fig.savefig(output_dir / "detector_layer_hit_distributions.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-path", default=str(REPO_ROOT / "trackml_raw/detectors.csv"))
    parser.add_argument("--raw-data-dir", default=str(REPO_ROOT / "trackml_raw/train_all"))
    parser.add_argument("--output-dir", default=str(
        REPO_ROOT / "Examples/TrackML_Quickstart/datasets/coplanarity_500gev_lambda20"))
    parser.add_argument("--n-events", type=int, default=30)
    parser.add_argument("--sm-max-hits", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = pd.read_csv(args.detector_path)
    modules = build_detector_module_table(detector)
    catalog = build_barrel_layer_catalog(modules)
    layer_lookup = {
        (int(r.volume_id), int(r.layer_id)): int(r.analysis_layer)
        for r in catalog.itertuples(index=False)
    }
    n_layers = len(catalog)
    hit_files = sorted(Path(args.raw_data_dir).glob("*-hits.csv"))
    if len(hit_files) < 2 * args.n_events:
        raise ValueError(f"Need {2 * args.n_events} SM hit files, found {len(hit_files)}")

    with ProcessPoolExecutor(
        max_workers=max(1, args.workers), initializer=_init_worker,
        initargs=(args.detector_path,),
    ) as pool:
        simulated = list(pool.map(_simulate_signal, range(args.n_events)))

    records = []
    for i, simulation in enumerate(simulated):
        quirk_hits = pd.DataFrame(simulation["hits"])
        quirk_xyz = quirk_hits[["x", "y", "z"]].to_numpy(dtype=np.float64)
        truth_ds, truth_dw, *_ = compute_T_tensor(quirk_xyz)
        outer_hits = quirk_hits[quirk_hits.analysis_layer == n_layers]
        outer_phi = np.arctan2(outer_hits.y.to_numpy(), outer_hits.x.to_numpy())
        outer_dphi = (float(abs(np.arctan2(np.sin(outer_phi[0] - outer_phi[1]),
                                           np.cos(outer_phi[0] - outer_phi[1]))))
                      if len(outer_phi) >= 2 else float("nan"))
        outer_seed_xyz = quirk_hits[
            quirk_hits.analysis_layer.isin([n_layers - 1, n_layers])
        ][["x", "y", "z"]].to_numpy(dtype=np.float64)
        seed_ds, seed_dw, *_ = compute_T_tensor(outer_seed_xyz)
        sm_hits = _barrel_background(hit_files[i], layer_lookup, args.sm_max_hits, 1000 + i)
        mixed = pd.concat([quirk_hits, sm_hits], ignore_index=True, sort=False)
        simple, paper_score, result = _score_event(mixed, n_layers)
        records.append({
            "class": "signal", "event_id": i,
            "simple_score": simple, "paper_score": paper_score,
            "paper_found": bool(result["found"]),
            "delta_s_mm": float(result["delta_s"]),
            "delta_w_mm": float(result["delta_w"]),
            "n_plane_hits": int(result["n_plane_hits"]),
            "n_layers_2hits": int(result["n_layers_2hits"]),
            "outer_seed_layer": int(result["outer_layer"]),
            "second_seed_layer": int(result["second_layer"]),
            "n_seeds_tried": int(result["n_seeds_tried"]),
            "max_r_q_mm": simulation["max_r_q_mm"],
            "max_r_aq_mm": simulation["max_r_aq_mm"],
            "n_quirk_hits": int(len(quirk_hits)), "n_sm_hits": int(len(sm_hits)),
            "truth_quirk_ds_mm": float(truth_ds),
            "truth_quirk_dw_mm": float(truth_dw),
            "truth_outer_seed_ds_mm": float(seed_ds),
            "truth_outer_seed_dw_mm": float(seed_dw),
            "truth_outer_pair_dphi_rad": outer_dphi,
            "signal_layer_counts": [int(((quirk_hits.analysis_layer == j)).sum()) for j in range(1, n_layers + 1)],
            "background_layer_counts": [int(((sm_hits.analysis_layer == j)).sum()) for j in range(1, n_layers + 1)],
        })

        bg_hits = _barrel_background(
            hit_files[args.n_events + i], layer_lookup, args.sm_max_hits, 2000 + i)
        simple, paper_score, result = _score_event(bg_hits, n_layers)
        records.append({
            "class": "background", "event_id": i,
            "simple_score": simple, "paper_score": paper_score,
            "paper_found": bool(result["found"]),
            "delta_s_mm": float(result["delta_s"]),
            "delta_w_mm": float(result["delta_w"]),
            "n_plane_hits": int(result["n_plane_hits"]),
            "n_layers_2hits": int(result["n_layers_2hits"]),
            "outer_seed_layer": int(result["outer_layer"]),
            "second_seed_layer": int(result["second_layer"]),
            "n_seeds_tried": int(result["n_seeds_tried"]),
        })

    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "event_results.csv", index=False)
    signal = frame[frame["class"] == "signal"]
    background = frame[frame["class"] == "background"]
    y_true = np.r_[np.ones(len(signal), dtype=int), np.zeros(len(background), dtype=int)]
    y_pred = np.r_[signal.paper_found.astype(int), background.paper_found.astype(int)]
    metrics = evaluate_classifier(y_pred, y_true)
    metrics.update({
        "simple_auc": _auc(signal.simple_score, background.simple_score),
        "paper_candidate_auc": _auc(signal.paper_score, background.paper_score),
        "signal_efficiency": metrics["recall"],
        "background_rejection": metrics["tn"] / max(metrics["tn"] + metrics["fp"], 1),
        "purity": metrics["precision"],
        "n_signal_events": int(len(signal)), "n_background_events": int(len(background)),
        "signal_planes_found": int(signal.paper_found.sum()),
        "background_planes_found": int(background.paper_found.sum()),
        "min_track_max_radius_mm": float(min(signal.max_r_q_mm.min(), signal.max_r_aq_mm.min())),
        "outer_detector_radius_mm": float(catalog.radius_mm.max()),
    })
    payload = {
        "physics": PHYSICS, "paper_cuts": PAPER_CUTS,
        "barrel_layers": catalog.to_dict("records"), "metrics": metrics,
    }
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(payload, handle, indent=2)
    _plot_results(records, output_dir, n_layers)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
