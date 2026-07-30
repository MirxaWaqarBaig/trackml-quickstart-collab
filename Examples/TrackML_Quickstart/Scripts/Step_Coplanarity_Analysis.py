"""
Coplanarity-based binary event classifier for quirk detection.

Runs TWO methods and compares them:

  Method A — Simple (our original):
      Centred covariance SVD. Returns dimensionless score in [0, 1/3].
      Fast. Good for clean events.

  Method B — Paper (Knapen, Lou, Papucci, Setford 2017, arXiv:1708.02243):
      Uncentred tensor T from IP. Seeding from outer layers.
      Iterative inside-out fitting. delta_s and delta_w in mm.
      Matches the algorithm proposed for ATLAS/CMS.

Supervisor task:
    Signal   = quirk events (all mass / lambda combinations)
    Background = SM-only events
    Classify signal vs background and compare both methods.

Usage:
    python Scripts/Step_Coplanarity_Analysis.py \
        --quirk-dir    datasets/quickstart_quirk_example \
        --raw-data-dir $EXATRKX_DATA/train_all \
        --output-dir   datasets/coplanarity_results \
        --max-events   100
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_repo_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, _repo_root)

from Pipelines.TrackML_Example.LightningModules.Processing.utils.coplanarity import (
    reconstruct_xyz,
    assign_layers,
    # Simple method
    compute_coplanarity_score,
    event_coplanarity_features,
    # Paper method
    compute_T_tensor,
    paper_plane_finding,
    paper_classify_event,
    # Evaluation
    threshold_classify,
    evaluate_classifier,
    roc_curve_points,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_feature_store(directory, max_events=None):
    files = sorted(
        f for f in os.listdir(directory)
        if not f.endswith(".py") and not f.startswith(".")
    )
    if max_events:
        files = files[:max_events]
    events = []
    for fname in files:
        try:
            data = torch.load(os.path.join(directory, fname), weights_only=False)
            events.append(data)
        except Exception as e:
            print(f"  [warn] {fname}: {e}")
    return events


def load_sm_events_from_csv(raw_data_dir, max_events=100, max_hits_per_event=None):
    import pandas as pd
    from pathlib import Path
    rng = np.random.default_rng(42)
    hits_files = sorted(Path(raw_data_dir).glob("*-hits.csv"))[:max_events]
    if not hits_files:
        raise FileNotFoundError(f"No *-hits.csv in {raw_data_dir}")
    events = []
    for hf in hits_files:
        try:
            hits = pd.read_csv(hf, usecols=["x", "y", "z"])
            xyz  = hits[["x", "y", "z"]].to_numpy(dtype=np.float64)
            if max_hits_per_event and len(xyz) > max_hits_per_event:
                idx = rng.choice(len(xyz), max_hits_per_event, replace=False)
                xyz = xyz[idx]
            if len(xyz) >= 4:
                events.append({"xyz": xyz, "source": "sm_csv"})
        except Exception as e:
            print(f"  [warn] {hf.name}: {e}")
    return events


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_events_simple(events, is_feature_store=True):
    """Run the simple centred-SVD method on a list of events."""
    scores, records = [], []
    for i, ev in enumerate(events):
        try:
            xyz = reconstruct_xyz(ev.x.numpy()) if is_feature_store \
                  else ev["xyz"]
            feats = event_coplanarity_features(xyz)
            feats["event_idx"] = i
            scores.append(feats["coplanarity_all"])
            records.append(feats)
        except Exception as e:
            print(f"  [simple warn] event {i}: {e}")
    return scores, records


def score_events_paper(events, is_feature_store=True, verbose=False,
                       paper_kwargs=None):
    """
    Run the full paper algorithm on a list of events.

    paper_kwargs — overrides for paper_plane_finding() parameters.
    Defaults are adapted for our simulation's oscillation amplitudes (~5mm),
    which are larger than the paper's LHC benchmark (0.1mm).
    """
    if paper_kwargs is None:
        paper_kwargs = dict(
            ds_seed=10.0, dw_seed=50.0,
            ds_iter=10.0, dw_iter=50.0,
            ds_final=15.0, dw_final=60.0,
            min_hits_per_layer=1,
            min_layers_with_2hits=1,
        )
    results, records = [], []
    for i, ev in enumerate(events):
        try:
            xyz       = reconstruct_xyz(ev.x.numpy()) if is_feature_store \
                        else ev["xyz"]
            layer_ids = assign_layers(xyz)
            res       = paper_plane_finding(xyz, layer_ids=layer_ids,
                                            **paper_kwargs)
            res["event_idx"] = i
            results.append(res)
            records.append(res)
            if verbose and i < 5:
                print(f"    Event {i:3d}: found={res['found']}  "
                      f"ds={res['delta_s']:.3f} mm  "
                      f"dw={res['delta_w']:.3f} mm  "
                      f"n_plane_hits={res['n_plane_hits']}")
        except Exception as e:
            print(f"  [paper warn] event {i}: {e}")
    return results, records


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_score_distribution(scores_q, scores_sm, output_dir, threshold=None,
                             title="Coplanarity Score Distribution",
                             fname="coplanarity_score_distribution.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, max(np.percentile(scores_q + scores_sm, 99), 0.1), 50)
    ax.hist(scores_q,  bins=bins, alpha=0.6, color="cyan",
            label=f"Quirk signal (n={len(scores_q)})", density=True)
    ax.hist(scores_sm, bins=bins, alpha=0.6, color="orange",
            label=f"SM background (n={len(scores_sm)})", density=True)
    if threshold is not None:
        ax.axvline(threshold, color="red", ls="--",
                   label=f"Threshold = {threshold:.4f}")
    ax.set_xlabel("Coplanarity score  [λ₃/(λ₁+λ₂+λ₃)]", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(); ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_roc_curve(scores_q, scores_sm, output_dir,
                   label="Coplanarity", fname="roc_simple.png"):
    _, tpr, fpr, auc = roc_curve_points(np.array(scores_q), np.array(scores_sm))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="blue", lw=2, label=f"{label}  AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {label}", fontsize=13)
    ax.legend(); ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return auc


def plot_paper_ds_dw(results_q, results_sm, output_dir):
    """Scatter plot of delta_s vs delta_w for paper method."""
    def _extract(results):
        ds = [r["delta_s"] for r in results if r["found"] and not np.isnan(r["delta_s"])]
        dw = [r["delta_w"] for r in results if r["found"] and not np.isnan(r["delta_w"])]
        return np.array(ds), np.array(dw)

    ds_q, dw_q = _extract(results_q)
    ds_sm, dw_sm = _extract(results_sm)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: delta_s distribution
    ax = axes[0]
    all_ds = np.concatenate([ds_q, ds_sm]) if len(ds_sm) else ds_q
    bins = np.linspace(0, max(all_ds.max() if len(all_ds) else 1, 0.1), 40)
    if len(ds_q):
        ax.hist(ds_q,  bins=bins, alpha=0.6, color="cyan",
                label=f"Quirk (found={len(ds_q)})", density=True)
    if len(ds_sm):
        ax.hist(ds_sm, bins=bins, alpha=0.6, color="orange",
                label=f"SM (found={len(ds_sm)})", density=True)
    ax.axvline(1.0, color="red", ls="--", label="Paper cut: Δs < 1.0 mm")
    ax.set_xlabel("Δs — plane thickness [mm]", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Paper Method: Δs Distribution", fontsize=12)
    ax.legend(); ax.grid(True, alpha=0.3)

    # Right: delta_w distribution
    ax = axes[1]
    all_dw = np.concatenate([dw_q, dw_sm]) if len(dw_sm) else dw_q
    bins2 = np.linspace(0, max(all_dw.max() if len(all_dw) else 10, 1), 40)
    if len(dw_q):
        ax.hist(dw_q,  bins=bins2, alpha=0.6, color="cyan",
                label=f"Quirk (found={len(dw_q)})", density=True)
    if len(dw_sm):
        ax.hist(dw_sm, bins=bins2, alpha=0.6, color="orange",
                label=f"SM (found={len(dw_sm)})", density=True)
    ax.axvline(10.0, color="red", ls="--", label="Paper cut: Δw < 10 mm")
    ax.set_xlabel("Δw — oscillation width [mm]", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Paper Method: Δw Distribution", fontsize=12)
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle("Paper Algorithm (Knapen et al. 2017) — Δs and Δw", fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, "paper_ds_dw_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_3d_event_with_plane(event_data, event_idx, paper_result, output_dir,
                              is_feature_store=True):
    """3D scatter of one event with both simple and paper plane overlaid."""
    xyz    = reconstruct_xyz(event_data.x.numpy()) if is_feature_store \
             else event_data["xyz"]
    labels = event_data.source_label.numpy() \
             if (is_feature_store and hasattr(event_data, "source_label")) else None

    score, normal_s, centroid, residuals = compute_coplanarity_score(xyz)

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")

    # Hits coloured by source label
    if labels is not None:
        for lbl, col, name in [(1, "cyan", "Quirk"),
                                (2, "magenta", "Anti-Quirk"),
                                (0, "gray", "SM")]:
            mask = labels == lbl
            if mask.any():
                ax.scatter(xyz[mask, 0], xyz[mask, 1], xyz[mask, 2],
                           c=col, s=25, label=name, alpha=0.9, zorder=3)
    else:
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c="cyan", s=20)

    # Simple method plane (yellow)
    def _draw_plane(ax, normal, centroid, span, color, alpha, label):
        u = np.array([1., 0., 0.])
        if abs(np.dot(normal, u)) > 0.9:
            u = np.array([0., 1., 0.])
        u = np.cross(normal, u)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        g = np.linspace(-span, span, 8)
        G1, G2 = np.meshgrid(g, g)
        Px = centroid[0] + G1 * u[0] + G2 * v[0]
        Py = centroid[1] + G1 * u[1] + G2 * v[1]
        Pz = centroid[2] + G1 * u[2] + G2 * v[2]
        ax.plot_surface(Px, Py, Pz, alpha=alpha, color=color)

    span = max(np.abs(xyz - centroid).max() * 0.7, 100.)
    _draw_plane(ax, normal_s, centroid, span, "yellow", 0.12,
                f"Simple plane (score={score:.4f})")

    # Paper method plane (green) if found
    if paper_result and paper_result["found"] and paper_result["n1"] is not None:
        plane_idx = paper_result["plane_hit_idx"]
        pxyz = xyz[plane_idx]
        ax.scatter(pxyz[:, 0], pxyz[:, 1], pxyz[:, 2],
                   c="lime", s=80, marker="*", zorder=5,
                   label=f"Paper inliers (n={len(plane_idx)})")
        c_paper = pxyz.mean(axis=0)
        _draw_plane(ax, paper_result["n1"], c_paper, span * 0.6, "green", 0.12,
                    f"Paper plane (Δs={paper_result['delta_s']:.2f}mm)")

    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")
    ax.set_title(f"Event {event_idx} — Simple score={score:.4f}  |  "
                 f"Paper found={paper_result['found'] if paper_result else False}",
                 fontsize=10)
    ax.legend(fontsize=7, loc="upper left")

    path = os.path.join(output_dir, f"event_{event_idx:04d}_3d_both_methods.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_residuals(event_data, event_idx, output_dir, is_feature_store=True):
    xyz    = reconstruct_xyz(event_data.x.numpy()) if is_feature_store \
             else event_data["xyz"]
    labels = event_data.source_label.numpy() \
             if (is_feature_store and hasattr(event_data, "source_label")) else None
    _, _, _, residuals = compute_coplanarity_score(xyz)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(-200, 200, 60)
    if labels is not None:
        for lbl, col, name in [(1, "cyan", "Quirk"),
                                (2, "magenta", "Anti-Quirk"),
                                (0, "gray", "SM")]:
            mask = labels == lbl
            if mask.any():
                ax.hist(residuals[mask], bins=bins, alpha=0.6, color=col, label=name)
    else:
        ax.hist(residuals, bins=bins, alpha=0.7, color="cyan")
    ax.set_xlabel("Distance from best-fit plane [mm]", fontsize=11)
    ax.set_ylabel("Hits", fontsize=11)
    ax.set_title(f"Event {event_idx} — Residuals from Quirk Plane", fontsize=12)
    ax.legend(); ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, f"event_{event_idx:04d}_residuals.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_comparison_summary(simple_q, simple_sm, paper_q, paper_sm, output_dir):
    """Side-by-side ROC curves for simple vs paper method."""
    fig, ax = plt.subplots(figsize=(7, 6))

    # Simple method ROC
    if simple_sm:
        _, tpr_s, fpr_s, auc_s = roc_curve_points(
            np.array(simple_q), np.array(simple_sm))
        ax.plot(fpr_s, tpr_s, color="blue", lw=2,
                label=f"Simple (centred SVD)  AUC = {auc_s:.4f}")

    # Paper method ROC using delta_s as score
    if paper_sm:
        ds_q  = np.array([r["delta_s"] if (r["found"] and not np.isnan(r["delta_s"]))
                           else 999. for r in paper_q])
        ds_sm = np.array([r["delta_s"] if (r["found"] and not np.isnan(r["delta_s"]))
                           else 999. for r in paper_sm])
        if len(ds_q) and len(ds_sm):
            _, tpr_p, fpr_p, auc_p = roc_curve_points(ds_q, ds_sm)
            ax.plot(fpr_p, tpr_p, color="green", lw=2, ls="--",
                    label=f"Paper (Knapen et al.)  AUC = {auc_p:.4f}")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Comparison: Simple vs Paper Coplanarity Method", fontsize=12)
    ax.legend(); ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "roc_comparison_simple_vs_paper.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_layer_hit_counts(events, paper_results, output_dir, n_events=5):
    """Bar chart showing which detector layers have hits for the first N events."""
    fig, axes = plt.subplots(1, min(n_events, len(events)),
                              figsize=(4 * min(n_events, len(events)), 4),
                              sharey=True)
    if n_events == 1:
        axes = [axes]

    for i, (ev, res) in enumerate(zip(events[:n_events],
                                       paper_results[:n_events])):
        ax = axes[i]
        xyz       = reconstruct_xyz(ev.x.numpy())
        layer_ids = assign_layers(xyz)
        labels    = ev.source_label.numpy() if hasattr(ev, "source_label") else None

        from collections import Counter
        all_counts = Counter(layer_ids[layer_ids > 0])
        layers     = sorted(all_counts.keys())
        counts     = [all_counts[l] for l in layers]

        ax.bar(layers, counts, color="steelblue", alpha=0.6, label="All hits")

        # Highlight paper-identified inliers
        if res["found"] and len(res["plane_hit_idx"]):
            inlier_layers = layer_ids[res["plane_hit_idx"]]
            in_counts = Counter(inlier_layers[inlier_layers > 0])
            in_l = sorted(in_counts.keys())
            in_c = [in_counts[l] for l in in_l]
            ax.bar(in_l, in_c, color="lime", alpha=0.8, label="Paper inliers")

        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylabel("Hits" if i == 0 else "", fontsize=10)
        ax.set_title(f"Event {i}\nΔs={res['delta_s']:.2f}mm" if res["found"]
                     else f"Event {i}\n(not found)", fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xticks(range(1, 9))

    fig.suptitle("Hit Distribution per Detector Layer", fontsize=12)
    fig.tight_layout()
    path = os.path.join(output_dir, "layer_hit_counts.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_metrics(label, y_pred, y_true, threshold):
    m = evaluate_classifier(y_pred, y_true)
    print(f"\n  [{label}]  threshold = {threshold:.4f}")
    print(f"    Accuracy  = {m['accuracy']:.3f}")
    print(f"    Precision = {m['precision']:.3f}")
    print(f"    Recall    = {m['recall']:.3f}")
    print(f"    F1        = {m['f1']:.3f}")
    print(f"    TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")


def auto_threshold(scores_q, scores_sm):
    """Choose threshold that maximises F1 on this dataset."""
    all_s  = np.array(scores_q + scores_sm)
    y_true = np.array([1] * len(scores_q) + [0] * len(scores_sm))
    best_f1, best_thr = 0.0, 0.0
    for thr in np.linspace(all_s.min(), all_s.max(), 300):
        y_pred = threshold_classify(all_s, thr)
        f1 = evaluate_classifier(y_pred, y_true)["f1"]
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


def main():
    parser = argparse.ArgumentParser(
        description="Coplanarity classifier — simple method vs paper method")
    parser.add_argument("--quirk-dir",    required=True)
    parser.add_argument("--sm-dir",       default=None)
    parser.add_argument("--raw-data-dir", default=None)
    parser.add_argument("--output-dir",   default="datasets/coplanarity_results")
    parser.add_argument("--max-events",   type=int, default=None)
    parser.add_argument("--plot-events",  type=int, default=3)
    parser.add_argument("--threshold",    type=float, default=None)
    parser.add_argument("--sm-max-hits",  type=int, default=2000,
                        help="Subsample SM events to this many hits (speeds up paper method)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n=== Coplanarity Quirk Classifier: Simple vs Paper ===\n")

    # ── Load quirk events ────────────────────────────────────────────────────
    print(f"Loading quirk events: {args.quirk_dir}")
    quirk_events = load_feature_store(args.quirk_dir, max_events=args.max_events)
    print(f"  Loaded {len(quirk_events)} events")

    # ── Load SM events ───────────────────────────────────────────────────────
    sm_events = []
    if args.sm_dir:
        print(f"Loading SM events (feature store): {args.sm_dir}")
        sm_events = load_feature_store(args.sm_dir, max_events=args.max_events)
        sm_fs = True
    elif args.raw_data_dir:
        print(f"Loading SM events (raw CSV): {args.raw_data_dir}")
        sm_events = load_sm_events_from_csv(
            args.raw_data_dir, max_events=args.max_events or 100,
            max_hits_per_event=args.sm_max_hits)
        print(f"  (subsampled to max {args.sm_max_hits} hits/event for paper algo speed)")
        sm_fs = False
    else:
        print("  No SM events provided — quirk-only mode.\n")
        sm_fs = False
    print(f"  Loaded {len(sm_events)} SM events")

    # ── Method A: Simple centred SVD ─────────────────────────────────────────
    print("\n--- Method A: Simple (centred covariance SVD) ---")
    simple_q_scores, _ = score_events_simple(quirk_events, is_feature_store=True)
    print(f"  Quirk  — mean={np.mean(simple_q_scores):.4f}  "
          f"std={np.std(simple_q_scores):.4f}  "
          f"range=[{np.min(simple_q_scores):.4f}, {np.max(simple_q_scores):.4f}]")

    simple_sm_scores = []
    if sm_events:
        simple_sm_scores, _ = score_events_simple(sm_events, is_feature_store=sm_fs)
        print(f"  SM     — mean={np.mean(simple_sm_scores):.4f}  "
              f"std={np.std(simple_sm_scores):.4f}  "
              f"range=[{np.min(simple_sm_scores):.4f}, {np.max(simple_sm_scores):.4f}]")

    # ── Method B: Paper algorithm ─────────────────────────────────────────────
    print("\n--- Method B: Paper Algorithm (Knapen et al. 2017) ---")
    print("  Scoring quirk events (seeding + iterative fitting)...")
    paper_q_results, _ = score_events_paper(quirk_events, is_feature_store=True,
                                             verbose=True)
    n_found_q = sum(1 for r in paper_q_results if r["found"])
    ds_q = [r["delta_s"] for r in paper_q_results if r["found"]
            and not np.isnan(r["delta_s"])]
    dw_q = [r["delta_w"] for r in paper_q_results if r["found"]
            and not np.isnan(r["delta_w"])]
    print(f"  Quirk: plane found in {n_found_q}/{len(quirk_events)} events")
    if ds_q:
        print(f"    Δs — mean={np.mean(ds_q):.3f} mm  "
              f"range=[{np.min(ds_q):.3f}, {np.max(ds_q):.3f}] mm")
        print(f"    Δw — mean={np.mean(dw_q):.3f} mm  "
              f"range=[{np.min(dw_q):.3f}, {np.max(dw_q):.3f}] mm")

    paper_sm_results = []
    if sm_events:
        print("  Scoring SM events...")
        paper_sm_results, _ = score_events_paper(sm_events, is_feature_store=sm_fs)
        n_found_sm = sum(1 for r in paper_sm_results if r["found"])
        ds_sm = [r["delta_s"] for r in paper_sm_results if r["found"]
                 and not np.isnan(r["delta_s"])]
        print(f"  SM:    plane found in {n_found_sm}/{len(sm_events)} events")
        if ds_sm:
            print(f"    Δs — mean={np.mean(ds_sm):.3f} mm  "
                  f"range=[{np.min(ds_sm):.3f}, {np.max(ds_sm):.3f}] mm")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")

    # Individual event 3D plots
    n_plot = min(args.plot_events, len(quirk_events))
    for i in range(n_plot):
        pr = paper_q_results[i] if i < len(paper_q_results) else None
        plot_3d_event_with_plane(quirk_events[i], i, pr, args.output_dir)
        plot_residuals(quirk_events[i], i, args.output_dir)

    # Layer hit count bar charts
    plot_layer_hit_counts(quirk_events, paper_q_results, args.output_dir,
                          n_events=min(5, len(quirk_events)))

    # Paper Δs and Δw distributions
    plot_paper_ds_dw(paper_q_results, paper_sm_results, args.output_dir)

    if simple_sm_scores:
        # ── Classification metrics — Method A ────────────────────────────────
        print("\n=== Classification Results ===")
        thr_a, f1_a = auto_threshold(simple_q_scores, simple_sm_scores)
        all_s  = np.array(simple_q_scores + simple_sm_scores)
        y_true = np.array([1] * len(simple_q_scores) + [0] * len(simple_sm_scores))
        y_pred_a = threshold_classify(all_s, thr_a)
        print_metrics("Simple method", y_pred_a, y_true, thr_a)

        # Score distribution plot
        plot_score_distribution(simple_q_scores, simple_sm_scores,
                                args.output_dir, threshold=thr_a)

        # ROC for simple method
        plot_roc_curve(simple_q_scores, simple_sm_scores, args.output_dir,
                       label="Simple (centred SVD)", fname="roc_simple.png")

        # ── Classification metrics — Method B ────────────────────────────────
        if paper_sm_results:
            # Use delta_s as the classification score (lower = more signal)
            # Events where plane was not found get a large sentinel score
            SENTINEL = 9999.0
            ds_q_cls  = [r["delta_s"] if (r["found"] and not np.isnan(r["delta_s"]))
                          else SENTINEL for r in paper_q_results]
            ds_sm_cls = [r["delta_s"] if (r["found"] and not np.isnan(r["delta_s"]))
                          else SENTINEL for r in paper_sm_results]

            thr_b, f1_b = auto_threshold(ds_q_cls, ds_sm_cls)
            all_ds  = np.array(ds_q_cls + ds_sm_cls)
            y_pred_b = threshold_classify(all_ds, thr_b)
            print_metrics("Paper method (Δs)", y_pred_b, y_true, thr_b)

            # Δs/Δw distributions
            plot_roc_curve(ds_q_cls, ds_sm_cls, args.output_dir,
                           label="Paper (Knapen et al.) — Δs", fname="roc_paper.png")

        # ── Combined ROC comparison ───────────────────────────────────────────
        plot_comparison_summary(simple_q_scores, simple_sm_scores,
                                paper_q_results, paper_sm_results,
                                args.output_dir)

    # ── Save scores CSV ───────────────────────────────────────────────────────
    out_path = os.path.join(args.output_dir, "coplanarity_scores.txt")
    with open(out_path, "w") as f:
        f.write("class,simple_score,paper_found,paper_ds,paper_dw,"
                "paper_n_plane_hits,paper_n_layers_2hits\n")
        for i, sq in enumerate(simple_q_scores):
            pr = paper_q_results[i] if i < len(paper_q_results) else {}
            f.write(f"quirk,{sq:.6f},{pr.get('found', False)},"
                    f"{pr.get('delta_s', np.nan):.4f},"
                    f"{pr.get('delta_w', np.nan):.4f},"
                    f"{pr.get('n_plane_hits', 0)},"
                    f"{pr.get('n_layers_2hits', 0)}\n")
        for i, sq in enumerate(simple_sm_scores):
            pr = paper_sm_results[i] if i < len(paper_sm_results) else {}
            f.write(f"sm,{sq:.6f},{pr.get('found', False)},"
                    f"{pr.get('delta_s', np.nan):.4f},"
                    f"{pr.get('delta_w', np.nan):.4f},"
                    f"{pr.get('n_plane_hits', 0)},"
                    f"{pr.get('n_layers_2hits', 0)}\n")
    print(f"\n  Scores saved: {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
