"""
Diagnostic script: professor's geometry-based quirk hit finding.

Tests the exact segment-plane intersection method on Step-0 .pt files and
produces a before/after report across up to N events.

Usage
-----
    python check_module_intersection.py \
        --data-dir Examples/TrackML_Quickstart/datasets/quickstart_quirk_example \
        --detector-path trackml_raw/detectors.csv \
        --n-events 10

Output
------
  - Per-event table: old hits (proximity) vs new hits (exact geometry)
  - First 5 module intersection diagnostics for event 0 (corners, F, local u/v)
  - Aggregate summary over all events
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from Pipelines.TrackML_Example.LightningModules.Processing.utils.quirk_intersections import (
    build_detector_module_table,
    intersect_track_with_modules,
    compute_before_after_stats,
    print_diagnostic_log,
)
from Pipelines.TrackML_Example.LightningModules.Processing.utils.quirk_trajectory import (
    simulate_quirk_pair_tracks,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Module-intersection diagnostics")
    p.add_argument("--data-dir",       default="Examples/TrackML_Quickstart/datasets/quickstart_quirk_example/train")
    p.add_argument("--detector-path",  default="trackml_raw/detectors.csv")
    p.add_argument("--n-events",       type=int, default=5)
    p.add_argument("--max-hits",       type=int, default=150)
    p.add_argument("--quirk-pt",       type=float, default=6.0)
    p.add_argument("--quirk-pz",       type=float, default=3.0)
    p.add_argument("--n-steps",        type=int, default=3000)
    p.add_argument("--simulate",       action="store_true",
                   help="Simulate fresh quirk pairs instead of loading .pt files")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Load detector modules
# ---------------------------------------------------------------------------
def load_detector(detector_path):
    det_path = Path(detector_path)
    if not det_path.exists():
        print(f"[WARN] detectors.csv not found at {det_path}. Using synthetic modules.")
        from Pipelines.TrackML_Example.LightningModules.Processing.utils.quirk_intersections import (
            make_synthetic_detector_modules,
        )
        return make_synthetic_detector_modules()
    det_df = pd.read_csv(det_path)
    return build_detector_module_table(det_df)


# ---------------------------------------------------------------------------
# Get quirk trajectories from .pt files or simulation
# ---------------------------------------------------------------------------
def get_trajectories_from_pt(data_dir, n_events, n_steps):
    """
    Load Step-0 .pt event files and extract quirk trajectory arrays from
    stored source_label / hit positions.
    Returns list of (track_xyz, event_id) tuples.
    """
    pt_files = sorted(Path(data_dir).glob("*.pt"))[:n_events]
    if not pt_files:
        return []
    trajs = []
    for pt_file in pt_files:
        data = torch.load(pt_file, map_location="cpu")
        if not hasattr(data, "source_label"):
            continue
        lbl = data.source_label
        x   = data.x.numpy()  # (N, 3) normalized
        # Denormalize: r in [0,1] -> mm, phi in [0,1] -> radians, z in [0,1] -> mm
        r_mm  = x[:, 0] * 1000.0
        phi   = x[:, 1] * np.pi
        z_mm  = x[:, 2] * 1000.0
        xyz   = np.stack([
            r_mm * np.cos(phi),
            r_mm * np.sin(phi),
            z_mm,
        ], axis=1)
        for q_type in [1, 2]:
            mask = (lbl == q_type).numpy().astype(bool)
            if mask.sum() >= 5:
                trajs.append((xyz[mask], str(pt_file.stem) + f"_q{q_type}"))
    return trajs


def get_trajectories_from_simulation(n_events, n_steps, quirk_pt, quirk_pz):
    """
    Simulate n_events quirk pairs and return trajectory arrays.
    Returns list of (track_xyz_mm, label) tuples.
    """
    trajs = []
    for i in range(n_events):
        try:
            result = simulate_quirk_pair_tracks(
                event_id=i,
                pair_pt=quirk_pt,
                pair_pz=quirk_pz,
                n_steps=n_steps,
                b_field=2.0,
                string_tension=0.2,
                velocity_scale=1200.0,
            )
            # result is a dict; trajectories are under 'quirk_xyz' and 'anti_quirk_xyz'
            # or result may be a tuple — check
            if isinstance(result, tuple):
                traj_q, traj_aq = result[0], result[1]
            elif isinstance(result, dict):
                traj_q  = result.get("xyz_q",  result.get("quirk_xyz",      result.get("track1", None)))
                traj_aq = result.get("xyz_aq", result.get("anti_quirk_xyz", result.get("track2", None)))
            else:
                continue
            for traj, lbl in [(traj_q, f"sim_{i}_q"), (traj_aq, f"sim_{i}_aq")]:
                if traj is not None and len(traj) >= 10:
                    trajs.append((np.asarray(traj), lbl))
        except Exception as e:
            print(f"  [WARN] simulation {i} failed: {e}")
    return trajs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    print("\n" + "=" * 70)
    print("  Quirk Module Intersection Diagnostics")
    print("  Branch: quirk-module-intersection")
    print("=" * 70)

    # Load detector
    print(f"\nLoading detector geometry from: {args.detector_path}")
    det_mods = load_detector(args.detector_path)
    print(f"  {len(det_mods)} modules loaded")
    print(f"  half_u range: {det_mods['half_u'].min():.1f} – {det_mods['half_u'].max():.1f} mm")
    print(f"  half_v range: {det_mods['half_v'].min():.1f} – {det_mods['half_v'].max():.1f} mm")

    # Get trajectories
    if args.simulate:
        print(f"\nSimulating {args.n_events} quirk pairs ...")
        trajs = get_trajectories_from_simulation(
            args.n_events, args.n_steps, args.quirk_pt, args.quirk_pz
        )
    else:
        print(f"\nLoading trajectories from Step-0 files in: {args.data_dir}")
        trajs = get_trajectories_from_pt(args.data_dir, args.n_events, args.n_steps)
        if not trajs:
            print("  No .pt files found or no quirk hits in them.")
            print("  Re-running with --simulate flag ...")
            trajs = get_trajectories_from_simulation(
                args.n_events, args.n_steps, args.quirk_pt, args.quirk_pz
            )

    if not trajs:
        print("[ERROR] No trajectories to analyse. Check --data-dir or use --simulate.")
        sys.exit(1)

    print(f"  {len(trajs)} quirk track trajectories to analyse")

    # Per-track stats
    print("\n" + "-" * 70)
    print(f"  {'Track':<30}  {'Steps':>6}  {'Segs':>6}  {'OldHits':>8}  {'NewHits':>8}  {'Delta':>6}")
    print("-" * 70)

    all_stats   = []
    first_diag  = None

    for track_xyz, track_id in trajs:
        stats = compute_before_after_stats(
            track_xyz, det_mods, max_hits=args.max_hits,
        )
        all_stats.append(stats)
        delta = stats["new_hits"] - stats["old_hits"]
        sign  = "+" if delta >= 0 else ""
        print(
            f"  {track_id:<30}  "
            f"{stats['n_trajectory_steps']:>6}  "
            f"{stats['n_segments']:>6}  "
            f"{stats['old_hits']:>8}  "
            f"{stats['new_hits']:>8}  "
            f"{sign}{delta:>5}"
        )
        if first_diag is None and stats["diag_log"]:
            first_diag = stats["diag_log"]

    # Aggregate
    n         = len(all_stats)
    avg_old   = np.mean([s["old_hits"] for s in all_stats])
    avg_new   = np.mean([s["new_hits"] for s in all_stats])
    avg_segs  = np.mean([s["n_segments"] for s in all_stats])
    avg_in    = np.mean([s["n_inside_boundary"]  for s in all_stats])
    avg_out   = np.mean([s["n_outside_boundary"] for s in all_stats])

    print("\n" + "=" * 70)
    print(f"  AGGREGATE SUMMARY  ({n} tracks)")
    print("=" * 70)
    print(f"  Avg segments per track       : {avg_segs:.0f}")
    print(f"  Avg hits  — OLD (proximity)  : {avg_old:.1f}")
    print(f"  Avg hits  — NEW (exact geom) : {avg_new:.1f}")
    print(f"  Avg delta (new - old)        : {avg_new - avg_old:+.1f}")
    print(f"  Avg candidates inside bound  : {avg_in:.1f}  (accepted)")
    print(f"  Avg candidates outside bound : {avg_out:.1f}  (rejected)")
    if avg_in + avg_out > 0:
        acceptance = 100.0 * avg_in / (avg_in + avg_out)
        print(f"  Module boundary acceptance   : {acceptance:.1f}%")

    # Detailed diagnostics for first track
    if first_diag:
        print_diagnostic_log(first_diag, max_entries=5)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
