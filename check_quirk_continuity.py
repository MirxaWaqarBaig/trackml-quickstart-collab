"""
Diagnostic script: inspect Step 0 quirk+SM feature-store files and report
whether Quirk truth tracks are continuous.

Usage:
    python check_quirk_continuity.py [--data-dir PATH] [--n-events N]

Checks performed per event
--------------------------
1.  Number of SM hits          (source_label == 0)
2.  Number of Quirk hits       (source_label == 1)
3.  Number of Anti-Quirk hits  (source_label == 2)
4.  Quirk r range              (normalised units from data.x[:,0])
5.  Quirk z range              (normalised units from data.x[:,2])
6.  Number of Quirk true edges        (both endpoints source_label == 1)
7.  Number of Anti-Quirk true edges   (both endpoints source_label == 2)
8.  Number of mixed Quirk-SM edges    (one endpoint quirk, one SM)
9.  Connected-component sizes for Quirk-only true graph
10. Number of unique detector modules touched by Quirk hits
11. Number of unique detector layers touched by Quirk hits (if available)

Continuity-loss root-cause notes
---------------------------------
A. Trajectory simulation   -- if Quirk r range is near 0, sep0 is too small;
                               the pair never reaches outer detector layers.
B. Detector intersection   -- if quirk hits exist but edge count is 0, the
                               module-tolerance window is too tight and
                               sequential hits fall on different modules with
                               no accepted intersection.
C. True-edge construction  -- edges are built by _build_edges_from_group_order
                               which chains sorted hits sequentially; gaps arise
                               only when fewer than 2 hits share a particle_id.
D. Plotting code           -- source_label may be absent in old cached files;
                               convenience_utils._get_source_labels guards this
                               but the display falls back to Viridis colors.
E. Step 5 track building   -- connected-components on a score-cut graph will
                               fragment sparse Quirk subgraphs; if component
                               sizes here are already small (1-2 nodes each)
                               the GNN cannot recover them.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Quirk continuity diagnostic")
    parser.add_argument(
        "--data-dir",
        default="Examples/TrackML_Quickstart/datasets/quickstart_quirk_example",
        help="Path to Step 0 output directory",
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=5,
        help="Number of events to inspect (default: 5)",
    )
    return parser.parse_args()


def connected_components(node_ids, edge_index):
    """Union-Find connected components on a subset of nodes."""
    parent = {n: n for n in node_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        if u in parent and v in parent:
            union(u, v)

    from collections import defaultdict
    groups = defaultdict(list)
    for n in node_ids:
        groups[find(n)].append(n)
    return [sorted(g) for g in groups.values()]


def inspect_event(path):
    data = torch.load(path, map_location="cpu", weights_only=False)

    # ---- source labels ----
    if not hasattr(data, "source_label"):
        print("  [WARN] No source_label found — old cached file, cannot classify hits.")
        return

    src = data.source_label.numpy()
    n_sm  = int((src == 0).sum())
    n_q   = int((src == 1).sum())
    n_aq  = int((src == 2).sum())
    total = int(src.size)

    print(f"  Hits  — SM: {n_sm}  Quirk: {n_q}  Anti-Quirk: {n_aq}  Total: {total}")

    # ---- spatial ranges ----
    x = data.x.numpy()  # (N, 3) normalised [r/1000, phi/pi, z/1000]
    q_mask  = src == 1
    aq_mask = src == 2

    if q_mask.sum() > 0:
        r_q = x[q_mask, 0]
        z_q = x[q_mask, 2]
        print(f"  Quirk r range (norm): [{r_q.min():.4f}, {r_q.max():.4f}]"
              f"  =>  [{r_q.min()*1000:.1f} mm, {r_q.max()*1000:.1f} mm]")
        print(f"  Quirk z range (norm): [{z_q.min():.4f}, {z_q.max():.4f}]"
              f"  =>  [{z_q.min()*1000:.1f} mm, {z_q.max()*1000:.1f} mm]")
    else:
        print("  Quirk r/z range: N/A (no quirk hits)")

    if aq_mask.sum() > 0:
        r_aq = x[aq_mask, 0]
        print(f"  Anti-Quirk r range (norm): [{r_aq.min():.4f}, {r_aq.max():.4f}]"
              f"  =>  [{r_aq.min()*1000:.1f} mm, {r_aq.max()*1000:.1f} mm]")

    # ---- true edges ----
    edges = None
    for attr in ["modulewise_true_edges", "layerwise_true_edges"]:
        if hasattr(data, attr):
            edges = getattr(data, attr).numpy()
            break

    if edges is None or edges.shape[1] == 0:
        print("  True edges: NONE (root cause likely B or C)")
        return

    src0 = src[edges[0]]
    src1 = src[edges[1]]

    n_q_edges   = int(((src0 == 1) & (src1 == 1)).sum())
    n_aq_edges  = int(((src0 == 2) & (src1 == 2)).sum())
    n_mix_edges = int((((src0 == 1) | (src0 == 2)) & (src1 == 0)).sum()
                    + ((src0 == 0) & ((src1 == 1) | (src1 == 2))).sum())

    print(f"  True edges — Quirk: {n_q_edges}  Anti-Quirk: {n_aq_edges}  Mixed Quirk-SM: {n_mix_edges}")

    # ---- connected components (quirk-only subgraph) ----
    q_node_ids = set(np.where(q_mask)[0].tolist())
    q_edge_mask = (src0 == 1) & (src1 == 1)
    q_edges = edges[:, q_edge_mask]

    if len(q_node_ids) > 0:
        components = connected_components(q_node_ids, q_edges)
        sizes = sorted([len(c) for c in components], reverse=True)
        print(f"  Quirk connected components: {len(components)} total, "
              f"sizes (top 10): {sizes[:10]}")
        if max(sizes) <= 2:
            print("  [WARN] All Quirk components have <=2 nodes — likely cause A or B")
    else:
        print("  Quirk connected components: N/A (no quirk hits)")

    # ---- module and layer coverage ----
    if hasattr(data, "modules"):
        modules = data.modules.numpy()
        q_modules = modules[q_mask]
        print(f"  Unique detector modules touched by Quirk: {len(np.unique(q_modules))}")
    else:
        print("  Module info: not available in this file")

    if hasattr(data, "layers"):
        layers = data.layers.numpy()
        q_layers = layers[q_mask]
        print(f"  Unique detector layers touched by Quirk: {len(np.unique(q_layers))}")
    else:
        print("  Layer info: not stored separately (modules encode layer via layer_id)")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"ERROR: data directory not found: {data_dir}")
        print("Run Step 0 first:")
        print("  python Examples/TrackML_Quickstart/Scripts/Step_0_Build_Quirk_Feature_Store.py --n-files 5")
        sys.exit(1)

    event_files = sorted([
        f for f in data_dir.iterdir()
        if f.is_file() and not f.suffix and f.name.isdigit()
    ])

    if not event_files:
        # also accept files with no extension that are numeric
        event_files = sorted([f for f in data_dir.iterdir() if f.is_file()])

    if not event_files:
        print(f"ERROR: No event files found in {data_dir}")
        sys.exit(1)

    event_files = event_files[: args.n_events]
    print(f"Inspecting {len(event_files)} event(s) from {data_dir}\n")
    print("=" * 70)

    for ef in event_files:
        print(f"Event: {ef.name}")
        try:
            inspect_event(ef)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
        print()

    print("=" * 70)
    print("Continuity root-cause guide:")
    print("  A. r range near 0 (< 0.05)  => sep0 too small in quirk_trajectory.py")
    print("  B. Quirk edges = 0 despite hits => tolerance_mm too tight in intersections")
    print("  C. Edges exist but components all size 1 => edge-building gap (particle_id mismatch)")
    print("  D. Plots look wrong but data OK => source_label missing in old cache; delete and regenerate")
    print("  E. Components small here => Step 5 will fragment further; increase score_cut or min_track_length")


if __name__ == "__main__":
    main()
