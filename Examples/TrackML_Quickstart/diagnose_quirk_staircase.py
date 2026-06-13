import os
import glob
import torch
import numpy as np

print("\n=== Locate generated event files ===")

base = "Examples/TrackML_Quickstart/datasets/quickstart_quirk_example"
files = (
    glob.glob(os.path.join(base, "**", "*.pt"), recursive=True)
    + glob.glob(os.path.join(base, "**", "*.pyg"), recursive=True)
)

print("Found files:", len(files))
for f in files[:10]:
    print(" ", f)

if not files:
    raise SystemExit("No .pt or .pyg files found. Check output directory.")

event_path = files[0]
print("\nUsing event file:", event_path)

event = torch.load(event_path, map_location="cpu")
print("\nEvent object:")
print(event)

if hasattr(event, "keys"):
    print("\nKeys:", list(event.keys()))

x = event.x.detach().cpu().numpy()
print("\nFeature shape:", x.shape)

print("\nFeature stats:")
for i in range(x.shape[1]):
    print(f"  col {i}: min={x[:, i].min():.4f}, max={x[:, i].max():.4f}")

if hasattr(event, "source_label"):
    labels = event.source_label.detach().cpu().numpy()
    print("\nSource label counts:")
    for val in np.unique(labels):
        print(f"  label {val}: {(labels == val).sum()} hits")

    quirk_mask = (labels == 1) | (labels == 2)
    qx = x[quirk_mask]

    print("\nQuirk feature ranges:")
    print(f"  col0 range: {qx[:, 0].min():.4f} to {qx[:, 0].max():.4f}")
    print(f"  col1 range: {qx[:, 1].min():.4f} to {qx[:, 1].max():.4f}")
    print(f"  col2 range: {qx[:, 2].min():.4f} to {qx[:, 2].max():.4f}")

if hasattr(event, "modulewise_true_edges"):
    e = event.modulewise_true_edges.detach().cpu().numpy()
    print("\nmodulewise_true_edges shape:", e.shape)

    src = e[0]
    dst = e[1]
    valid = (src >= 0) & (src < len(x)) & (dst >= 0) & (dst < len(x))

    print("Valid edge fraction:", valid.mean())

    dx = x[dst[valid], 0] - x[src[valid], 0]
    dy = x[dst[valid], 1] - x[src[valid], 1]
    dz = x[dst[valid], 2] - x[src[valid], 2]
    dr = np.sqrt(dx * dx + dy * dy + dz * dz)

    print("\nEdge jump distance statistics:")
    print("  min:", float(np.min(dr)))
    print("  median:", float(np.median(dr)))
    print("  p90:", float(np.percentile(dr, 90)))
    print("  p99:", float(np.percentile(dr, 99)))
    print("  max:", float(np.max(dr)))

print("\nDone.")
