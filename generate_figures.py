"""
Generate matplotlib PNG figures for the TrackML quirk particle tracking pipeline report.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUTPUT_DIR = "/home/waqar/rafia_clone/docs_from_june/latest_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE = "/home/waqar/rafia_clone/trackml-quickstart-collab/Examples/TrackML_Quickstart/artifacts"
ML_BASE  = os.path.join(BASE, "metric_learning/trackml_quickstart_quirk")
GNN_BASE = os.path.join(BASE, "gnn/trackml_quickstart_quirk")

DPI = 150

# ──────────────────────────────────────────────────────────────────────────────
# Helper: load a metrics.csv and separate val/train rows
# ──────────────────────────────────────────────────────────────────────────────
def load_metrics(path):
    df = pd.read_csv(path)
    # val rows: val_loss is present
    val = df.dropna(subset=["val_loss"]).copy()
    val = val.sort_values("epoch").reset_index(drop=True)
    # train rows: train_loss is present
    train = df.dropna(subset=["train_loss"]).copy()
    train = train.sort_values("epoch").reset_index(drop=True)
    return val, train

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 1: ml_loss.png
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Figure 1: ml_loss.png ===")
ml_val, ml_train = load_metrics(os.path.join(ML_BASE, "version_2/metrics.csv"))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ml_train["epoch"], ml_train["train_loss"], color="orange", label="Train Loss", linewidth=1.5)
ax.plot(ml_val["epoch"],   ml_val["val_loss"],     color="blue",   label="Val Loss",   linewidth=1.5)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("Metric Learning: Training & Validation Loss (100 epochs)", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "ml_loss.png")
plt.savefig(out, dpi=DPI)
plt.close()
print(f"Saved: {out}")
print(f"  Train loss: start={ml_train['train_loss'].iloc[0]:.6f}, end={ml_train['train_loss'].iloc[-1]:.6f}")
print(f"  Val   loss: start={ml_val['val_loss'].iloc[0]:.6f},   end={ml_val['val_loss'].iloc[-1]:.6f}")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 2: ml_eff_pur.png
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Figure 2: ml_eff_pur.png ===")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(ml_val["epoch"], ml_val["eff"], color="steelblue", linewidth=1.5)
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Efficiency", fontsize=12)
ax1.set_title("Efficiency vs Epoch", fontsize=12)
ax1.grid(True, alpha=0.3)

ax2.plot(ml_val["epoch"], ml_val["pur"], color="darkorange", linewidth=1.5)
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Purity", fontsize=12)
ax2.set_title("Purity vs Epoch", fontsize=12)
ax2.grid(True, alpha=0.3)

fig.suptitle("Metric Learning: Efficiency and Purity", fontsize=14, fontweight='bold')
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "ml_eff_pur.png")
plt.savefig(out, dpi=DPI)
plt.close()
print(f"Saved: {out}")
print(f"  Best eff={ml_val['eff'].max():.4f} (epoch {ml_val.loc[ml_val['eff'].idxmax(),'epoch']})")
print(f"  Best pur={ml_val['pur'].max():.4f} (epoch {ml_val.loc[ml_val['pur'].idxmax(),'epoch']})")
print(f"  Final eff={ml_val['eff'].iloc[-1]:.4f}, Final pur={ml_val['pur'].iloc[-1]:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 3: gnn_loss.png
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Figure 3: gnn_loss.png ===")
gnn_val, gnn_train = load_metrics(os.path.join(GNN_BASE, "version_2/metrics.csv"))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(gnn_train["epoch"], gnn_train["train_loss"], color="orange", label="Train Loss", linewidth=1.5)
ax.plot(gnn_val["epoch"],   gnn_val["val_loss"],     color="blue",   label="Val Loss",   linewidth=1.5)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("GNN: Training & Validation Loss (30 epochs)", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "gnn_loss.png")
plt.savefig(out, dpi=DPI)
plt.close()
print(f"Saved: {out}")
print(f"  Train loss: start={gnn_train['train_loss'].iloc[0]:.6f}, end={gnn_train['train_loss'].iloc[-1]:.6f}")
print(f"  Val   loss: start={gnn_val['val_loss'].iloc[0]:.6f},   end={gnn_val['val_loss'].iloc[-1]:.6f}")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 4: gnn_metrics.png
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Figure 4: gnn_metrics.png ===")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(gnn_val["epoch"], gnn_val["auc"], color="purple", linewidth=1.5)
axes[0].set_xlabel("Epoch", fontsize=12)
axes[0].set_ylabel("AUC", fontsize=12)
axes[0].set_title("AUC vs Epoch", fontsize=12)
axes[0].grid(True, alpha=0.3)

axes[1].plot(gnn_val["epoch"], gnn_val["eff"], color="steelblue", linewidth=1.5)
axes[1].set_xlabel("Epoch", fontsize=12)
axes[1].set_ylabel("Efficiency", fontsize=12)
axes[1].set_title("Efficiency vs Epoch", fontsize=12)
axes[1].grid(True, alpha=0.3)

axes[2].plot(gnn_val["epoch"], gnn_val["pur"], color="darkorange", linewidth=1.5)
axes[2].set_xlabel("Epoch", fontsize=12)
axes[2].set_ylabel("Purity", fontsize=12)
axes[2].set_title("Purity vs Epoch", fontsize=12)
axes[2].grid(True, alpha=0.3)

fig.suptitle("GNN: AUC, Efficiency and Purity", fontsize=14, fontweight='bold')
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "gnn_metrics.png")
plt.savefig(out, dpi=DPI)
plt.close()
print(f"Saved: {out}")
print(f"  Best AUC={gnn_val['auc'].max():.4f} (epoch {gnn_val.loc[gnn_val['auc'].idxmax(),'epoch']})")
print(f"  Best eff={gnn_val['eff'].max():.4f} (epoch {gnn_val.loc[gnn_val['eff'].idxmax(),'epoch']})")
print(f"  Best pur={gnn_val['pur'].max():.4f} (epoch {gnn_val.loc[gnn_val['pur'].idxmax(),'epoch']})")
print(f"  Final AUC={gnn_val['auc'].iloc[-1]:.4f}, eff={gnn_val['eff'].iloc[-1]:.4f}, pur={gnn_val['pur'].iloc[-1]:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 5: comparison_versions.png
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Figure 5: comparison_versions.png ===")

# Collect stats per version
versions = [0, 1, 2]
ml_final_val_loss = []
ml_best_eff       = []
gnn_best_auc      = []

for v in versions:
    ml_v, _  = load_metrics(os.path.join(ML_BASE,  f"version_{v}/metrics.csv"))
    gnn_v, _ = load_metrics(os.path.join(GNN_BASE, f"version_{v}/metrics.csv"))
    ml_final_val_loss.append(ml_v["val_loss"].iloc[-1])
    ml_best_eff.append(ml_v["eff"].max())
    gnn_best_auc.append(gnn_v["auc"].max())

print(f"  ML final val_loss: {ml_final_val_loss}")
print(f"  ML best eff:       {ml_best_eff}")
print(f"  GNN best AUC:      {gnn_best_auc}")

x = np.arange(len(versions))
width = 0.25
labels = [f"v{v}" for v in versions]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# ML final val_loss (lower is better)
bars0 = axes[0].bar(x, ml_final_val_loss, width=0.5, color=["#4C72B0","#DD8452","#55A868"])
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, fontsize=12)
axes[0].set_ylabel("Final Val Loss", fontsize=11)
axes[0].set_title("ML: Final Val Loss\n(lower is better)", fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars0, ml_final_val_loss):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00005,
                 f"{val:.4f}", ha='center', va='bottom', fontsize=9)

# ML best efficiency
bars1 = axes[1].bar(x, ml_best_eff, width=0.5, color=["#4C72B0","#DD8452","#55A868"])
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, fontsize=12)
axes[1].set_ylabel("Best Efficiency", fontsize=11)
axes[1].set_title("ML: Best Efficiency\n(higher is better)", fontsize=11)
axes[1].set_ylim(0.80, 0.92)
axes[1].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars1, ml_best_eff):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                 f"{val:.4f}", ha='center', va='bottom', fontsize=9)

# GNN best AUC
bars2 = axes[2].bar(x, gnn_best_auc, width=0.5, color=["#4C72B0","#DD8452","#55A868"])
axes[2].set_xticks(x)
axes[2].set_xticklabels(labels, fontsize=12)
axes[2].set_ylabel("Best AUC", fontsize=11)
axes[2].set_title("GNN: Best AUC\n(higher is better)", fontsize=11)
axes[2].set_ylim(0.70, 1.01)
axes[2].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars2, gnn_best_auc):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f"{val:.4f}", ha='center', va='bottom', fontsize=9)

fig.suptitle("Training Progress Across Versions", fontsize=14, fontweight='bold')
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "comparison_versions.png")
plt.savefig(out, dpi=DPI)
plt.close()
print(f"Saved: {out}")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 6: edge_performance.png
# GNN summary: val_loss + AUC + eff + pur over epochs (v2)
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Figure 6: edge_performance.png ===")

fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# Top-left: val_loss
ax_loss = fig.add_subplot(gs[0, 0])
ax_loss.plot(gnn_val["epoch"], gnn_val["val_loss"], color="blue", linewidth=1.5)
ax_loss.plot(gnn_train["epoch"], gnn_train["train_loss"], color="orange", linewidth=1.5, linestyle='--')
ax_loss.set_xlabel("Epoch", fontsize=11)
ax_loss.set_ylabel("Loss", fontsize=11)
ax_loss.set_title("GNN Loss (v2)", fontsize=11)
ax_loss.legend(["Val Loss", "Train Loss"], fontsize=9)
ax_loss.grid(True, alpha=0.3)

# Top-right: AUC
ax_auc = fig.add_subplot(gs[0, 1])
ax_auc.plot(gnn_val["epoch"], gnn_val["auc"], color="purple", linewidth=2)
best_auc_ep = int(gnn_val.loc[gnn_val["auc"].idxmax(), "epoch"])
best_auc_val = gnn_val["auc"].max()
ax_auc.axhline(best_auc_val, color='purple', linestyle=':', alpha=0.5)
ax_auc.annotate(f"Best: {best_auc_val:.4f}\n(ep {best_auc_ep})",
                xy=(best_auc_ep, best_auc_val),
                xytext=(best_auc_ep - 5, best_auc_val - 0.015),
                fontsize=8, color='purple',
                arrowprops=dict(arrowstyle='->', color='purple', lw=1))
ax_auc.set_xlabel("Epoch", fontsize=11)
ax_auc.set_ylabel("AUC", fontsize=11)
ax_auc.set_title("GNN AUC (v2)", fontsize=11)
ax_auc.grid(True, alpha=0.3)

# Bottom-left: efficiency
ax_eff = fig.add_subplot(gs[1, 0])
ax_eff.plot(gnn_val["epoch"], gnn_val["eff"], color="steelblue", linewidth=1.5)
ax_eff.set_xlabel("Epoch", fontsize=11)
ax_eff.set_ylabel("Efficiency", fontsize=11)
ax_eff.set_title("GNN Efficiency (v2)", fontsize=11)
ax_eff.grid(True, alpha=0.3)

# Bottom-right: purity
ax_pur = fig.add_subplot(gs[1, 1])
ax_pur.plot(gnn_val["epoch"], gnn_val["pur"], color="darkorange", linewidth=1.5)
ax_pur.set_xlabel("Epoch", fontsize=11)
ax_pur.set_ylabel("Purity", fontsize=11)
ax_pur.set_title("GNN Purity (v2)", fontsize=11)
ax_pur.grid(True, alpha=0.3)

fig.suptitle(f"GNN Edge Classification Performance (Version 2)\nBest AUC = {best_auc_val:.4f}",
             fontsize=13, fontweight='bold')
out = os.path.join(OUTPUT_DIR, "edge_performance.png")
plt.savefig(out, dpi=DPI, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")
print(f"  Best AUC={best_auc_val:.4f} at epoch {best_auc_ep}")
print(f"  Final val_loss={gnn_val['val_loss'].iloc[-1]:.5f}")
print(f"  Final eff={gnn_val['eff'].iloc[-1]:.4f}, Final pur={gnn_val['pur'].iloc[-1]:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 7: neighbor_performance.png
# ML summary: val_loss + eff + pur over epochs (v2) with annotations
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Figure 7: neighbor_performance.png ===")

fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# Top-left: val_loss
ax_loss2 = fig.add_subplot(gs[0, 0])
ax_loss2.plot(ml_train["epoch"], ml_train["train_loss"], color="orange", linewidth=1.5, linestyle='--', label="Train Loss")
ax_loss2.plot(ml_val["epoch"],   ml_val["val_loss"],     color="blue",   linewidth=1.5, label="Val Loss")
ax_loss2.set_xlabel("Epoch", fontsize=11)
ax_loss2.set_ylabel("Loss", fontsize=11)
ax_loss2.set_title("ML Loss (v2)", fontsize=11)
ax_loss2.legend(fontsize=9)
ax_loss2.grid(True, alpha=0.3)

# Top-right: efficiency
ax_eff2 = fig.add_subplot(gs[0, 1])
ax_eff2.plot(ml_val["epoch"], ml_val["eff"], color="steelblue", linewidth=1.5)
best_eff_ep = int(ml_val.loc[ml_val["eff"].idxmax(), "epoch"])
best_eff_val = ml_val["eff"].max()
ax_eff2.axhline(best_eff_val, color='steelblue', linestyle=':', alpha=0.5)
ax_eff2.annotate(f"Best: {best_eff_val:.4f}\n(ep {best_eff_ep})",
                 xy=(best_eff_ep, best_eff_val),
                 xytext=(best_eff_ep + 3, best_eff_val - 0.002),
                 fontsize=8, color='steelblue',
                 arrowprops=dict(arrowstyle='->', color='steelblue', lw=1))
ax_eff2.set_xlabel("Epoch", fontsize=11)
ax_eff2.set_ylabel("Efficiency", fontsize=11)
ax_eff2.set_title("ML Efficiency (v2)", fontsize=11)
ax_eff2.grid(True, alpha=0.3)

# Bottom-left: purity
ax_pur2 = fig.add_subplot(gs[1, 0])
ax_pur2.plot(ml_val["epoch"], ml_val["pur"], color="darkorange", linewidth=1.5)
best_pur_ep = int(ml_val.loc[ml_val["pur"].idxmax(), "epoch"])
best_pur_val = ml_val["pur"].max()
ax_pur2.annotate(f"Best: {best_pur_val:.4f}\n(ep {best_pur_ep})",
                 xy=(best_pur_ep, best_pur_val),
                 xytext=(max(0, best_pur_ep - 20), best_pur_val - 0.0003),
                 fontsize=8, color='darkorange',
                 arrowprops=dict(arrowstyle='->', color='darkorange', lw=1))
ax_pur2.set_xlabel("Epoch", fontsize=11)
ax_pur2.set_ylabel("Purity", fontsize=11)
ax_pur2.set_title("ML Purity (v2)", fontsize=11)
ax_pur2.grid(True, alpha=0.3)

# Bottom-right: eff vs pur scatter (color by epoch)
ax_scatter = fig.add_subplot(gs[1, 1])
sc = ax_scatter.scatter(ml_val["eff"], ml_val["pur"],
                        c=ml_val["epoch"], cmap='viridis', s=15, alpha=0.8)
cbar = plt.colorbar(sc, ax=ax_scatter)
cbar.set_label("Epoch", fontsize=9)
ax_scatter.set_xlabel("Efficiency", fontsize=11)
ax_scatter.set_ylabel("Purity", fontsize=11)
ax_scatter.set_title("ML Efficiency vs Purity\n(color = epoch)", fontsize=11)
ax_scatter.grid(True, alpha=0.3)

fig.suptitle(
    f"Metric Learning Embedding Performance (Version 2)\n"
    f"Best Eff={best_eff_val:.4f}, Best Pur={best_pur_val:.4f}",
    fontsize=13, fontweight='bold'
)
out = os.path.join(OUTPUT_DIR, "neighbor_performance.png")
plt.savefig(out, dpi=DPI, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")
print(f"  Best eff={best_eff_val:.4f} at epoch {best_eff_ep}")
print(f"  Best pur={best_pur_val:.4f} at epoch {best_pur_ep}")
print(f"  Final val_loss={ml_val['val_loss'].iloc[-1]:.6f}")

print("\n=== All figures saved to", OUTPUT_DIR, "===")
