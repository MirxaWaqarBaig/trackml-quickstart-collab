import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def _resolve_gnn_dir(gnn_dir):
    gnn_dir = Path(gnn_dir)

    candidates = [
        gnn_dir,
        Path.cwd() / gnn_dir,
        Path.cwd() / "datasets" / "quickstart_gnn_processed",
        Path.cwd() / "Examples/TrackML_Quickstart/datasets/quickstart_gnn_processed",
        Path.cwd().parent / "datasets" / "quickstart_gnn_processed",
        Path.cwd().parent / "Examples/TrackML_Quickstart/datasets/quickstart_gnn_processed",
    ]

    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(
        "Could not find quickstart_gnn_processed. Tried:\n" +
        "\n".join(str(c) for c in candidates)
    )


def plot_gnn_event(
    gnn_dir="datasets/quickstart_gnn_processed",
    score_cut=0.75,
    event_idx=0,
    max_edges=300,
    save=False
):
    gnn_dir = _resolve_gnn_dir(gnn_dir)

    files = []
    for sub in ["testset", "test", "valset", "val", "trainset", "train"]:
        p = gnn_dir / sub
        if p.exists():
            files = sorted([f for f in p.glob("*") if f.is_file()])
            if files:
                print(f"Using GNN files from: {p}")
                break

    if not files:
        raise FileNotFoundError(
            f"No GNN graph files found inside {gnn_dir}."
        )

    data = torch.load(files[event_idx], map_location="cpu", weights_only=False)

    if not hasattr(data, "scores"):
        raise AttributeError(
            "data.scores not found. Run GNN inference first."
        )

    x_feat = data.x.detach().cpu()

    r = x_feat[:, 0].numpy() * 1000.0
    phi = x_feat[:, 1].numpy()

    x_h = r * np.cos(phi)
    y_h = r * np.sin(phi)

    edges = data.edge_index.detach().cpu().numpy()
    scores = data.scores.detach().cpu().numpy()
    y_true = data.y.detach().cpu().bool().numpy()

    kept = scores >= score_cut

    tp = kept & y_true
    fp = kept & ~y_true
    fn = ~kept & y_true
    tn = ~kept & ~y_true

    pid = data.pid.detach().cpu().numpy() if hasattr(data, "pid") else np.zeros(len(x_h))
    upid = np.unique(pid[pid > 0])

    cmap = plt.cm.tab10
    pid_col = {p: cmap(i % 10) for i, p in enumerate(upid)}
    hit_col = [pid_col.get(pid[i], (0.75, 0.75, 0.75, 1)) for i in range(len(x_h))]

    # 2x2 layout: better inside notebook
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), dpi=120, facecolor="white")
    axes = axes.flatten()

    fig.suptitle(
        f"GNN edge filtering, score cut = {score_cut}\n"
        f"TP={tp.sum()} | FP={fp.sum()} | FN={fn.sum()} | TN={tn.sum()}",
        fontsize=12,
        y=0.98
    )

    panels = [
        (axes[0], tp, "#1565C0", f"Kept true edges, TP = {tp.sum()}"),
        (axes[1], fp, "#C62828", f"Kept wrong edges, FP = {fp.sum()}"),
        (axes[2], fn, "#E65100", f"Removed true edges, FN = {fn.sum()}"),
        (axes[3], tn, "#2E7D32", f"Removed wrong edges, TN = {tn.sum()}"),
    ]

    for ax, mask, col, title in panels:
        ax.set_facecolor("white")

        for R in [32, 72, 116, 172, 260, 360, 500, 660, 820, 1020]:
            ax.add_patch(
                plt.Circle(
                    (0, 0),
                    R,
                    fill=False,
                    color="#DDDDDD",
                    linewidth=0.5,
                    linestyle="--"
                )
            )

        src = edges[0, mask]
        dst = edges[1, mask]

        for s, d in zip(src[:max_edges], dst[:max_edges]):
            ax.plot(
                [x_h[s], x_h[d]],
                [y_h[s], y_h[d]],
                color=col,
                alpha=0.45,
                linewidth=0.8,
                zorder=1
            )

        ax.scatter(
            x_h,
            y_h,
            c=hit_col,
            s=8,
            zorder=2,
            linewidths=0
        )

        ax.set_xlim(-1100, 1100)
        ax.set_ylim(-1100, 1100)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.grid(alpha=0.15)

    plt.tight_layout()

    if save:
        plt.savefig(
            "gnn_graph.png",
            dpi=150,
            facecolor="white",
            bbox_inches="tight"
        )
        print("Saved: gnn_graph.png")

    plt.show()

    print("Plotted file:", files[event_idx])
