# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/blob/master/Examples/TrackML_Quickstart/DM_colab_quickstart.ipynb)

# %% [markdown]
# # TrackML Quickstart

# %% [markdown]
# ## Install Libraries

# %% [markdown]
# **Note: Before running notebook, ensure your runtime is set to GPU**
# 
# First, we just install a few libraries (this should take around 5 minutes and automatically restart the kernel), and load in the repository.

# %%
!pip install -q condacolab
import condacolab
condacolab.install()

# %%
!pip install seaborn bokeh 
!conda install pandas scipy matplotlib cupy "cudatoolkit>=11.3" "pytorch>=1.10.2" "pytorch-lightning>=1.6" pyg faiss-gpu -c pytorch -c pyg -c conda-forge

# %%
!git clone https://github.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX.git
%cd Tracking-ML-Exa.TrkX/Examples/TrackML_Quickstart

# %% [markdown]
# # Import libraries

# %%
import sys, os
sys.path.append("../../")
from Scripts import train_metric_learning, run_metric_learning_inference, train_gnn, run_gnn_inference, build_track_candidates, evaluate_candidates
from Scripts.utils.convenience_utils import get_example_data, plot_true_graph, get_training_metrics, plot_training_metrics, plot_neighbor_performance, plot_predicted_graph, plot_track_lengths, plot_edge_performance, plot_graph_sizes
import yaml

import warnings
warnings.filterwarnings("ignore")
CONFIG = 'pipeline_config.yaml'

# %% [markdown]
# ## Download Data

# %%
%%capture
!mkdir datasets
!wget https://portal.nersc.gov/cfs/m3443/dtmurnane/TrackML_Example/trackml_quickstart_dataset.tar.gz -O datasets/trackml_quickstart_dataset.tar.gz

# %%
%%capture
!tar -xvf datasets/trackml_quickstart_dataset.tar.gz -C datasets;
!rm datasets/trackml_quickstart_dataset.tar.gz

# %% [markdown]
# ## TrackML Dataset
# 
# The TrackML dataset contains simulated indepedent proton-proton collision events, each generating hundreds of particles, each of which hits cells and layers of the detector layers multiple times. The detector records the spatial coordinates and other auxillary information of these hits which, if properly connected, form tracks associated with the parent particle and the collision event from which it originates. The challenge and goal of this project is to associate each and every hit to one single track with optimal purity and efficiency, whose precise definition will be given later.

# %% [markdown]
# Each entry in the particles data frame contains a unique identifier of the particle (particle_id), its charge (q), its initial position or vertex $(v_x, v_y, v_z)$, its initial momentum in GeV/c $(p_x, p_y, p_z)$ and its associated number of detector hits. 
# 
# Many particles do not leave behind any detector hits and obviously cannot be associated to any track. This is called "detector inefficiency". They are among "uninterested particles" and will be mostly filtered out by a simple momentum cut.

# %% [markdown]
# ### Training data
# Let us take a look at the data before training. In this example pipeline, we have preprocessed the TrackML data into a more convenient form. We calculated directional information and summary statistics from the charge deposited in each spacepoints, and append them to its cyclidrical coordinates. Let us load an example data file and inspect the content.

# %%
with open(CONFIG, 'r') as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)

# %%
example_data_df, example_data_pyg = get_example_data(configs)
example_data_df.head()

# %% [markdown]
# ### Visualize tracks

# %% [markdown]
# A "true track" is defined as a set of sequential hits, all left by the same particle. Therefore a true edge is the edge formed by two sequential hits. Let's visualize a random set of 200 true tracks:

# %%
plot_true_graph(example_data_pyg, num_tracks=200)

# %% [markdown]
# # 1. Train Metric Learning

# %% [markdown]
# ## Train metric learning model
# 
# Finally we come to model training. By default, we train the MLP for 30 epochs, which takes approximately 15 minutes on an NVidia V100. Feel free to adjust the epoch number in pipeline_config.yml

# %%
metric_learning_trainer, metric_learning_model = train_metric_learning(CONFIG)

# %% [markdown]
# ## Plot training metrics

# %% [markdown]
# We can examine how the training went. This is stored in a simple dataframe:

# %%
embedding_metrics = get_training_metrics(metric_learning_trainer)
embedding_metrics.head()

# %%
plot_training_metrics(embedding_metrics)

# %% [markdown]
# ## Evaluate model performance on sample test data
# 
# Here we evaluate the model performace on one sample test data. We look at how the efficiency and purity change with the embedding radius.

# %%
plot_neighbor_performance(metric_learning_model)

# %% [markdown]
# ## Plot example truth and predicted graphs

# %%
plot_predicted_graph(metric_learning_model)

# %% [markdown]
# ## Track lengths

# %%
plot_track_lengths(metric_learning_model)

# %%
plot_graph_sizes(metric_learning_model)

# %% [markdown]
# # 2. Construct graphs from metric learning inference
# 
# This step performs model inference on the entire input datasets (train, validation and test), to obtain input graphs to the graph neural network. Optionally, we also clear the directory.

# %%
graph_builder = run_metric_learning_inference(CONFIG)

# %%
import os, gc, torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print(f"Cleared CUDA cache on {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available")

print("Run this cell right before Step 3. If allocator behavior is unchanged, restart kernel once.")


# %% [markdown]
# # 3. Train graph neural networks

# %% [markdown]
# We have a set of graphs constructed. We now train a GNN to classify edges as either "true" (belonging to the same track) or "false" (not belonging to the same track).

# %%
gnn_trainer, gnn_model = train_gnn(CONFIG)

# %% [markdown]
# ## Plot training metrics

# %%
gnn_metrics = get_training_metrics(gnn_trainer)
gnn_metrics.head()

# %%
plot_training_metrics(gnn_metrics)

# %% [markdown]
# ## Evaluate model performance on sample test data
# 
# Here we evaluate the model performace on one sample test data. We look at how the efficiency and purity change with the embedding radius.

# %%
plot_edge_performance(gnn_model)

# %% [markdown]
# # Step 4: GNN inference 

# %%
run_gnn_inference(CONFIG)

# %% [markdown]
# # Step 5: Build track candidates from GNN

# %%
build_track_candidates(CONFIG)

# %% [markdown]
# # Step 6: Evaluate track candidates

# %% [markdown]
# We can control the matching style in the pipeline config file. The following all require at least a majority of hits to match in each scheme (i.e. matching fraction = 50%).
# A discussion of each matching style and some worked examples can be found in the [Documentation](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/performance/matching_definitions/).

# %% [markdown]
# ATLAS style matching is the default.

# %%
evaluated_events, reconstructed_particles, particles, matched_tracks, tracks = evaluate_candidates(CONFIG)

# %%
# Optional: save selected plots from this notebook
import os
from pathlib import Path

import matplotlib.pyplot as plt
from bokeh.io import export_png

SAVE_PLOTS = True  # Set True when you want to export
OUTPUT_DIR = Path("saved_plots")
MATPLOTLIB_DPI = 500
BOKEH_SCALE_FACTOR = 5

# Choose what to save
TO_SAVE = [
    "plot_true_graph",
    "embedding_training",
    "neighbor_performance",
    "predicted_graph",
    "track_lengths",
    "graph_sizes",
    "gnn_training",
    "edge_performance",
]

plot_registry = {
    "plot_true_graph": locals().get("plot_true_graph_fig"),
    "embedding_training": locals().get("embedding_training_figs"),
    "neighbor_performance": locals().get("neighbor_perf_figs"),
    "predicted_graph": locals().get("predicted_graph_figs"),
    "track_lengths": locals().get("track_length_figs"),
    "graph_sizes": locals().get("graph_sizes_fig"),
    "gnn_training": locals().get("gnn_training_figs"),
    "edge_performance": locals().get("edge_perf_figs"),
}

def _bokeh_figures(obj):
    if obj is None:
        return []
    if isinstance(obj, dict) and "figures" in obj:
        return obj["figures"]
    return [obj]

def _save_matplotlib(fig, outpath):
    fig.savefig(outpath, dpi=MATPLOTLIB_DPI, bbox_inches="tight")

def _save_bokeh(fig, outpath):
    # Requires selenium + browser driver for PNG export.
    export_png(fig, filename=str(outpath), scale_factor=BOKEH_SCALE_FACTOR)

if SAVE_PLOTS:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved, skipped, failed = [], [], []

    for name in TO_SAVE:
        obj = plot_registry.get(name)
        if obj is None:
            skipped.append((name, "plot object not found (run the source plotting cell first)"))
            continue

        if name == "graph_sizes":
            try:
                out = OUTPUT_DIR / f"{name}.png"
                _save_matplotlib(obj, out)
                saved.append(str(out))
            except Exception as exc:
                failed.append((name, str(exc)))
            continue

        figs = _bokeh_figures(obj)
        for idx, fig in enumerate(figs, start=1):
            suffix = f"_{idx}" if len(figs) > 1 else ""
            out = OUTPUT_DIR / f"{name}{suffix}.png"
            try:
                _save_bokeh(fig, out)
                saved.append(str(out))
            except Exception as exc:
                failed.append((f"{name}{suffix}", str(exc)))

    print(f"Saved {len(saved)} plot files")
    if saved:
        print("\n".join(saved))
    if skipped:
        print("\nSkipped:")
        for n, msg in skipped:
            print(f"- {n}: {msg}")
    if failed:
        print("\nFailed:")
        for n, msg in failed:
            print(f"- {n}: {msg}")
else:
    print("Set SAVE_PLOTS = True and rerun this cell to export selected plots.")

# %%
print(plot_true_graph_fig if "plot_true_graph_fig" in locals() else "missing")
print(embedding_training_figs if "embedding_training_figs" in locals() else "missing")



