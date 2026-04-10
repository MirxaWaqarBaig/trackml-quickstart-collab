"""
Build quirk+SM mixed feature-store events for the TrackML quickstart workflow.
Requires real TrackML detector geometry and TrackML event CSV files.
"""

import argparse
import os
from pathlib import Path
import sys

import yaml


def parse_args():
    parser = argparse.ArgumentParser("Step_0_Build_Quirk_Feature_Store.py")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to quirk processing YAML config. Defaults to prepare_quickstart_quirk.yaml in Processing module.",
    )
    parser.add_argument(
        "--n-files",
        type=int,
        default=None,
        help="Override number of quirk events to generate.",
    )
    parser.add_argument(
        "--detector-path",
        default=None,
        help="Override path to TrackML detectors.csv.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Override TrackML input directory containing '*-hits.csv' files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    script_path = Path(__file__).resolve()
    quickstart_dir = script_path.parents[1]
    repo_root = script_path.parents[3]
    sys.path.append(str(repo_root))

    default_cfg = (
        repo_root
        / "Pipelines"
        / "TrackML_Example"
        / "LightningModules"
        / "Processing"
        / "prepare_quickstart_quirk.yaml"
    )
    cfg_path = Path(args.config).resolve() if args.config else default_cfg

    with open(cfg_path) as handle:
        hparams = yaml.load(handle, Loader=yaml.FullLoader)

    # Force output into this quickstart copy so notebook can consume directly.
    hparams["output_dir"] = str(quickstart_dir / "datasets" / "quickstart_quirk_example")
    if args.n_files is not None:
        hparams["n_files"] = int(args.n_files)
    if args.detector_path is not None:
        hparams["detector_path"] = str(Path(args.detector_path).resolve())
    if args.input_dir is not None:
        hparams["input_dir"] = str(Path(args.input_dir).resolve())

    os.makedirs(hparams["output_dir"], exist_ok=True)

    # Local import after path setup to avoid environment-side import ordering issues.
    from Pipelines.TrackML_Example.LightningModules.Processing.Models.feature_construction import (
        TrackMLFeatureStore,
    )

    print(f"Building quirk feature store with config: {cfg_path}")
    print(f"Output directory: {hparams['output_dir']}")
    print(f"Number of events: {hparams['n_files']}")
    print(f"Detector path: {hparams.get('detector_path', '')}")
    print(f"TrackML input dir: {hparams.get('input_dir', '')}")

    store = TrackMLFeatureStore(hparams)
    store.prepare_data()

    print("Done. Quirk feature store generation complete.")


if __name__ == "__main__":
    main()
