# System imports
import sys
import os
import multiprocessing as mp
from functools import partial

# 3rd party imports
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from torch.nn import Linear
import torch.nn as nn
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

# Local imports
from ..feature_store_base import FeatureStoreBase
from ..utils.event_utils import prepare_event, prepare_quirk_event
from ..utils.detector_utils import load_detector


class TrackMLFeatureStore(FeatureStoreBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.detector_path = self.hparams.get("detector_path", "")
        self.dataset_mode = self.hparams.get("dataset_mode", "trackml").lower()

    def prepare_data(self):
        if self.dataset_mode == "quirk":
            all_events = np.arange(self.n_files, dtype=np.int64)
        else:
            # Find the input files
            all_files = os.listdir(self.input_dir)
            all_events = sorted(
                np.unique([os.path.join(self.input_dir, event[:14]) for event in all_files])
            )[: self.n_files]

        # Split the input files by number of tasks and select my chunk only
        all_events = np.array_split(all_events, self.n_tasks)[self.task]

        # Define the cell features to be added to the dataset

        cell_features = [
            "cell_count",
            "cell_val",
            "leta",
            "lphi",
            "lx",
            "ly",
            "lz",
            "geta",
            "gphi",
        ]
        if self.dataset_mode == "quirk":
            if not self.detector_path or not os.path.exists(self.detector_path):
                raise FileNotFoundError(
                    "dataset_mode=quirk requires a valid detector_path to TrackML detectors.csv."
                )
            if not self.input_dir or not os.path.exists(self.input_dir):
                raise FileNotFoundError(
                    "dataset_mode=quirk requires input_dir with TrackML '*-hits.csv' files for SM background and module centers."
                )

        detector_orig, detector_proc = load_detector(self.detector_path)

        # Prepare output
        # output_dir = os.path.expandvars(self.output_dir) FIGURE OUT HOW TO USE THIS!
        os.makedirs(self.output_dir, exist_ok=True)
        print("Writing outputs to " + self.output_dir)

        # Process input files with a worker pool and progress bar
        prepare_fn = prepare_quirk_event if self.dataset_mode == "quirk" else prepare_event
        process_func = partial(
            prepare_fn,
            detector_orig=detector_orig,
            detector_proc=detector_proc,
            cell_features=cell_features,
            **self.hparams
        )
        process_map(process_func, all_events, max_workers=self.n_workers)
