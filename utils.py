from argparse import Namespace
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typing import List, Optional

import glob
import torch
import numpy as np
import pandas as pd
import re

TAG_REGEX = r".*\/evaluation_(\w+)\/.*"
EXPERIMENT_REGEX = r".*\/(.+=.+)\/.*"


def get_experiment_name(args: Namespace) -> str:
    """Returns a name for the experiment based on the command line arguments."""
    experiment_name = f"agg={args.aggregation}_probe={args.probe_layer}"
    if args.task == "DEP":
        experiment_name += f"_concat-mode={args.concat_mode}"
    return experiment_name


def get_xneutr_df(experiment_path: str) -> pd.DataFrame:
    """Returns a dataframe with the target names as columns and the neutralizer names as index."""
    neutralizers = {}
    for run_path in glob.glob(experiment_path):
        # Extract the neutralizer name from the path
        res = re.search(TAG_REGEX, run_path)
        neutr_tag = res.group(1).upper()

        # Get the data for the run from Tensorboard
        ea = EventAccumulator(run_path)
        ea.Reload()

        scalars = {}
        for metric in ea.Tags()["scalars"]:
            if "test" not in metric:
                # Only interested in evaluation runs
                continue

            # Extract the target name from the metric
            if "/" not in metric:
                target_tag = "avg"
            else:
                target_tag = metric.split("_")[-1].upper()

            # Save the corresponding metric value for the target
            scalars[target_tag] = ea.Scalars(metric)[0].value

        # Add the metrics to the neutralizer's entry
        neutralizers[neutr_tag] = scalars

    # Create dataframe and name the columns and indices
    df = pd.DataFrame(neutralizers)
    df.index.name = "Target"
    df = df.T
    df.index.name = "Neutralizer"
    return df


def get_baseline_series(experiment_path: str) -> pd.Series:
    """Returns a series with the baseline metrics for the given experiment."""
    if "*" in experiment_path:
        # Remove wildstar from string by taking the first substitution
        experiment_path = glob.glob(experiment_path)[0]

    # Get the data for the run from Tensorboard
    ea = EventAccumulator(experiment_path)
    ea.Reload()

    scalars = {}
    for metric in ea.Tags()["scalars"]:
        if "test" not in metric:
            # Only interested in evaluation runs
            continue

        # Extract the target name from the metric
        parts = metric.split("_")
        if len(parts) == 2:
            target_tag = "avg"
        else:
            target_tag = parts[-1].upper()

        # Save the corresponding metric value for the target
        scalars[target_tag] = ea.Scalars(metric)[0].value

    return pd.Series(scalars)


def get_acc_drop(eval_path: str, keep_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Returns a dataframe with the relative accuracy drop with cross-neutralizing.

    Args:
        eval_path (str): the path to the evaluation directory
        keep_cols (List[str]): the columns to keep in the dataframe

    Returns:
        a dataframe with the target names as columns and the neutralizer names as index
    """
    # Get the baseline and cross-neutralizing results
    base_series = get_baseline_series(f"{eval_path}/events*")
    xn_df = get_xneutr_df(f"{eval_path}_*/events*")

    # Filter the columns that don't appear in the baseline
    nulls = base_series.isnull()
    base_series = base_series[~nulls]
    xn_df = xn_df.T[~nulls].T

    # Calculate relative accuracy drop
    acc_drop = (xn_df - base_series) / base_series
    acc_drop.sort_index(axis=0, inplace=True)
    acc_drop.sort_index(axis=1, inplace=True)

    # Filter the columns that don't appear in the index and vice versa
    drop_indices = set(acc_drop.T.columns).difference(acc_drop.columns)
    drop_columns = set(acc_drop.columns).difference(acc_drop.T.columns).difference(["avg"])

    # Explicitly add more indices/columns to hide
    for hidden in ("DEP", "APPOS"):
        if hidden in acc_drop.T.columns:
            drop_indices.add(hidden)
        if hidden in acc_drop.columns:
            drop_columns.add(hidden)

    # Drop the indices/columns specified above
    acc_drop.drop(index=drop_indices, inplace=True)
    acc_drop.drop(columns=drop_columns, inplace=True)

    if keep_cols is not None:
        # Apply further filtering if specified
        acc_drop = (acc_drop.loc[keep_cols])[keep_cols]

    return acc_drop


def get_experiments_df(
    task: str,
    treebank: str,
    model: str,
    logdir: str = "lightning_logs",
) -> pd.DataFrame:
    """Returns a dataframe with all experiments for the given task, treebank and model."""
    experiments = {}
    for path in glob.glob(f"{logdir}/{model}/{treebank}/{task}/*/evaluation"):
        # Extract the experiment name from the path
        res = re.search(EXPERIMENT_REGEX, path)
        experiment = res.group(1)

        # Get the baseline and cross-neutralizing results
        baseline = get_baseline_series(f"{path}/events*")
        acc_drop = get_acc_drop(path)

        self_neutr = {}
        # Save the self-neutralizing results for each tag
        for tag in acc_drop.columns.values[:-1]:
            tag = tag.upper()
            self_neutr[tag] = acc_drop[tag][tag]

        # Calculate the average self-neutralizing change
        self_neutr["avg"] = np.nanmean(list(self_neutr.values()))
        # Save the baseline accuracy
        self_neutr["baseline"] = baseline["avg"]
        # Save the results for the experiment
        experiments[experiment] = self_neutr

    # Create the dataframe and sort by average change
    df = pd.DataFrame(experiments)
    df.sort_values(by="avg", axis=1, inplace=True)

    # Filter out indices with no results
    nulls = df.isnull().all(1)
    df = df[~nulls]

    # Add names to the columns and indices
    df.index.name = "Neutralizer"
    df = df.T
    df.index.name = "Experiment"
    return df


def select_best_mode(experiments_df: pd.DataFrame) -> str:
    """Selects the best configuration from the experiments by taking the top quartile by
    baseline accuracy and then the top-1 by self-neutralizing relative accuracy drop."""
    top_quartile = experiments_df["baseline"].nlargest(len(experiments_df) // 4)
    config = experiments_df.loc[top_quartile.index]["avg"].nsmallest(1).index[0]

    return config


def extract_centroids(ckpt_file: str, centroids_file: str, no_gpu: bool = False):
    """
    Extracts the dictionary of centroids from a specified model checkpoint and saves them
    to a new .pt file.

    Args:
        ckpt_file (str): The checkpoint of the model to extract the centroids from.
        centroids_file (str): The path to the file to save the centroids at.
        no_gpu (bool): Whether to load the checkpoint file on CPU.
    """
    device = "cpu" if no_gpu or (not torch.cuda.is_available()) else "cuda"
    c = torch.load(ckpt_file, map_location=device)

    centroids = c["class_centroids"]
    classes = c["hyper_parameters"]["class_map"]
    centroids = {classes[k]: v for k, v in centroids.items()}
    torch.save(centroids, centroids_file)
