from argparse import Namespace
from typing import List, Optional
import urllib.request
import glob
import os
import zipfile
import re

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import pandas as pd
from tqdm import tqdm

TAG_REGEX = r".*\/evaluation_(\w+)\/.*"
EXPERIMENT_REGEX = r".*\/(.+=.+)\/.*"


def get_experiment_name(args: Namespace) -> str:
    """Returns a name for the experiment based on the command line arguments."""
    experiment_name = f"agg={args.aggregation}_probe={args.probe_layer}"
    if args.task == "DEP":
        experiment_name += f"_concat-mode={args.concat_mode}"
    return experiment_name


def get_xneutr_df(
    experiment_path: str, suffix_filter: Optional[str] = None, metric_name: str = "acc"
) -> pd.DataFrame:
    """Returns a dataframe with the target names as columns and the neutralizer names as index."""
    neutralizers = {}
    if suffix_filter is not None:
        pattern = re.compile(suffix_filter)
    for run_path in glob.glob(experiment_path):
        # Extract the neutralizer name from the path
        res = re.search(TAG_REGEX, run_path)
        neutr_tag = res.group(1)

        if suffix_filter is not None:
            # skip this neutralising tag
            if not pattern.match(neutr_tag):
                continue

        neutr_tag = neutr_tag.upper()
        

        # Get the data for the run from Tensorboard
        ea = EventAccumulator(run_path)
        ea.Reload()

        scalars = {}
        for metric in ea.Tags()["scalars"]:
            if "test" not in metric:
                # Only interested in evaluation runs
                continue

            if metric_name not in metric:
                # Only interested in requested metric
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


def get_baseline_series(experiment_path: str, metric_name: str = "acc") -> pd.Series:
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
        if metric_name not in metric:
            # Only interested in requested metric
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


def compute_perf_change(
    base_series, xn_df, keep_cols, suffix_filter, change: str = "rel"
):
    # Filter the columns that don't appear in the baseline
    nulls = base_series.isnull()
    base_series = base_series[~nulls]
    xn_df = xn_df.T[~nulls].T

    # Calculate relative accuracy drop
    if change == "rel":
        acc_drop = (xn_df - base_series) / base_series
    elif change == "abs":
        acc_drop = xn_df - base_series
    else:
        raise ValueError(f"Unknown change type {change}")
    acc_drop.sort_index(axis=0, inplace=True)
    acc_drop.sort_index(axis=1, inplace=True)

    # we don't do this for cross-task cross neutralisation
    if suffix_filter is None:
        # Filter the columns that don't appear in the index and vice versa
        drop_indices = set(acc_drop.T.columns).difference(acc_drop.columns)
        drop_columns = (
            set(acc_drop.columns).difference(acc_drop.T.columns).difference(["avg"])
        )
    else:
        drop_indices = set()
        drop_columns = set()

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


def get_acc_drop(
    eval_path: str,
    keep_cols: Optional[List[str]] = None,
    suffix_filter=None,
    metric_name: str = "acc",
    change="rel",
) -> pd.DataFrame:
    """Returns a dataframe with the performance drop with cross-neutralizing.

    Args:
        eval_path (str): the path to the evaluation directory
        keep_cols (List[str]): the columns to keep in the dataframe

    Returns:
        a dataframe with the target names as columns and the neutralizer names as index
    """
    # Get the baseline and cross-neutralizing results
    base_series = get_baseline_series(f"{eval_path}/events*")
    xn_df = get_xneutr_df(
        f"{eval_path}_*/events*", suffix_filter=suffix_filter, metric_name=metric_name
    )

    return compute_perf_change(base_series, xn_df, keep_cols, suffix_filter, change)


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


# following 2 class/funtion are thanks to https://stackoverflow.com/a/53877507/9889508
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(
        unit="B",
        unit_scale=True,
        miniters=1,
        desc=url.split("/")[-1],
    ) as t:
        urllib.request.urlretrieve(
            url,
            filename=output_path,
            reporthook=t.update_to,
        )


def download_and_unzip(url, download_dir, target_filename):

    file_path = os.path.join(download_dir, target_filename)
    print("Downloading...")
    download_url(url, file_path)
    print("Unzipping...")
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(download_dir)


def list_of_zero():
    """
    Necessary for pickling our default dicts
    see https://stackoverflow.com/a/16439720/9889508
    """
    return [0]


def just_zero():
    """
    Necessary for pickling our default dicts
    see https://stackoverflow.com/a/16439720/9889508
    """
    return 0


def just_underscore():
    """
    Necessary for pickling our default dicts
    see https://stackoverflow.com/a/16439720/9889508
    """
    return "_"
