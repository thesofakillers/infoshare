import re
import glob
from argparse import Namespace

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import numpy as np


def get_experiment_name(args: Namespace) -> str:
    experiment_name = f"agg={args.aggregation}_probe={args.probe_layer}"
    if args.task == "DEP":
        experiment_name += f"_concat-mode={args.concat_mode}"
    return experiment_name


TAG_REGEX = r".*\/evaluation_(\w+)\/.*"
EXPERIMENT_REGEX = r".*\/(.+=.+)\/.*"


def get_xn_df(glob_path):
    neutralizers = {}
    for path in glob.glob(glob_path):
        res = re.search(TAG_REGEX, path)
        base_tag = res.group(1).upper()

        ea = EventAccumulator(path)
        ea.Reload()

        scalars = {}
        for metric in ea.Tags()["scalars"]:
            if "test" not in metric:
                continue

            scalar = ea.Scalars(metric)[0].value

            if "/" not in metric:
                tag = "avg"
            else:
                tag = metric.split("_")[-1].upper()

            scalars[tag] = scalar

        neutralizers[base_tag] = scalars

    df = pd.DataFrame(neutralizers)
    df.index.name = "Target"
    df = df.T
    df.index.name = "Neutralizer"
    return df


def get_base_series(path):
    if "*" in path:
        path = glob.glob(path)[0]

    ea = EventAccumulator(path)
    ea.Reload()

    scalars = {}
    for metric in ea.Tags()["scalars"]:
        if "test" not in metric:
            continue

        scalar = ea.Scalars(metric)[0].value

        parts = metric.split("_")
        if len(parts) == 2:
            tag = "avg"
        else:
            tag = parts[-1].upper()

        scalars[tag] = scalar

    return pd.Series(scalars)


def get_acc_drop(eval_path, keep_cols=None):
    base_series = get_base_series(f"{eval_path}/events*")
    xn_df = get_xn_df(f"{eval_path}_*/events*")

    nulls = base_series.isnull()
    base_series = base_series[~nulls]
    xn_df = xn_df.T[~nulls].T

    acc_drop = (xn_df - base_series) / base_series
    acc_drop.sort_index(axis=0, inplace=True)
    acc_drop.sort_index(axis=1, inplace=True)

    drop_indices = set(acc_drop.T.columns).difference(acc_drop.columns)
    drop_columns = set(acc_drop.columns).difference(acc_drop.T.columns).difference(["avg"])

    for hidden in ("DEP", "APPOS"):
        if hidden in acc_drop.T.columns:
            drop_indices.add(hidden)
        if hidden in acc_drop.columns:
            drop_columns.add(hidden)

    acc_drop.drop(index=drop_indices, inplace=True)
    acc_drop.drop(columns=drop_columns, inplace=True)
    if keep_cols is not None:
        acc_drop = (acc_drop.loc[keep_cols])[keep_cols]
    return acc_drop


def get_experiments_df(task, treebank, model, logdir="../lightning_logs"):
    experiments = {}
    for path in glob.glob(f"{logdir}/{model}/{treebank}/{task}/*/evaluation"):
        res = re.search(EXPERIMENT_REGEX, path)
        experiment = res.group(1)
        acc_drop = get_acc_drop(path)

        self_neutr = {}
        for tag in acc_drop.columns.values[:-1]:
            tag = tag.upper()
            self_neutr[tag] = acc_drop[tag][tag]

        self_neutr["avg"] = np.nanmean(list(self_neutr.values()))
        experiments[experiment] = self_neutr

    df = pd.DataFrame(experiments)
    df.sort_values(by="avg", axis=1, inplace=True)

    nulls = df.isnull().all(1)
    df = df[~nulls]

    df.index.name = "Neutralizer"
    df = df.T
    df.index.name = "Experiment"
    return df

