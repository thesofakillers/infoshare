from argparse import Namespace


def get_experiment_name(args: Namespace) -> str:
    experiment_name = f"agg={args.aggregation}_probe={args.probe_layer}"
    if args.task == "DEP":
        experiment_name += f"_concat-mode={args.concat_mode}"
    return experiment_name
