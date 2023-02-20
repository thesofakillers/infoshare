"""Extracts class centroids from a given checkpoint"""
import torch


def main(ckpt_path, output_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    centroids = ckpt["class_centroids"]
    torch.save(centroids, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extracts class centroids from a given checkpoint"
    )

    parser.add_argument(
        "-ckpt",
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the checkpoint to extract centroids from",
    )

    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="Path to the output file",
    )

    args = parser.parse_args()

    main(args.checkpoint_path, args.output_path)
