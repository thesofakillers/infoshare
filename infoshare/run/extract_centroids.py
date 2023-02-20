"""Extracts class centroids from a given checkpoint"""
import torch

# fmt: off
pos_id2cname = [ "NOUN", "PUNCT", "ADP", "NUM", "SYM", "SCONJ", "ADJ", "PART", "DET",
                "CCONJ", "PROPN", "PRON", "X", "_", "ADV", "INTJ", "VERB", "AUX", ]
dep_id2cname = ['_', 'acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'case', 'cc',
                'ccomp', 'compound', 'conj', 'cop', 'csubj', 'dep', 'det', 'discourse',
                'dislocated', 'expl', 'fixed', 'flat', 'goeswith', 'iobj', 'list',
                'mark', 'nmod', 'nsubj', 'nummod', 'obj', 'obl', 'orphan', 'parataxis',
                'punct', 'reparandum', 'root', 'vocative', 'xcomp']
lswsd_id2cname = ['unk', 'then%4:02:01::', 'permit%2:32:00::', 'no%4:02:01::',
                  'today%1:28:00::', 'permit%1:04:00::', 'program%1:10:04::',
                  'plant%1:03:00::', 'second%5:00:00:ordinal:00', 'result%1:11:00::',
                  'individual%5:00:00:independent:00', 'next%4:02:00::',
                  'rate%2:31:00::', 'public%1:14:00::', 'list%1:07:00::',
                  'then%5:00:00:past:00', 'less%1:23:00::', 'age%1:07:00::',
                  'then%4:02:02::', 'permit%1:10:02::', 'no%4:02:02::',
                  'rate%2:42:01::', 'list%1:10:00::', 'no%3:00:00::', 'fear%2:37:00::',
                  'enough%1:23:00::', 'express%2:32:03::',
                  'express%5:00:00:explicit:00', 'here%5:00:00:present:02',
                  'most%4:02:02::', 'here%4:02:01::', 'age%1:28:00::',
                  'result%1:19:00::', 'next%5:00:00:incoming:00', 'attempt%2:41:00::',
                  'most%4:02:01::', 'second%5:00:00:intermediate:00', 'just%4:02:00::',
                  'no%1:10:00::', 'public%3:00:00::', 'next%5:00:00:close:01',
                  'age%2:30:00::', 'now%4:02:01::', 'fear%1:12:00::', 'same%3:00:04::',
                  'cost%2:42:01::', 'pressure%1:07:00::', 'cost%1:21:00::',
                  'result%2:42:02::', 'some%3:00:00::', 'second%1:28:00::',
                  'same%4:02:00::', 'now%4:02:02::', 'less%4:02:00::',
                  'second%4:02:00::', 'now%4:02:00::', 'age%2:30:01::',
                  'less%4:02:01::', 'express%2:32:02::', 'same%3:00:02::',
                  'then%1:28:00::', 'just%4:02:01::', 'most%3:00:01::',
                  'here%4:02:04::', 'some%5:00:00:unspecified:00', 'attempt%1:04:02::',
                  'local%5:00:00:native:01', 'individual%1:03:00::',
                  'express%2:32:00::', 'once%4:02:00::', 'cost%2:42:00::',
                  'once%4:02:01::', 'program%2:36:00::', 'some%4:02:00::',
                  'plant%1:06:01::', 'just%3:00:00::', 'here%4:02:00::',
                  'local%3:01:01::', 'program%1:09:01::', 'local%3:00:03::',
                  'now%4:02:04::', 'attempt%2:36:00::', 'less%3:00:00::',
                  'cost%1:07:01::', 'rate%1:21:00::', 'most%1:23:00::',
                  'just%4:02:04::', 'fear%1:12:01::', 'plant%2:36:00::',
                  'public%1:14:01::', 'express%2:32:01::',
                  'individual%5:00:00:personal:00', 'later%5:00:00:late:00',
                  'cost%1:07:00::', 'later%5:00:00:subsequent:00', 'plant%2:35:01::',
                  'today%4:02:01::', 'list%2:32:00::', 'some%5:00:00:many:00',
                  'age%1:28:02::', 'individual%3:00:00::', 'individual%1:18:00::',
                  'public%5:00:00:common:02', 'just%4:02:05::', 'then%4:02:00::',
                  'result%1:10:00::', 'next%5:00:00:succeeding(a):00',
                  'some%5:00:00:much(a):00', 'local%3:00:01::', 'program%2:32:00::',
                  'enough%4:02:00::', 'local%1:06:00::', 'most%4:02:00::',
                  'fear%2:37:03::', 'permit%2:41:00::', 'pressure%1:04:00::',
                  'second%1:04:00::', 'just%4:02:03::', 'most%3:00:02::',
                  'program%1:10:01::', 'here%4:02:02::', 'later%4:02:01::',
                  'now%4:02:05::', 'same%5:00:00:unchanged:00', 'result%2:42:00::',
                  'today%4:02:00::', 'plant%2:35:00::',
                  'public%5:00:00:common:02;3:00:00::',
                  'individual%5:00:00:unshared:00', 'program%1:09:00::',
                  'pressure%1:19:00::', 'later%4:02:02::',
                  'enough%5:00:00:sufficient:00', 'fear%2:37:02::',
                  'once%5:00:00:past:00', 'same%3:00:00::', 'second%1:28:01::',
                  'list%2:41:00::', 'rate%1:28:00::', 'once%4:02:02::',
                  'attempt%1:04:00::', 'now%1:28:00::', 'pressure%1:26:00::',
                  'today%1:28:01::' ]
# fmt: on

task_id2cname = {
    "POS": pos_id2cname,
    "DEP": dep_id2cname,
    "LSWSD": lswsd_id2cname,
}


def main(ckpt_path, output_path, task):
    id_to_cname = task_id2cname[task]

    ckpt = torch.load(ckpt_path, map_location="cpu")
    centroids = ckpt["class_centroids"]

    mapped_centroids = {id_to_cname[idx]: v for (idx, v) in centroids.items()}

    torch.save(mapped_centroids, output_path)


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

    parser.add_argument(
        "-t",
        "--centroid-task",
        type=str,
        help="Which task the centroids were computed from. One of {POS, DEP, LSWSD}",
        required=True,
    )

    args = parser.parse_args()

    main(args.checkpoint_path, args.output_path, args.centroid_task)
