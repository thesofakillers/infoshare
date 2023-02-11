from typing import Optional

from torch import Tensor

from infoshare.models.wsd_classifier import WSDClassifier


class LSWSDClassifier(WSDClassifier):
    """Same as WSD Classifier but we override the log_single_metric function"""

    def log_single_metric(
        self,
        metric_per_pos: Tensor,
        metric_name: str,
        batch_size: int,
        stage: str,
        dataloader_idx: Optional[int] = None,
        prefix: str = "",
    ):
        """
        This is exactly the same as WSDClassifier but
        we only deal with one dataset so no need to index and preprend metrics

        The commented out code is the only difference
        """
        log_name = f"{prefix}{stage}_{metric_name}"
        # if stage == "test":
        #     curr_dataset = self.trainer.datamodule.idx_to_dataset[dataloader_idx]
        #     log_name = f"{curr_dataset}/" + log_name

        metric_avg = metric_per_pos.nanmean()  # this is part of the shortcut
        self.log(log_name, metric_avg, batch_size=batch_size)

        if stage != "test":
            # No need to log per-pos-tag metric for train and val
            return

        # log average metric per-pos tag
        for i, metric_i in enumerate(metric_per_pos):
            pos_name = self.hparams.pos_map[i]
            self.log(
                f"{log_name}_{pos_name}",
                metric_i,
                batch_size=batch_size,
            )
