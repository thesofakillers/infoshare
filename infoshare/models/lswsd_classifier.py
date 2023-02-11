from typing import Optional

from infoshare.models.wsd_classifier import WSDClassifier


class LSWSDClassifier(WSDClassifier):
    """Same as WSD Classifier but we override the prepare_log_name function"""

    def prepare_log_name(
        self,
        prefix: str,
        stage: str,
        metric_name: str,
        dataloader_idx: Optional[int] = None,
    ):
        """
        exactly the same as in WSD classifier,
        but now we only have one test dataloader
        so we dont need to index
        """
        log_name = f"{prefix}{stage}_{metric_name}"
        # if stage == "test":
        #     curr_dataset = self.trainer.datamodule.idx_to_dataset[dataloader_idx]
        #     log_name = f"{curr_dataset}/" + log_name
        return log_name
