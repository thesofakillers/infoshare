"""functionality for cross-neutralising"""
from typing import Optional

import torch.utils.data as torch_data
import torch
from tqdm import tqdm


def get_tag_vectors(
    model: torch.nn.Module,
    dataloader: torch_data.DataLoader,
    n_sents: Optional[int] = None,
):
    """
    Approximates the tag centroid for each tag

    Parameters
    ----------
    model : torch.nn.Module
        instance of a Classifier model
    dataloader: torch.utils.data.DataLoader
    n_sents : int
        the number of sentences to use

    Returns
    -------
    dict
        a dictionary with key : value :: tag_id : mean vector
    """
    model.eval()
    with torch.no_grad():
        tag_vectors = {}
        # for each batch
        for encodings, targets in tqdm(dataloader, total=len(dataloader)):
            batched_embs = model.bert(encodings)
            batched_preds = model.classifier(batched_embs)
            # for each sentence in batch
            for embs, preds, target in tqdm(
                zip(batched_embs, batched_preds, targets), total=dataloader.batch_size
            ):
                preds = preds[: len(target), :].argmax(dim=1).tolist()
                # for each word in sentence
                for emb, pred in zip(embs, preds):
                    if pred not in tag_vectors:
                        tag_vectors[pred] = [emb]
                    else:
                        tag_vectors[pred].append(emb)
    print("computing mean vectors")
    tag_vectors = {k: torch.mean(torch.vstack(v), dim=0) for k, v in tag_vectors.items()}
    return tag_vectors
