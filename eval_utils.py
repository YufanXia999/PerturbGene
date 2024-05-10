import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score

import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers.set_seed(seed)


def preprocess_logits_argmax(logits, labels):
    """
    We currently only need the top predicted class instead of all the logits,
    so this preprocessing saves significant memory.
    """
    if isinstance(logits, tuple):
        # should not happen for `GeneBert` variants, but other models may have extra tensors like `past_key_values`
        logits = logits[0]

    return logits.argmax(dim=-1)


def compute_general_metrics(flat_preds: np.ndarray, flat_labels: np.ndarray) -> dict[str, float]:
    """
    Args:
        flat_preds: Flat numpy array of predictions (argmax of logits)
        flat_labels: Flat numpy array of labels, with the same shape as `flat_preds`
        Note that it is assumed that the labels corresponding to -100 have already been filtered out.

    Returns:
        Dictionary of different metric values ("accuracy", "precision", "recall", "f1").

    Note: Setting `average='macro'` for macro-average (average over classes)
    Using `zero_division=0` to handle cases where there are no true or predicted samples for a class
    """

    metrics = {
        "accuracy": accuracy_score(flat_labels, flat_preds),
        "precision": precision_score(flat_labels, flat_preds, average='macro', zero_division=0),
        "recall": recall_score(flat_labels, flat_preds, average='macro', zero_division=0),
    }

    metrics["f1"] = 2 / (1 / metrics["precision"] + 1 / metrics["recall"]) \
        if metrics["precision"] != 0 and metrics["recall"] != 0 else 0
    return metrics


def mlm_metrics_wrapper(tokenizer, working_dir: str = None):
    eval_iter = 0

    def compute_mlm_metrics(p: transformers.EvalPrediction):
        """
        Computes MLM accuracy from EvalPrediction object.

        Args:
            - p (EvalPrediction): An object containing the predictions and labels.

        Returns:
            - dict: A dictionary containing the accuracy under the key 'accuracy'.
        """
        all_metrics = dict()

        # Extract predictions and labels from the EvalPrediction object
        assert isinstance(p.predictions, np.ndarray), p.predictions
        preds = p.predictions  # already took `argmax` in `preprocess_logits_for_metrics`
        labels = p.label_ids
        assert preds.shape == labels.shape, f"{preds.shape=} != {labels.shape=}"

        # Ignoring -100 used for non-masked tokens
        mask = labels != -100
        flat_preds, flat_labels = preds[mask], labels[mask]  # flattened
        overall_metrics = compute_general_metrics(flat_preds, flat_labels)

        all_metrics.update({f"{metric_name}_overall": metric_val
                            for metric_name, metric_val in overall_metrics.items()})

        label_names = tokenizer.flattened_tokens
        conf_matrix = confusion_matrix(flat_preds, flat_labels, labels=np.arange(len(label_names)))

        df_cm = pd.DataFrame(conf_matrix, index=label_names, columns=label_names)
        # # Save DataFrame to CSV
        nonlocal eval_iter
        if working_dir is not None:
            df_cm.to_csv(f'{working_dir}/confusion_matrix_{eval_iter}.csv')

        eval_iter += 1

        for i, phenotypic_type in enumerate(tokenizer.config.included_phenotypes):
            curr_preds, curr_labels = preds[:, i + 1], labels[:, i + 1]  # first index is CLS; TODO don't hardcode
            curr_mask = curr_labels != -100
            curr_metrics = compute_general_metrics(curr_preds[curr_mask], curr_labels[curr_mask])
            all_metrics.update({f"{metric_name}_{phenotypic_type[:3]}": metric_val
                                for metric_name, metric_val in curr_metrics.items()})

        if tokenizer.config.gene_mask_prob > 0:  # TODO: better checks to avoid nans
            mlm_preds, mlm_labels = preds[:, tokenizer.genes_start_ind:], labels[:, tokenizer.genes_start_ind:]
            mlm_mask = mlm_labels != -100
            mlm_metrics = compute_general_metrics(mlm_preds[mlm_mask], mlm_labels[mlm_mask])
            all_metrics.update({f"{metric_name}_mlm": metric_val
                                for metric_name, metric_val in mlm_metrics.items()})

        return all_metrics

    return compute_mlm_metrics


def cls_metrics_wrapper(tokenizer, working_dir: str = None):
    eval_iter = 0

    def compute_cls_metrics(p: transformers.EvalPrediction):
        assert isinstance(p.predictions, np.ndarray), p.predictions
        preds = p.predictions  # already took `argmax` in `preprocess_logits_for_metrics`

        labels = p.label_ids.squeeze(1)
        assert preds.shape == labels.shape, f"{preds.shape=} != {labels.shape=}"

        metrics = compute_general_metrics(preds, labels)

        # Copied from IterableAnnDataset
        if tokenizer.config.binary_label is None:
            label_names = tokenizer.phenotypic_tokens_map[tokenizer.config.phenotype_category]
        else:
            label_names = [f"NOT {tokenizer.config.binary_label}", tokenizer.config.binary_label]

        # Copied from train_mlm_wrapper
        conf_matrix = confusion_matrix(preds, labels, labels=np.arange(len(label_names)))

        df_cm = pd.DataFrame(conf_matrix, index=label_names, columns=label_names)
        # # Save DataFrame to CSV
        nonlocal eval_iter
        if working_dir is not None:
            df_cm.to_csv(f'{working_dir}/confusion_matrix_{eval_iter}.csv')

        eval_iter += 1

        return metrics

    return compute_cls_metrics
