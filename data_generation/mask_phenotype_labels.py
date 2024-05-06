"""
Set phenotype labels to -100, so that the model predictions are not included in the evaluation metrics.
Currently using this with gene masking, where we evaluate phenotype predictions on one dataset
and gene expression predictions on this dataset.
"""

import json
import os

import torch
from tqdm.auto import tqdm

from data_utils import GeneTokenizer


unmasked_json_path = "perturbgene/data/validation_data/seq1024_mlm_gene015_pheno050_shard0_v0.json"


with open(unmasked_json_path, "r") as f:
    unmasked_shard = json.load(f)


metadata = unmasked_shard["metadata"]


class FakeConfig(dict):
    """
    GeneTokenizer accesses keys via __getattr__(), since that works for the Config classes,
    but it does not work for standard dictionaries.
    https://stackoverflow.com/a/23689767
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


tokenizer = GeneTokenizer(FakeConfig(metadata))
masked_shard: list[dict] = []  # more efficient to just write to unmasked_shard["data"]
for cell in tqdm(unmasked_shard["data"]):
    cell_tensors = {key: torch.tensor(val) for key, val in cell.items()}
    phenotype_mask = tokenizer.get_phenotypic_tokens_mask(cell_tensors["labels"])  # we want to remove phenotype classes from labels
    cell_tensors["labels"][phenotype_mask] = -100

    masked_shard.append({key: val.tolist() for key, val in cell_tensors.items()})


metadata["masked_phenotype_labels"] = True
prev_filename = os.path.basename(unmasked_json_path)
assert prev_filename.lower().endswith(".json")
with open(os.path.join(os.path.dirname(unmasked_json_path), f"{prev_filename[:-5]}_no_pheno.json"), "w") as f:
    json.dump({
        "metadata": metadata,
        "data": masked_shard,
    }, f)
