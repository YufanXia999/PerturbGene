from typing import Callable

import torch
from anndata import AnnData
from transformers.modeling_outputs import SequenceClassifierOutput

from perturbgene.configs import BaseConfig
from perturbgene.data_utils.tokenization import phenotype_to_token, GeneTokenizer
from perturbgene.model import GeneBertForPhenotypicMLM, GeneBertForClassification, GeneBertModel


def get_inference_config(
        bin_edges: list[int], pretrained_model_path: str, max_length: int, num_top_genes: int,
        vocab_path: str = "/home/kevin/Documents/perturbgene/data/phenotypic_tokens_map.json",
        per_device_eval_batch_size: int = 4096
):
    """
    Only need a subset of BaseConfig for inference. This config will mainly be used for creating `GeneTokenizer`s.
    """
    return BaseConfig(
        subcommand=None,
        bin_edges=bin_edges,
        bin_edges_path=None,
        pretrained_model_path=pretrained_model_path,
        model_arch=None,
        shard_size=None,
        eval_data_paths=[],
        max_length=max_length,
        num_top_genes=num_top_genes,
        vocab_path=vocab_path,
        included_phenotypes=None,
        use_flash_attn=True,
        per_device_eval_batch_size=per_device_eval_batch_size,
        dataloader_num_workers=0,
        auto_find_batch_size=False,
        output_dir=None,
    )


def prepare_cell(cell: AnnData, model_type: str, tokenizer: GeneTokenizer,
                 label2id: dict[str, int] = None) -> dict[str, torch.Tensor]:
    """
    Converts an h5ad cell to `input_ids`.

    Args:
        cell: AnnData object with n_obs = 1
        model_type: Expecting "mlm" or "cls"
        tokenizer: To encode cell into `input_ids` and `token_type_ids`
        label2id: Only required for model_type == "cls"
    """
    input_ids, token_type_ids = tokenizer(cell)
    cell_data = {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
    }

    if model_type == "cls":
        label_name = cell.obs[tokenizer.config.phenotype_category].item()
        cell_data["labels"] = torch.tensor(label2id[phenotype_to_token(label_name)],
                                           dtype=torch.long).unsqueeze(0)

    return cell_data


def test_cell(prepared_cell: dict[str, torch.Tensor], model: GeneBertForPhenotypicMLM | GeneBertForClassification,
              data_collator: Callable) -> SequenceClassifierOutput:
    """
    Args:
        prepared_cell: Output of `prepare_cell`
        model: Model to perform inference with
        data_collator: Basically just needed to unsqueeze tensors into batch with one example
    """
    batched_cell = data_collator([prepared_cell])  # batch with 1 cell
    with torch.no_grad():
        output = model(**{key: val.to(model.device) for key, val in batched_cell.items()})

    return output


def mlm_for_phenotype_cls(cell: AnnData, phenotype_category: str, model: GeneBertForPhenotypicMLM,
                          tokenizer: GeneTokenizer, data_collator: Callable) -> int:
    """
    Performing phenotype classification using an MLM model by masking the corresponding input token
    and returning the word with the highest predicted probability.

    Args:
        cell: AnnData object with n_obs = 1
        phenotype_category: the category to perform classification on; e.g. "tissue" or "cell_type"
        model: passed to `test_cell`
        tokenizer: passed to `prepare_cell`
        data_collator: passed to `test_cell`

    Returns:
        the input_id with the highest predicted probability
    """
    assert phenotype_category in tokenizer.config.included_phenotypes

    phenotype_ind = 1 + tokenizer.config.included_phenotypes.index(phenotype_category)  # offset by 1 b/c CLS token
    prepared_cell = prepare_cell(cell, "mlm", tokenizer)
    assert prepared_cell["token_type_ids"][phenotype_ind] == tokenizer.phenotypic_types.index(phenotype_category) + 1, \
        (prepared_cell["token_type_ids"][phenotype_ind], tokenizer.phenotypic_types)

    prepared_cell["input_ids"][phenotype_ind] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    output = test_cell(prepared_cell, model, data_collator)
    return output.logits.argmax(dim=-1).squeeze(0)[phenotype_ind].item()

def get_gene_embedding(cell: AnnData, model: GeneBertModel, 
                       tokenizer: GeneTokenizer, data_collator: Callable, 
                       gene_name: str = None) -> torch.tensor:
    """
    Getting Inference from a pre-trained Bert model and returning the gene embedding

    Args:
        cell: AnnData object with n_obs = 1
        model: passed to `test_cell`
        tokenizer: passed to `prepare_cell`
        data_collator: passed to `test_cell`
        gene_name: the name of gene you want to get in AnnData object --> assume is str and is saved in var feature_name

    Returns:
        the whole cell embedding or if gene_name is specify, the embedding of that gene
    """

    prepared_cell = prepare_cell(cell, "mlm", tokenizer)
    batched_cell = data_collator([prepared_cell])
    with torch.no_grad():
        output = model(**{key: val.to(model.device) for key, val in batched_cell.items()}, output_hidden_states=True)
    if gene_name is not None:
        cell_idx = tokenizer.genes_start_ind + 1 + cell.var.index.get_loc(cell.var.index[cell.var['feature_name'] == gene_name][0])
        return output.last_hidden_state[:,torch.nonzero(batched_cell["token_type_ids"] == cell_idx)[0,1]]
    else:
        return output.last_hidden_state