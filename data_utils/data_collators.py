"""

"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers.data.data_collator import DataCollatorForLanguageModeling

from .tokenization import GeneTokenizer


# constants for `torch_mask_tokens`
SWAP_GENE_IDS_PROB = 0.9  # sequence wise
SWAP_GENE_EXPR_PROB = 0.1
SHOW_GENE_EXPR_PROB = 0.1
DROP_PHENOTYPE_PROB = 0.1  # multiplied by phenotype_mlm_prob

# We expect SEQ_KEYS in every example, and their tensors should have the same length
SEQ_KEYS = {"input_ids", "token_type_ids", "attention_mask"}
ALL_POSSIBLE_KEYS = SEQ_KEYS.union({"labels"})  # `labels` are optional


def collate_fn_wrapper(tokenizer: GeneTokenizer, pad_to_multiple_of=None):
    """
    Trainer expects a function that takes in just `examples`, so using a wrapper to pass in other arguments.
    """
    def collate_fn(examples: List[Dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Pads all examples in a batch to the same length, which is guaranteed to be a multiple of `pad_to_multiple_of`,
        if specified. Enforces strong assumptions about the examples; specifically, each example should take the form:
        {
            "input_ids": torch.Tensor of size (seq_len,)  # TODO: verify sizes
            "token_type_ids": torch.Tensor of size (seq_len,)
            "attention_mask": torch.Tensor of size (seq_len,)
            (and optionally) "labels": torch.Tensor of size (1,)
        }
        """
        all_keys = examples[0].keys()  # assuming all examples have the same keys, is verified later
        assert all(key in all_keys for key in SEQ_KEYS)  # TODO: create attention_mask if missing
        assert ALL_POSSIBLE_KEYS.issuperset(all_keys), f"Unexpected keys: {all_keys - ALL_POSSIBLE_KEYS=}"

        # `max_length` is the length of the longest sequence in the batch, or at most `self.tokenizer.config.max_length`
        max_length = -1
        for example in examples:
            # TODO: check __eq__ for dictkeys?
            assert example.keys() == all_keys, (f"Expected all examples to have the same keys, "
                                                f"but {example.keys()=} != {all_keys=}")
            max_length = max(len(example["input_ids"]), max_length)

        if pad_to_multiple_of is not None:
            # If self.tokenizer.config.max_length is not a perfect multiple we might pad over it
            assert tokenizer.config.max_length % pad_to_multiple_of == 0, \
                f"Expected {tokenizer.config.max_length=} to be a multiple of {pad_to_multiple_of=}"
            max_length = int(math.ceil(max_length / pad_to_multiple_of) * pad_to_multiple_of)
            assert max_length % pad_to_multiple_of == 0  # TODO: remove

        assert max_length <= tokenizer.config.max_length, (max_length, tokenizer.config.max_length)  # TODO: remove
        batch = {key: [] for key in all_keys}
        for example in examples:
            # IterableAnnDataset should only yield tensors
            assert all(isinstance(val, torch.Tensor) for val in example.values())
            assert len({len(example[key]) for key in SEQ_KEYS}) == 1, \
                f"Values have different lengths, {[(key, len(example[key])) for key in SEQ_KEYS]=}"

            for key in all_keys:
                tensor = example[key]
                if key in SEQ_KEYS:
                    assert len(tensor) <= max_length, "Expected tokenizer to truncate"
                    if len(tensor) < max_length:  # pad example to `max_length`
                        if key == "input_ids":  # add `pad_token`s
                            tensor = torch.cat([tensor, torch.full(
                                [max_length - len(tensor)],
                                fill_value=tokenizer.convert_tokens_to_ids(tokenizer.pad_token),  # TODO: cache
                                dtype=torch.long,
                                device=tensor.device
                            )])
                        elif key == "token_type_ids":  # add 0s (the token_type_id for `pad_token`)
                            tensor = torch.cat([tensor, torch.zeros(  # TODO: func, don't hardcode 0?
                                [max_length - len(tensor)],
                                dtype=torch.long,
                                device=tensor.device
                            )])
                        elif key == "attention_mask":  # add 0s
                            tensor = torch.cat([tensor, torch.zeros(  # same code as token_type_ids actually
                                [max_length - len(tensor)],
                                dtype=torch.long,
                                device=tensor.device
                            )])
                else:
                    if key == "labels":
                        if tokenizer.config.subcommand == "mlm":  # For MLM, there is one label for each token, so pad
                            tensor = torch.cat([tensor, torch.full(
                                [max_length - len(tensor)],
                                fill_value=-100,
                                dtype=torch.long,
                                device=tensor.device
                            )])
                        elif tokenizer.config.subcommand != "cls":
                            raise NotImplementedError("Check if `labels` should be padded.")
                    else:
                        raise NotImplementedError(f"Check if {key=} nees to be truncated.")

                batch[key].append(tensor)

        tensor_batch = {}
        for key in all_keys:
            tensor_batch[key] = torch.stack(batch.pop(key))  # stack list of tensors into a single tensor

        return tensor_batch

    return collate_fn


def torch_mask_tokens(batch, tokenizer, gene_mask_prob, phenotype_mask_prob=None, rng=None):
    """
    Modified version of `DataCollatorForLanguageModeling.torch_mask_tokens`, signature is different though.
    Mutates `batch` in-place.

    Masking strategy for phenotype tokens:
        We mask phenotype tokens randomly with `phenotype_mask_prob`. To prepare the model for the inference scenario
        where some phenotypes are missing from the input (i.e. not even masked), we randomly "drop" masked tokens
        with `DROP_PHENOTYPE_PROB` (i.e. `phenotype_mask_prob * DROP_PHENOTYPE_PROB` of all phenotype tokens).
        To keep the indices of the input consistent, the dropping is done by replacing the input_id with the pad token.
    Masking strategy for gene tokens:
        We mask gene tokens randomly with `gene_mask_prob`.
        For each sequence in a batch, with probability`SWAP_GENE_IDS_PROB` we will randomly swap expressed genes
        with not expressed ones with `1 / num_bins / SWAP_GENE_IDS_PROB` probability.
        This is primarily designed for the two bin (not expressed/expressed) case,
        where the gene expression prediction would otherwise be trivial (always predict expressed bin).
        Note that the not expressed gene is selected uniformly from all genes not in the input,
        which is suboptimal because:
        1) Some genes are never expressed in any cell (~1k/58k genes in the current dataset, even more with truncation).
        2) On shorter sequence lengths, we could swap in an expressed gene that got truncated from the input.
        On `1 - SWAP_GENE_IDS_PROB` of sequences we perform no swapping,
        since that probably aligns closer with downstream tasks.
    """
    batch["labels"] = torch.full_like(batch["input_ids"], -100, dtype=torch.long)

    phenotypic_tokens_mask = tokenizer.get_phenotypic_tokens_mask(batch["input_ids"])
    gene_tokens_mask = tokenizer.get_gene_tokens_mask(batch["input_ids"])

    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    gene_prob_matrix = torch.full(batch["input_ids"].shape, gene_mask_prob)  # TODO: full_like but cpu?
    gene_prob_matrix.masked_fill_(~gene_tokens_mask, value=0.0)   # only mask gene tokens

    # Gene MLM
    if gene_mask_prob > 0:
        gene_masked_indices = torch.bernoulli(gene_prob_matrix, generator=rng).bool()
        batch["labels"][gene_masked_indices] = batch["input_ids"][gene_masked_indices]
        batch["input_ids"][gene_masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # Replacing 1 / (num_bins + 1) with a random not expressed token, so that bin_0 is a label sometimes
        # trying to get bin_0 to appear as often as the other bins,
        # scaling by additional `1 / SWAP_GENE_PROB` because bin_0 never appears `1 - SWAP_GENE_PROB` of the time
        rand_swap_prob = 1 / (tokenizer.num_bins + 1) / SWAP_GENE_IDS_PROB
        rand_swapped_indices = torch.bernoulli(gene_masked_indices.float() * rand_swap_prob, generator=rng).bool()
        # Giving up on pure torch for now...  TODO
        for batch_idx in range(len(batch["input_ids"])):
            if torch.rand(1, generator=rng).item() > SWAP_GENE_IDS_PROB:
                continue

            not_expr_genes = torch.ones(tokenizer.config.num_top_genes)
            not_expr_genes.scatter_(0, batch["token_type_ids"][batch_idx][gene_tokens_mask[batch_idx]] - tokenizer.gene_token_type_offset,
                                    torch.zeros_like(not_expr_genes))
            num_samples = rand_swapped_indices[batch_idx].long().sum().item()
            if num_samples > 0:  # very unlikely to have 0 samples
                rand_not_expr_gene_indices = torch.multinomial(not_expr_genes, num_samples, replacement=False)
                batch["labels"][batch_idx][rand_swapped_indices[batch_idx]] = tokenizer.convert_tokens_to_ids("bin_0")
                batch["token_type_ids"][batch_idx][rand_swapped_indices[batch_idx]] = rand_not_expr_gene_indices \
                    + tokenizer.gene_token_type_offset

    # Phenotype MLM
    if phenotype_mask_prob > 0:
        phenotype_prob_matrix = torch.zeros(batch["labels"].shape)
        phenotype_prob_matrix.masked_fill_(phenotypic_tokens_mask, value=phenotype_mask_prob)
        phenotype_masked_indices = torch.bernoulli(phenotype_prob_matrix, generator=rng).bool()
        batch["labels"][phenotype_masked_indices] = batch["input_ids"][phenotype_masked_indices]
        batch["input_ids"][phenotype_masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # Replace DROP_PHENOTYPE_PROB of masked phenotypes with pad tokens.
        dropped_phenotype_masked_indices = torch.bernoulli(phenotype_masked_indices.float() * DROP_PHENOTYPE_PROB,
                                                           generator=rng).bool()
        batch["labels"][dropped_phenotype_masked_indices] = -100
        batch["input_ids"][dropped_phenotype_masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        batch["token_type_ids"][dropped_phenotype_masked_indices] = 0


@dataclass
class DataCollatorForPhenotypicMLM(DataCollatorForLanguageModeling):
    tokenizer: GeneTokenizer
    # `mlm_probability` corresponds to `gene_mask_prob`
    phenotype_mask_prob: float = 0.5

    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Specifically adapted for IterableAnnDataset.
        Pads examples in a batch using `collate_fn` and additionally masks tokens (see `torch_mask_tokens`).
        """
        assert isinstance(examples[0], dict)
        batch = collate_fn_wrapper(self.tokenizer, self.pad_to_multiple_of)(examples)

        if not self.mlm:
            raise NotImplementedError

        torch_mask_tokens(batch, self.tokenizer, self.mlm_probability, self.phenotype_mask_prob)

        return batch

    def torch_mask_tokens(self, input_ids: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        raise NotImplementedError("Switched to `_torch_mask_tokens`")
