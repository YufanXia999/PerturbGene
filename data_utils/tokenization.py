import json
from typing import Union, List, Dict, Optional, Iterable

import numpy as np
import scipy.sparse
import torch

from perturbgene.configs import BaseConfig


def _prepend_bin(arr):
    """
    Prepends 'bin_' to every element of a numpy array.
    E.g. [1, 1, 2, 3] -> ['bin_1', 'bin_1', 'bin_2', 'bin_3']
    """
    return np.char.add("bin_", arr.astype(str))


def phenotype_to_token(phenotype_str: str):
    """ Convert phenotype string to a standard format. """
    return f"[{phenotype_str.replace(' ', '_').strip()}]"


class GeneTokenizer:
    """ Tokenizer for individual AnnData cells. """
    def __init__(self, config: BaseConfig):
        """
        Initializes the vocabulary. This includes:
            - special tokens (currently 4)
            - phenotype tokens, loaded from config.vocab_path
            - binned expression tokens (currently `len(config.bin_edges)`)
        """
        self.config = config

        with open(config.vocab_path, "r") as f:
            self.phenotypic_tokens_map = json.load(f)

        if self.config.bin_edges_path is not None:
            with open(config.bin_edges_path, "r") as f:
                self.bin_edges_by_gene = json.load(f)

            self.num_bins = len(self.bin_edges_by_gene[0])
            assert all(len(edges) == self.num_bins for edges in self.bin_edges_by_gene.values()), \
                "Expected the same number of bins per gene."
        else:
            self.num_bins = len(self.config.bin_edges)

        self.flattened_tokens = ["[CLS]", "[EOS]", "[MASK]", "[PAD]"]  # special tokens
        self.cls_token, self.eos_token, self.mask_token, self.pad_token = self.flattened_tokens
        self.num_special_tokens = len(self.flattened_tokens)  # might be pointless since we've hardcoded 4 above anyway

        # sort for consistency, so that all tokenizers initialized on identical vocab have the same `token_to_id_map`
        self.phenotypic_types = sorted(self.phenotypic_tokens_map.keys())
        for phenotypic_type in self.phenotypic_types:
            self.flattened_tokens.extend(self.phenotypic_tokens_map[phenotypic_type])

        self.num_phenotypic_tokens = len(self.flattened_tokens) - self.num_special_tokens

        self.genes_start_ind = 1 + len(self.config.included_phenotypes)  # expected index of first gene in input seq
        # token_type_id for gene 0, after accounting for special and phenotype tokens
        self.gene_token_type_offset = 1 + len(self.phenotypic_types)

        # For clarity, the gene expression tokens will be: 'bin_0', 'bin_1', 'bin_2', ..., 'bin_n'
        # 'bin_0' corresponds to not expressed genes and should not be included in the input, but helpful for MLM
        # self.flattened_tokens.extend(_prepend_bin(np.arange(self.num_bins) + 1).tolist())
        self.flattened_tokens.extend(_prepend_bin(np.arange(self.num_bins + 1)).tolist())  # Important! - breaking backwards compatability by adding bin_0
        self.token_to_id_map = {token: i for i, token in enumerate(self.flattened_tokens)}

    def __call__(self, cell) -> tuple[torch.LongTensor, torch.LongTensor]:
        """
        Represents `cell` as a combination of `input_ids` and `token_type_ids`.

        E.g.
                           [CLS] [AGE_TOKEN] [CELL_TOKEN] ... [TISSUE_TOKEN] gene0 gene100 ... gene57991 [EOS]
        `input_ids` =      [0,   16,         32,          ... 256,           276,  275,    ... 276,      1]
        `token_type_ids` = [0,   1,          2,           ... 5,             6,    106,    ... 57992,    0]

        In the above example, only genes 5, 7, ... are expressed, so only these genes are included in the input.
        Note that special tokens share the same token_type_id (0),
        but each phenotype category has a distinct token_type_id.
        This distinction is important during MLM, because the model can only determine the phenotype category
        it is expected to predict through the token_type_id.

        The actual order and input ids in the example are arbitrary.
        In fact, I believe the genes are not always in ascending order.

        Returns:
            (input_ids, token_type_ids)
        """
        # [CLS]
        input_tokens = [self.cls_token]  # Will have special tokens, phenotypic tokens, and gene expression tokens
        token_type_ids = [0]

        # e.g. [AGE_TOKEN] [CELL_TOKEN] ... [TISSUE_TOKEN]
        input_tokens.extend([phenotype_to_token(cell.obs[phenotypic_type].item())
                             for phenotypic_type in self.config.included_phenotypes])  # Filter to `included_phenotypes`
        # Distinct token_type_id for each phenotype token
        token_type_ids.extend([1 + self.phenotypic_types.index(phenotypic_type)
                               for phenotypic_type in self.config.included_phenotypes])

        # e.g. gene0 gene100 ... gene57991
        bin_ids, gene_ids = self._bin_genes(cell.X)
        # We avoid loading the dense gene expression array
        input_tokens.extend(_prepend_bin(bin_ids))
        token_type_ids.extend(gene_ids + self.gene_token_type_offset)

        # [EOS]
        input_tokens.append(self.eos_token)
        token_type_ids.append(0)

        assert len(input_tokens) == len(token_type_ids)

        if len(input_tokens) > self.config.max_length:
            # print(f"Truncating {len(input_tokens)} tokens to {self.config.max_length=}")
            input_tokens = input_tokens[:self.config.max_length]
            token_type_ids = token_type_ids[:self.config.max_length]

        assert self._check_valid_tokens(input_tokens), input_tokens
        return (torch.LongTensor(self.convert_tokens_to_ids(input_tokens)),
                torch.LongTensor(token_type_ids))

    def _bin_genes(self, sparse_expr_arr: scipy.sparse.csr_matrix) -> tuple:
        """
        Args:
            sparse_expr_arr: the sparse matrix containing the gene expression levels for a single cell

        Returns: (bin_ids, gene_ids)
            bin_ids: the indices of gene expression bins, with
                bin_ids[i] corresponding to the expression level of gene gene_ids[i]
            gene_ids: the indices of the expressed genes

        Note that while `bin_ids` and `gene_ids` loosely correspond with the input_ids and token_type_ids respectively,
        they are both an offset away from the correct index in our vocabulary, so
        additional postprocessing is performed on both `bin_ids` and `gene_ids` in __call__().
        """
        # We avoid loading the dense gene expression array
        nonzero_indices = sparse_expr_arr.indices  # corresponds to the indices of genes with nonzero expression
        indexed_data = sparse_expr_arr.data  # corresponds to the gene expression values for these genes

        # Sort indices so that highly expressed genes are never truncated
        # (also potentially makes implementing MLM masking easier).
        if not np.all(nonzero_indices[:-1] <= nonzero_indices[1:]):  # O(n) check to see if indices are sorted
            # Sometimes (I think when `config.num_top_genes` is the maximum), indices are already sorted.
            sorted_indices = np.argsort(nonzero_indices)
            nonzero_indices = nonzero_indices[sorted_indices]
            indexed_data = indexed_data[sorted_indices]

        gene_expr_bins = np.digitize(indexed_data, self.config.bin_edges)  # bin gene expression values
        expressed_genes_mask = np.flatnonzero(gene_expr_bins)  # only 1 dim, used to filter out low expression genes
        return (gene_expr_bins[expressed_genes_mask],
                nonzero_indices[expressed_genes_mask])

    def _check_valid_tokens(self, tokens: str | list[str]) -> bool:
        """ Checks if all tokens are valid (i.e. in the vocabulary) """
        if isinstance(tokens, str):
            tokens = [tokens]

        return all(token in self.token_to_id_map for token in tokens)

    def _convert_token_to_id(self, token: str) -> int:
        return self.token_to_id_map[token]  # will fail on unknown tokens

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        """
        If tokens is a string, returns the integer index it corresponds to in the vocabulary.
        If tokens is a list of strings, returns the list of indices each string corresponds to in the vocabulary.
        """
        assert self._check_valid_tokens(tokens)
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        else:
            return list(map(self._convert_token_to_id, tokens))

    def get_special_tokens_mask(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """  # TODO: check why it differs, boolTensor for one
        Note: This is different from `PretrainedTokenizer.get_special_tokens_mask`, even the signature is different.
        Args:
            input_ids: Should always be a `torch.LongTensor` for IterableAnnDataset, but works with any iterable.
        Returns:
            A tensor of booleans: True for a special token, False for a phenotype/gene token.
        """
        # The first `self.num_special_tokens` ids correspond to the special tokens
        return torch.lt(input_ids, self.num_special_tokens)

    def get_phenotypic_tokens_mask(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Returns:
            A tensor of booleans: True for a phenotype token, False for a special/gene token.
        """
        return torch.ge(input_ids, self.num_special_tokens) \
            & torch.lt(input_ids, self.num_special_tokens + self.num_phenotypic_tokens)

    def get_gene_tokens_mask(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Returns:
            A tensor of booleans: True for a gene expression (bin_id) token, False for a special/phenotype token.
        """
        return torch.ge(input_ids, self.num_special_tokens + self.num_phenotypic_tokens)

    @property
    def vocab_size(self) -> int:
        """ Number of unique `input_ids` """
        return len(self.flattened_tokens)

    @property
    def type_vocab_size(self) -> int:
        """ Number of unique `token_type_ids` """
        return self.gene_token_type_offset + self.config.num_top_genes
