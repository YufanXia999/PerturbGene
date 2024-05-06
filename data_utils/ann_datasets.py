import json

import accelerate
import numpy as np
import pandas as pd
import scipy
import torch

from .data_collators import torch_mask_tokens
from .read import read_h5ad_file
from .tokenization import GeneTokenizer, phenotype_to_token
from configs import BaseConfig, TrainClassificationConfig


class IterableAnnDataset(torch.utils.data.IterableDataset):
    """
    IterableDataset for storing a collection of AnnData objects.
    We only store the filenames to the h5ad files at initialization,
    and the files are only loaded into AnnData objects during __iter__().

    Supports multiple workers, but only up to the number of data shards. This is because
    the shards are currently partitioned across workers, which avoids multiple workers loading the same shard.
    However, the partition gives each worker an integral number of shards, so additional workers will have 0 shards.

    This is an IterableDataset, so it cannot be shuffled by Trainer. The __iter__() function currently
    does not allow shuffling, but shuffling within a shard could be implemented.
    """
    def __init__(self, filenames: list[str], config: BaseConfig):
        """
        Args:
            filenames: paths to h5ad files
            config: any Config class, simplifies the function signature compared to only passing relevant args
        """
        super(IterableAnnDataset).__init__()
        # np.string_ is important because objects get copy-on-access for forked processes.
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        self.byte_filenames = np.fromiter(filenames, object).astype(np.string_)
        self.distributed_state = accelerate.PartialState()

        if config.shard_size is None:
            raise NotImplementedError

        # cumulative_shard_sizes[i + 1] will store the sum of sizes of shards 0 to i.
        self.cumulative_shard_sizes = np.arange(len(self.byte_filenames) + 1) * config.shard_size

        self.config = config
        self.tokenizer = GeneTokenizer(config)

        # if isinstance(self.config, TrainClassificationConfig):  # better type-hints
        if self.config.subcommand == "cls":
            self.cls = True
            with open(config.vocab_path, "r") as f:  # also `self.tokenizer.phenotypic_tokens_map`, but this is cleaner
                self.phenotype_category_labels = json.load(f)[self.config.phenotype_category]

            if self.config.binary_label is None:
                self.label2id = {label: i for i, label in enumerate(self.phenotype_category_labels)}
            else:
                self.label2id = {label: int(label == self.config.binary_label)
                                 for label in self.phenotype_category_labels}
        else:
            self.cls = False

    def __iter__(self):
        """ Generates a dictionary for each example. """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single worker, this case might be included in the multi-worker code already?
            for i, byte_filename in enumerate(self.byte_filenames):
                yield from self._single_shard_generator(byte_filename)
        elif worker_info.id >= len(self.byte_filenames):
            print(f"Warning: More workers than shards, worker {worker_info.id} is idle.")  # TODO: logger
        else:
            # Divide the shards across workers. Might not be evenly balanced right now.
            worker_shard_inds = []
            ideal_shard_size = self.__len__() / worker_info.num_workers
            ideal_start, ideal_end = ideal_shard_size * worker_info.id, ideal_shard_size * (worker_info.id + 1)
            for shard_ind, cumulative_shard_size in enumerate(self.cumulative_shard_sizes):
                # assigning shards based on whether the lower index is above the ideal index
                if cumulative_shard_size >= ideal_start:
                    worker_shard_inds.append(shard_ind)

                # stop whenever the next shard's lower index becomes ideal for the next worker
                if cumulative_shard_size == self.__len__() or self.cumulative_shard_sizes[shard_ind + 1] >= ideal_end:
                    break

            for i in worker_shard_inds:
                yield from self._single_shard_generator(self.byte_filenames[i])

    def __len__(self):
        return self.cumulative_shard_sizes[-1]

    def _single_shard_generator(self, byte_filename):
        """ Yields all the data in a single shard. Shuffling not implemented yet. """
        adata = read_h5ad_file(
            str(byte_filename, encoding="utf-8"),
            self.config.num_top_genes
        )

        # not sure if this is that helpful since X is still sparse
        # also might use too much memory if there are many workers?
        adata = adata.to_memory()
        if not scipy.sparse.issparse(adata.X):
            raise NotImplementedError

        # could iterate on shuffled permutation instead, but remember https://github.com/huggingface/transformers/blob/19e5ed736611227b004c6f55679ce3536db3c28d/src/transformers/trainer_pt_utils.py#L705
        for cell in adata:
            input_ids, token_type_ids = self.tokenizer(cell)
            cell_data = {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
            }

            if self.cls:
                label_name = cell.obs[self.config.phenotype_category].item()
                cell_data["labels"] = torch.tensor(self.label2id[phenotype_to_token(label_name)],
                                                   dtype=torch.long).unsqueeze(0)

            yield cell_data


class EvalJsonDataset(torch.utils.data.Dataset):
    """
    Map-style dataset for storing MLM validation data. This class is specialized for validation data because:
    1) All data is stored in memory, so this scales poorly to big datasets or many workers,
    but the validation dataset should be small. If needed, we can convert this to an IterableDataset instead,
    which would be more efficient memory-wise, but then the same steps taken for the training dataset would
    have to be applied (partitioning `json_filenames` between processes and overriding `Trainer.get_eval_dataloader()`.
    2) We assume the shards are stored in a JSON format instead of h5ad.
    These JSON shards can be generated with scripts in the `data_generation` directory.

    Notably, the data in the shards is already masked, so that we always evaluate on the same, static dataset.
    In contrast, we use dynamic masking for MLM training (with `data_collators.DataCollatorForPhenotypicMLM`).
    IterableAnnDataset can still be used for classification training since no masking is ever applied in that case.
    """
    def __init__(self, json_filenames: list[str], config: BaseConfig):
        """
        Args:
            json_filenames: paths to json validation files
            config: any Config class, simplifies the function signature compared to only passing relevant args
        """
        super().__init__()
        if len(json_filenames) > 3:  # arbitrary, depends on shard size and number of processes
            raise NotImplementedError("Currently storing all data in memory, inefficient implementation.")

        self.config = config
        self.tokenizer = GeneTokenizer(config)

        self.data = []
        for json_filename in json_filenames:
            with open(json_filename, "r") as f:
                self.data.extend(json.load(f)["data"])

        if self.config.subcommand != "mlm":
            raise NotImplementedError("Expected EvalJsonDataset to be used for MLM validation.")

    def __getitem__(self, idx):
        # self.data[idx] has lists as values since we can't store tensors in JSON
        return {key: torch.tensor(val) for key, val in self.data[idx].items()}

    def __len__(self):
        return len(self.data)
