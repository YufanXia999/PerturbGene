from typing import Optional

import datasets
import torch
import transformers
from torch.utils.data import Dataset, DataLoader

from perturbgene.data_utils import IterableAnnDataset, EvalJsonDataset
from perturbgene.data_utils.data_collators import collate_fn_wrapper


class ShardedTrainer(transformers.Trainer):
    """
    Modified Trainer for our use case of distributed training on multiple GPUs with a sharded IterableDataset.
    Instead of loading all batches on GPU 0 and dispatching the batches to other GPUs, each GPU loads its own batches.
    """

    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Copied from Trainer.get_train_dataloader(), only last 2 lines modified.

        Assumes that `self.train_dataset` has already been filtered to the relevant shards for this process.
        In this case, we can directly wrap self.train_dataset in a DataLoader without considering distributed loading.
        """
        assert not self.args.dispatch_batches
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if transformers.is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        assert isinstance(train_dataset, torch.utils.data.IterableDataset), \
            "Assuming IterableDataset, normal Trainer should work for map-style datasets"

        return DataLoader(train_dataset, **dataloader_params)

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Copied from `Trainer.get_eval_dataloader()`.
        Specifically for `eval_dataset`, `EvalJsonDataset` will default to `Trainer.get_eval_dataloader()` otherwise.

        Changing `data_collator` from `self.data_collator` to `collate_fn_wrapper` (i.e. still pad but no masking),
        because masking (if relevant) is already in eval_dataset.
        This change enables deterministic evaluation ("static masking").
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if not isinstance(eval_dataset, EvalJsonDataset):  # default to normal Trainer behavior
            return super().get_eval_dataloader(eval_dataset)

        # Change from `self.data_collator` to `collate_fn_wrapper`
        data_collator = collate_fn_wrapper(eval_dataset.tokenizer)

        if transformers.is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))
