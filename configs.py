"""
Parses user arguments into the relevant Config class.
The Config classes are unnecessary but help for PyCharm type hints.
This comes with the cost that any update to the arguments requires changing both the parser and a Config class.
"""

import argparse
import json
import os
from dataclasses import dataclass

from braceexpand import braceexpand


@dataclass
class BaseConfig:
    subcommand: str
    bin_edges: list[float]
    bin_edges_path: str
    pretrained_model_path: str
    model_arch: str
    shard_size: int
    eval_data_paths: list[str]
    max_length: int
    num_top_genes: int

    vocab_path: str
    included_phenotypes: list[str]  # TODO: assert choices

    use_flash_attn: bool

    output_dir: str
    per_device_eval_batch_size: int
    dataloader_num_workers: int
    # use_fp16: bool
    auto_find_batch_size: bool

    def __post_init__(self):
        """ Postprocessing and basic sanity checks """

        assert (self.bin_edges is not None) ^ (self.bin_edges_path is not None), \
            "Expected exactly one of --bin_edges and --bin_edges_path to be specified."

        if self.bin_edges is not None:
            assert min(self.bin_edges) >= 0, "Assuming no expressions will fall strictly below the lowest bin_edge."

        assert (self.pretrained_model_path is not None) ^ (self.model_arch is not None), \
            "Expected exactly one of --pretrained_model_path and --model_arch to be specified."
        for path in (self.bin_edges_path, self.vocab_path):
            assert path is None or os.path.isfile(path), f"{path=} is not a file."

        assert path is None or os.path.exists(self.pretrained_model_path)

        # could also try opening to check it's a JSON file
        assert self.vocab_path.lower().endswith(".json"), f"Expected json path but got {self.vocab_path=}"
        if self.included_phenotypes is None:
            self.included_phenotypes = []

        assert self.max_length <= self.num_top_genes + 2 + len(self.included_phenotypes)
        self.eval_data_paths = [path for paths in self.eval_data_paths for path in braceexpand(paths)]


@dataclass
class TrainConfig(BaseConfig):
    train_data_paths: list[str]
    num_hidden_layers: int
    num_attention_heads: int

    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    save_steps: int
    eval_steps: int

    def __post_init__(self):
        super().__post_init__()
        self.train_data_paths = [path for paths in self.train_data_paths for path in braceexpand(paths)]


@dataclass
class TrainMLMConfig(TrainConfig):
    gene_mask_prob: float
    phenotype_mask_prob: float


@dataclass
class TrainClassificationConfig(TrainConfig):
    phenotype_category: str
    binary_label: str
    balanced_dataset: bool

    def __post_init__(self):
        """ Classification-specific sanity checks """
        super().__post_init__()
        assert self.phenotype_category not in self.included_phenotypes, "Trivial classification."
        with (open(self.vocab_path, "r") as f):
            phenotypic_tokens_map = json.load(f)
            assert self.phenotype_category in phenotypic_tokens_map, \
                f"Unexpected {self.phenotype_category=} not found in {phenotypic_tokens_map.keys()}"


def parse_args(args: list[str] = None) -> BaseConfig:
    """
    Get command line arguments.

    Optional arguments to `parse_args`:
        args: If specified, will parse these `args`. Otherwise, defaults to `sys.argv`.
    """
    parser = argparse.ArgumentParser(description="Arguments for training/evaluating a model. Any arguments without "
                                                 "descriptions correspond directly to transformer.TrainingArguments.")
    subparsers = parser.add_subparsers(help="Sub-command help", dest="subcommand")

    mlm_subparser = subparsers.add_parser("mlm", description="Training with an MLM objective.")
    cls_subparser = subparsers.add_parser("cls", description="Training with a classification objective.")
    # eval_subparser = subparsers.add_parser("eval", description="Inference")

    train_subparsers = [mlm_subparser, cls_subparser]

    parser.add_argument("--bin_edges", nargs="+", type=float,
                        help="Provided `n` edges, will partition the gene expression values into `n + 1` bins "
                             "around those edges. These edges are shared across all genes.")
    parser.add_argument("--bin_edges_path", type=str,
                        help="Path to file [format TBD]. Allows specifying different edges for each gene.")
    parser.add_argument("--pretrained_model_path", type=str, help="Path to a pretrained model to initialize from.")
    parser.add_argument("--model_arch", type=str, help="The model architecture to initialize f, like `bert`.")
    parser.add_argument("--shard_size", default=10000, type=int,
                        help="The number of observations in each AnnData shard.")
    parser.add_argument("--eval_data_paths", nargs="+", type=str,
                        help="One or more (possibly `braceexpand`-able) paths to validation h5ad file(s).") # dir_path might be less confusing than braceexpand str
    parser.add_argument("--max_length", type=int, required=True,
                        help="The maximum sequence length for the transformer."
                             "All sequences will be exactly this length, so no padding is needed.")
    parser.add_argument("--num_top_genes", type=int, required=True,
                        help="The number of highest variable genes to use.")

    parser.add_argument("--vocab_path", type=str, required=True,
                        help="Path to the JSON file mapping token_types to tokens.")
    parser.add_argument("--included_phenotypes", nargs="*", type=str,
                        help="The phenotypes to include in the model input. The current types are: "
                             "'cell_type sex tissue'. (Also 'development_stage' and 'disease' but not recommended.)")

    parser.add_argument("--use_flash_attn", action="store_true",
                        help="Whether to use Flash Attention 2. If true, also expects `fp16`.")

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--per_device_eval_batch_size", type=int, required=True)
    parser.add_argument("--dataloader_num_workers", default=4, type=int)
    parser.add_argument("--auto_find_batch_size", action="store_true")

    for train_subparser in train_subparsers:
        train_subparser.add_argument("--train_data_paths", nargs="+", type=str,
                                     help="One or more (possible `braceexpand`-able) paths to training h5ad file(s).")
        train_subparser.add_argument("--num_hidden_layers", type=int, required=True,
                                     help="Number of hidden layers to use in the model.")
        train_subparser.add_argument("--num_attention_heads", type=int, required=True,
                                     help="Number of attention heads to use in the model.")

        train_subparser.add_argument("--num_train_epochs", type=int, required=True)
        train_subparser.add_argument("--per_device_train_batch_size", type=int, required=True)
        train_subparser.add_argument("--gradient_accumulation_steps", default=1, type=int)
        train_subparser.add_argument("--learning_rate", type=float, required=True)
        train_subparser.add_argument("--weight_decay", type=float, required=True)
        train_subparser.add_argument("--warmup_ratio", default=0.0, type=float)
        train_subparser.add_argument("--save_steps", type=int, required=True)
        train_subparser.add_argument("--eval_steps", type=int, required=True)

    mlm_subparser.add_argument("--gene_mask_prob", type=float, required=True,
                               help="The probability a gene token will be masked.")
    mlm_subparser.add_argument("--phenotype_mask_prob", type=float, required=True,
                               help="The probability a phenotype token will be masked.")

    cls_subparser.add_argument("--phenotype_category", type=str, required=True,
                               choices=["cell_type", "developmental_stage", "disease", "sex", "tissue"],
                               help="The name of the observation key we are training to predict.")
    cls_subparser.add_argument("--binary_label", type=str,
                               help="If specified, this would be one of the labels for the `phenotype_category`. "
                                    "Then, binary classification (one vs. rest) will be performed "
                                    "instead of multi-class classification.")
    cls_subparser.add_argument("--balanced_dataset", action="store_true",
                               help="If specified, will not weight classes in cross entropy loss "
                                    "since assuming they are already balanced.")

    args = parser.parse_args(args)
    if args.subcommand is None:
        return BaseConfig(**vars(args))

    if args.subcommand == "mlm":
        return TrainMLMConfig(**vars(args))
    elif args.subcommand == "cls":
        return TrainClassificationConfig(**vars(args))
    else:
        raise NotImplementedError
