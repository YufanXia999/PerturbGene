import argparse
import json
import os

from braceexpand import braceexpand
from tqdm.auto import tqdm

from data_utils import IterableAnnDataset
from data_utils.data_collators import torch_mask_tokens
from eval_utils import set_seed


def parse_data_args(args: list[str] | None = None) -> argparse.Namespace:
    """
    Essentially the subset of `configs.parse_args` needed to generate the validation dataset,
    and `version` (currently plan is to change it when binning method changes).

    Args:
        args: If specified, will parse these `args`. Otherwise, defaults to `sys.argv`.
    """
    parser = argparse.ArgumentParser(description="Arguments for generating validation dataset.")
    subparsers = parser.add_subparsers(help="Sub-command help", dest="subcommand")

    mlm_subparser = subparsers.add_parser("mlm", description="For training with an MLM objective.")
    # cls_subparser = subparsers.add_parser("cls", description="For training with a classification objective.")

    parser.add_argument("--bin_edges", nargs="+", type=float,
                        help="Provided `n` edges, will partition the gene expression values into `n + 1` bins "
                             "around those edges. These edges are shared across all genes.")
    parser.add_argument("--bin_edges_path", type=str,
                        help="Path to file [format TBD]. Allows specifying different edges for each gene.")
    parser.add_argument("--shard_size", default=10000, type=int,
                        help="The number of observations in each AnnData shard.")
    parser.add_argument("--eval_data_path", type=str, required=True,
                        help="`braceexpand`-able path to validation h5ad file(s).")  # dir_path might be less confusing than braceexpand str
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
    parser.add_argument("--version", type=int, required=True,
                        help=("The number which will be appended to the output file name, useful for differentiating "
                              "between other parameters like binning."))
    mlm_subparser.add_argument("--gene_mask_prob", type=float, required=True,
                               help="The probability a gene token will be masked.")
    mlm_subparser.add_argument("--phenotype_mask_prob", type=float, required=True,
                               help="The probability a phenotype token will be masked.")

    # cls_subparser.add_argument("--label_key", type=str, required=True,
    #                            choices=["cell_type", "developmental_stage", "disease", "sex", "tissue"],
    #                            help="The name of the observation key we are training to predict.")

    return parser.parse_args(args)


def check_args(args: argparse.Namespace) -> None:
    """
    Sanity checks on `args`, largely copied from __post_init__ of Config classes.
    """
    if args.included_phenotypes is None:
        args.included_phenotypes = []

    assert (args.bin_edges is not None) ^ (args.bin_edges_path is not None), \
        "Expected exactly one of --bin_edges and --bin_edges_path to be specified."

    if args.bin_edges is not None:
        assert min(args.bin_edges) >= 0, "Assuming no expressions will fall strictly below the lowest bin_edge."

    assert args.max_length <= args.num_top_genes + 2
    for path in (args.bin_edges_path, args.vocab_path):
        assert path is None or os.path.isfile(path), f"{path=} is not a file."

    # could also try opening to check it's a JSON file
    assert args.vocab_path.lower().endswith(".json"), f"Expected json path but got {args.vocab_path=}"


if __name__ == "__main__":
    set_seed(42)
    args = parse_data_args()
    check_args(args)

    if args.subcommand == "mlm":
        output_prefix = f"seq{args.max_length}_mlm_gene{args.gene_mask_prob*100:03.0f}_pheno{args.phenotype_mask_prob*100:03.0f}"
    else:
        raise NotImplementedError("Need to differentiate between 'labels' when it is a classification vs. MLM target.")

    # Eval dataset
    eval_paths = list(braceexpand(args.eval_data_path))
    eval_dataset = IterableAnnDataset(eval_paths, args)

    curr_shard: list[dict]
    all_shards: list[list[dict]] = []
    for i, single_cell in enumerate(tqdm(eval_dataset)):
        if i % args.shard_size == 0:
            curr_shard = []
            all_shards.append(curr_shard)

        # Perform masking here. `torch_mask_tokens` expects a batch. (TODO: more elegant, could just change collator)
        single_cell_batch = {key: val.unsqueeze(0) for key, val in single_cell.items()}
        torch_mask_tokens(single_cell_batch, eval_dataset.tokenizer, args.gene_mask_prob, args.phenotype_mask_prob)
        curr_shard.append({key: val.squeeze(0).tolist() for key, val in single_cell_batch.items()})  # list for JSON

    for shard_ind, curr_shard in enumerate(all_shards):
        with open(f"perturbgene/data/validation_data/{output_prefix}_shard{shard_ind}_v{args.version}.json",
                  "w") as f:
            json.dump({
                "metadata": vars(args),
                "data": curr_shard,
            }, f)
