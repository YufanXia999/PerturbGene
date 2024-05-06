# Validation Dataset Generation

Code for generating masked validation data, so that each evaluation iteration uses the same masking.
This improves the consistency and interpretation of the evaluation metrics within a run and between runs.

## Usage: 
```shell
python -m transformeromics.data_generation.gen_eval_data --help
usage: gen_eval_data.py [-h] [--bin_edges BIN_EDGES [BIN_EDGES ...]]
                        [--bin_edges_path BIN_EDGES_PATH]
                        [--shard_size SHARD_SIZE] --eval_data_path
                        EVAL_DATA_PATH --max_length MAX_LENGTH --num_top_genes
                        NUM_TOP_GENES --vocab_path VOCAB_PATH
                        [--included_phenotypes [INCLUDED_PHENOTYPES ...]]
                        --version VERSION
                        {mlm} ...

Arguments for generating validation dataset.

positional arguments:
  {mlm}                 Sub-command help

options:
  -h, --help            show this help message and exit
  --bin_edges BIN_EDGES [BIN_EDGES ...]
                        Provided `n` edges, will partition the gene expression
                        values into `n + 1` bins around those edges. These
                        edges are shared across all genes.
  --bin_edges_path BIN_EDGES_PATH
                        Path to file [format TBD]. Allows specifying different
                        edges for each gene.
  --shard_size SHARD_SIZE
                        The number of observations in each AnnData shard.
  --eval_data_path EVAL_DATA_PATH
                        `braceexpand`-able path to validation h5ad file(s).
  --max_length MAX_LENGTH
                        The maximum sequence length for the transformer.All
                        sequences will be exactly this length, so no padding
                        is needed.
  --num_top_genes NUM_TOP_GENES
                        The number of highest variable genes to use.
  --vocab_path VOCAB_PATH
                        Path to the JSON file mapping token_types to tokens.
  --included_phenotypes [INCLUDED_PHENOTYPES ...]
                        The phenotypes to include in the model input. The
                        current types are: 'cell_type sex tissue'. (Also
                        'development_stage' and 'disease' but not
                        recommended.)
  --version VERSION     The number which will be appended to the output file
                        name, useful for differentiating between other
                        parameters like binning.
```
The run `data_generation/mlm.sh`
from the root directory (parent of this `data_generation/` directory).



