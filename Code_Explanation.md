# Code Organization - March 11

## Setup notes:
To allow absolute imports, we treat `perturbgene` as a module and require all files 
to be run from the parent of the `perturbgene` directory,
as `python -m perturbgene.<script_name>`. Two example usages in `train_cls.sh` and `train_mlm.sh`.

For more usage details, see `python -m perturbgene.main -h`.


## Important files:
### `data_utils/tokenization.py`
Implements `GeneTokenizer`, which can tokenize individual `AnnData` cells. 
A cell is represented by three kinds of tokens:
- Special tokens (currently `"[CLS]", "[EOS]", "[MASK]", "[PAD]"`)
- Gene expression tokens
- Optionally, phenotypic tokens

The sequence consists of both `input_ids` and `token_type_ids`. 
There is a distinct "input_id" for each special/phenotypic token, and one "input_id" for each gene expression bin.
The individual genes are distinguished by the "token_type_id", 
with each gene corresponding to a unique "token_type_id", 
and 0 is reserved as the "token_type_id" for all special tokens. 
Furthermore, there is also a unique "token_type_id" for each phenotype category, 
because even though a phenotype can be fully determined by the "input_id",
the category becomes ambiguous is the phenotype token is masked during MLM.

Note that only expressed genes are included in the sequence, which greatly shortens the sequence length.
This also allows us to create the sequence without ever converting the gene expression matrix to a dense representation.

### `data_utils/iterable_ann_dataset.py`
Implements `IterableAnnDataset`, which yields tokenized cell data as a dictionary of tensors, 
including labels if performing classification.
Supports multiple workers, where each worker is assigned an integral number of shards.
This assignment makes it pointless to have more workers than shards.

### `data_utils/data_collators.py`
Implements `collate_fn_wrapper()` and `DataCollatorForPhenotypicMLM`.

`collate_fn_wrapper()` returns a function that can operate on a list of examples yielded by `IterableAnnDataset`, 
and returns a single dictionary of tensors, where the example sequences are padded to the same length.

`DataCollatorForPhenotypicMLM.__call__()` uses `collate_fn_wrapper()` to pad examples,
and then randomly masks some phenotypic and gene expression tokens.

### `model.py`

Implements `PositionlessEmbeddings`, `GeneBertModel`, `GeneBertForPhenotypicMLM`, and `GeneBertForClassification`, 
which roughly correspond to `distilbert.Embeddings`, `DistilBertModel`, `DistilBertForMaskedLM`, 
and `DistilBertForSequenceClassification` respectively, but with `token_type_ids` and no positional embeddings. 

### `configs.py`
Implements `parse_args()` and various Config classes.

`parse_args()` parses the commmand line arguments into a subclass of `BaseConfig`.
There are currently only two options, for training on MLM and training on classification tasks.

### `main.py`
The main script, see [Setup notes](#setup-notes).

## Utility files:

### `sharded_trainer.py`

### `eval_utils.py`


## Vast.ai Setup:

Starting in the `perturbgene` directory,
```bash
sudo apt-get install gcc g++
sudo apt-get install nano

# Download data in background
chmod +x download_dataset.sh 
./download_dataset.sh &

# Python environment
python -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt

wandb login

# Flash Attention steps
# Takes a long time: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local
pip install packaging ninja wheel
pip install flash-attn --no-build-isolation
```
