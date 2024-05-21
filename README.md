# Transformeromics

## Setup:

### General notes:
To allow absolute imports, we treat `perturbgene` as a module and require all files 
to be run from the parent of the `perturbgene` directory,
as `python -m perturbgene.<script_name>`. 

Example usages in `run_training.sh` and `run_bincls_training.sh`. For more usage details, see `python -m perturbgene.main -h`.

### Vast.ai Setup:

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

# Flash Attention steps
# Takes a long time: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local
pip install packaging ninja wheel
pip install flash-attn --no-build-isolation

wandb login  # enter API key
```

### PyCharm Setup
Go to Run/Debug Configurations. 
Change it to resemble ![PyCharm Config](./imgs/PyCharm_Config.png)

My Script parameters are: ```--mixed_precision=fp16 --num_processes=1 --num_machines 1 --dynamo_backend no main.py --bin_edges 0.1 --pretrained_model_path perturbgene/model_configs/distilbert_base.json --shard_size 10000 --eval_data_paths /home/shared_folder/TabulaSapiens/ranked/Tabula_Sapiens_ranked_47.h5ad --max_length 1024 --num_top_genes 58604 --vocab_path perturbgene/data/phenotypic_tokens_map.json --included_phenotypes cell_type sex tissue --use_flash_attn --per_device_eval_batch_size 256 --dataloader_num_workers 4 --output_dir output_v10_base_gene000 mlm --gene_mask_prob 0.00 --phenotype_mask_prob 0.5 --train_data_paths /home/shared_folder/TabulaSapiens/ranked/Tabula_Sapiens_ranked_{0..43}.h5ad --num_hidden_layers 12 --num_attention_heads 12 --per_device_train_batch_size 128 --learning_rate 1e-4 --weight_decay 5e-2 --warmup_ratio 0.1 --num_train_epochs 10 --eval_steps 1000 --save_steps 8000```

My Environment variables are: ```LAUNCH_FROM_PYCHARM=1;PYTHONUNBUFFERED=1;CUDA_DEVICE_ORDER=PCI_BUS_ID;CUDA_VISIBLE_DEVICES=1``` 
(most important one is `LAUNCH_FROM_PYCHARM=1`)

You should not need to worry about the Python Interpreter, Working directory, or Path mappings when running locally.

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
