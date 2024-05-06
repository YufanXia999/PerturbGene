#!/bin/bash

# Usage: data_generation/<script_name>.sh (i.e. run from root directory, not this directory)
cd ..  # Go to the parent of the root directory
python -m transformeromics.data_generation.gen_eval_data \
--bin_edges 0.1 \
--shard_size 10000 \
--eval_data_path '/home/shared_folder/TabulaSapiens/ranked/Tabula_Sapiens_ranked_47.h5ad' \
--max_length 1024  \
--num_top_genes 58604 \
--vocab_path 'transformeromics/data/phenotypic_tokens_map.json' \
--included_phenotypes cell_type sex tissue \
--version 0 \
mlm \
--gene_mask_prob 0.15 \
--phenotype_mask_prob 0.5

#python -m transformeromics.data_generation.gen_eval_data \
#--bin_edges 0.1 \
#--shard_size 10000 \
#--eval_data_path '/home/shared_folder/TabulaSapiens/ranked/Tabula_Sapiens_ranked_47.h5ad' \
#--max_length 1024  \
#--num_top_genes 58604 \
#--vocab_path 'transformeromics/data/phenotypic_tokens_map.json' \
#--included_phenotypes cell_type sex tissue \
#--version 0 \
#mlm \
#--gene_mask_prob 0 \
#--phenotype_mask_prob 0.5
