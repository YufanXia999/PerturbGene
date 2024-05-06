#/bin/bash

cd ..
accelerate launch --mixed_precision=fp16 --num_processes=4 --no_python python -m transformeromics.main \
--bin_edges $(< transformeromics/bin_edges_v1.txt) \
--pretrained_model_path transformeromics/model_configs/distilbert.json \
--shard_size 10000 \
--eval_data_path "transformeromics/ranked/Tabula_Sapiens_ranked_47.h5ad" \
--max_length 10000 \
--num_top_genes 58604 \
--vocab_path "transformeromics/data/phenotypic_tokens_map.json" \
--included_phenotypes cell_type development_stage sex tissue \
--use_flash_attn \
--per_device_eval_batch_size 32 \
--dataloader_num_workers 4 \
--auto_find_batch_size \
--use_fp16 \
--output_dir output_4gpu_mlm015_v4 \
mlm \
--mlm_probability 0.15 \
--phenotypic_mlm_probability 0.5 \
--train_data_path "transformeromics/ranked/Tabula_Sapiens_ranked_{0..43}.h5ad" \
--num_hidden_layers 12 \
--num_attention_heads 12 \
--per_device_train_batch_size 8 \
--learning_rate 2e-5 \
--warmup_ratio 0.1 \
--num_train_epochs 20 \
--eval_steps 5000 \
--save_steps 5000

#--pretrained_model_path transformeromics/model_configs/distilbert.json \
#--pretrained_model_path "transformeromics/output_4gpu_mlm015_v1_10000_mlm_bins10/tmp-checkpoint-4000" \
