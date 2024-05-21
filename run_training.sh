cd ..
CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES=1 accelerate launch --mixed_precision=fp16 --num_processes=1 \
--num_machines 1 --dynamo_backend no \
-m perturbgene.main \
--bin_edges 0.1 \
--pretrained_model_path 'perturbgene/model_configs/distilbert_base.json' \
--shard_size 10000 \
--eval_data_paths 'perturbgene/data/validation_data/' \
--max_length 128 \
--num_top_genes 58604 \
--vocab_path 'perturbgene/data/phenotypic_tokens_map.json' \
--included_phenotypes cell_type sex tissue \
--use_flash_attn \
--per_device_eval_batch_size 1024 \
--dataloader_num_workers 4 \
--output_dir 'output_v2_base' \
mlm \
--gene_mask_prob 0.15 \
--phenotype_mask_prob 0.5 \
--train_data_paths '/home/shared_folder/TabulaSapiens/ranked/Tabula_Sapiens_ranked_{0..43}.h5ad' \
--num_hidden_layers 12 \
--num_attention_heads 12 \
--per_device_train_batch_size 512 \
--learning_rate 1e-4 \
--weight_decay 5e-2 \
--warmup_ratio 0.1 \
--num_train_epochs 10 \
--eval_steps 1000 \
--save_steps 8000

#cd ..
#CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES=1 accelerate launch --mixed_precision=fp16 --num_processes=1 \
#--num_machines 1 --dynamo_backend no \
#-m perturbgene.main \
#--bin_edges 0.1 \
#--pretrained_model_path 'perturbgene/model_configs/distilbert_base.json' \
#--shard_size 10000 \
#--eval_data_paths '/home/shared_folder/TabulaSapiens/ranked/Tabula_Sapiens_ranked_47.h5ad' \
#--max_length 1024 \
#--num_top_genes 58604 \
#--vocab_path 'perturbgene/data/phenotypic_tokens_map.json' \
#--included_phenotypes cell_type sex tissue \
#--use_flash_attn \
#--per_device_eval_batch_size 256 \
#--dataloader_num_workers 4 \
#--output_dir 'output_v10_base_gene000' \
#mlm \
#--gene_mask_prob 0.00 \
#--phenotype_mask_prob 0.5 \
#--train_data_paths '/home/shared_folder/TabulaSapiens/ranked/Tabula_Sapiens_ranked_{0..43}.h5ad' \
#--num_hidden_layers 12 \
#--num_attention_heads 12 \
#--per_device_train_batch_size 128 \
#--learning_rate 1e-4 \
#--weight_decay 5e-2 \
#--warmup_ratio 0.1 \
#--num_train_epochs 10 \
#--eval_steps 1000 \
#--save_steps 8000

#cd ..
#CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES=1 accelerate launch --mixed_precision=fp16 --num_processes=1 \
#--num_machines 1 --dynamo_backend no \
#-m perturbgene.main \
#--bin_edges 0.1 \
#--pretrained_model_path 'perturbgene/model_configs/distilbert_large.json' \
#--shard_size 10000 \
#--eval_data_paths 'perturbgene/data/validation_data/seq1024_mlm_gene000_pheno050_shard0_v0.json' 'perturbgene/data/validation_data/seq1024_mlm_gene015_pheno050_shard0_v0_no_pheno.json' \
#--max_length 1024 \
#--num_top_genes 58604 \
#--vocab_path 'perturbgene/data/phenotypic_tokens_map.json' \
#--included_phenotypes cell_type sex tissue \
#--use_flash_attn \
#--per_device_eval_batch_size 256 \
#--dataloader_num_workers 4 \
#--output_dir output_v9_large_gene015 \
#mlm \
#--gene_mask_prob 0.15 \
#--phenotype_mask_prob 0.5 \
#--train_data_paths '/home/shared_folder/TabulaSapiens/ranked/Tabula_Sapiens_ranked_{0..43}.h5ad' \
#--num_hidden_layers 24 \
#--num_attention_heads 16 \
#--per_device_train_batch_size 64 \
#--gradient_accumulation_steps 2 \
#--learning_rate 5e-5 \
#--weight_decay 5e-2 \
#--warmup_ratio 0.1 \
#--num_train_epochs 10 \
#--eval_steps 1500 \
#--save_steps 15000
