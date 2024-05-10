cd ..
CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES=1 accelerate launch --mixed_precision=fp16 --num_processes=1 \
--num_machines 1 --dynamo_backend no \
-m perturbgene.main \
--bin_edges 0.1 \
--pretrained_model_path 'perturbgene/model_configs/distilbert_base.json' \
--shard_size 10000 \
--eval_data_paths '/home/shared_folder/TabulaSapiens/ranked/Tabula_Sapiens_ranked_47.h5ad' \
--max_length 130 \
--num_top_genes 128 \
--vocab_path 'perturbgene/data/phenotypic_tokens_map.json' \
--use_flash_attn \
--per_device_eval_batch_size 256 \
--dataloader_num_workers 4 \
--output_dir 'output_cls_lung_base_v2' \
cls \
--phenotype_category 'tissue' \
--binary_label '[lung]' \
--train_data_paths '/home/shared_folder/TabulaSapiens/ranked/Tabula_Sapiens_ranked_{0..43}.h5ad' \
--num_hidden_layers 12 \
--num_attention_heads 12 \
--per_device_train_batch_size 512 \
--learning_rate 1e-4 \
--weight_decay 0.1 \
--warmup_ratio 0.1 \
--num_train_epochs 10 \
--eval_steps 1000 \
--save_steps 4000
