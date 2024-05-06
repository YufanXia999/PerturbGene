"""
Supports training (either MLM or classification). Inference not implemented yet.
Training works across multiple GPUs on a single machine.
"""
import json
import os

import accelerate
import torch
import torch.nn as nn
import transformers
import wandb

from configs import parse_args, TrainConfig, TrainMLMConfig
from data_utils import (
    DataCollatorForPhenotypicMLM, GeneTokenizer, IterableAnnDataset, EvalJsonDataset, collate_fn_wrapper
)
from eval_utils import preprocess_logits_argmax, mlm_metrics_wrapper, cls_metrics_wrapper, set_seed
from model import GeneBertForPhenotypicMLM, GeneBertForClassification
from sharded_trainer import ShardedTrainer


if __name__ == "__main__":
    ##################################################
    #           General code for all tasks           #
    ##################################################
    set_seed(42)

    # config = parse_args()
    config = TrainMLMConfig(
        subcommand='mlm', bin_edges=[0.1], bin_edges_path=None,
        pretrained_model_path='model_configs/distilbert_base.json', model_arch=None, shard_size=10000,
        eval_data_paths=['/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_47.h5ad'], max_length=1024, num_top_genes=58604,
        vocab_path='data/phenotypic_tokens_map.json',
        included_phenotypes=['cell_type', 'sex', 'tissue'],
        use_flash_attn=True, output_dir='output_v10_base_gene000',
        per_device_eval_batch_size=256, dataloader_num_workers=4, auto_find_batch_size=False, train_data_paths=['/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_0.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_1.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_2.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_3.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_4.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_5.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_6.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_7.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_8.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_9.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_10.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_11.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_12.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_13.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_14.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_15.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_16.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_17.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_18.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_19.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_20.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_21.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_22.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_23.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_24.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_25.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_26.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_27.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_28.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_29.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_30.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_31.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_32.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_33.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_34.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_35.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_36.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_37.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_38.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_39.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_40.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_41.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_42.h5ad', '/home/rohola/TabulaSapiens/ranked/Tabula_Sapiens_ranked_43.h5ad'], num_hidden_layers=12, num_attention_heads=12, num_train_epochs=10, per_device_train_batch_size=128, gradient_accumulation_steps=1, learning_rate=0.0001, weight_decay=0.05, warmup_ratio=0.1, save_steps=8000, eval_steps=1000, gene_mask_prob=0.0, phenotype_mask_prob=0.5)


    distributed_state = accelerate.PartialState()

    tokenizer = GeneTokenizer(config)

    eval_path_exts = {os.path.splitext(eval_path)[1].lower() for eval_path in config.eval_data_paths}
    if eval_path_exts == {".json"}:
        eval_dataset = EvalJsonDataset(config.eval_data_paths, config)
    elif eval_path_exts == {".h5ad"}:
        eval_dataset = IterableAnnDataset(config.eval_data_paths, config)
    else:
        raise NotImplementedError(f"Unexpected extension(s): {config.eval_data_paths=}")

    ##################################################
    #               Task-specific code               #
    ##################################################
    if config.subcommand in ["mlm", "cls"]:  # Training
        config: TrainConfig = config  # cast for PyCharm type hints

        # Haven't actually checked if .update() is the same as the normal dict update
        os.environ.update({  # https://docs.wandb.ai/guides/track/environment-variables
            "WANDB_PROJECT": "genomics",
            "WANDB_NAME": f"A100_{config.max_length}_MLM{config.gene_mask_prob*100:03.0f}_2bins_Base_v4"
            if config.subcommand == "mlm"
            else f"A100_{config.max_length}_CLS_{config.phenotype_category}_"
                 f"{config.binary_label[1:-1] if config.binary_label is not None else None}_Base_v0",
            "WANDB_LOG_MODEL": "false",
            # "WANDB_RUN_GROUP": "cls-vasc-A100",
            "WANDB_RUN_GROUP": "mlm-tests-edelman",
            "WANDB-NOTES": "0.5 label smoothing, weighted classes",
        })

        # Divide `config.train_data_paths` across the processes.
        # Currently assuming that `len(train_paths)` is a perfect multiple of num_processes to make this easier.
        if len(config.train_data_paths) % distributed_state.num_processes != 0:
            raise NotImplementedError
        shards_per_process = len(config.train_data_paths) // distributed_state.num_processes
        rank = distributed_state.process_index
        process_train_paths = config.train_data_paths[shards_per_process * rank: shards_per_process * (rank + 1)]

        train_dataset = IterableAnnDataset(process_train_paths, config)

        working_dir = f"{config.output_dir}_{config.max_length}_{config.subcommand}_bins{len(config.bin_edges)}"

        # Load the configuration for a model
        if config.pretrained_model_path is not None:  # load a model with pre-trained weights
            model_config = transformers.AutoConfig.from_pretrained(config.pretrained_model_path)
        else:  # initialize a model with random weights
            model_config = transformers.AutoConfig.for_model(
                config.model_arch)  # probably want to pass this in as a file?

        model_config.num_hidden_layers = config.num_hidden_layers
        model_config.num_attention_heads = config.num_attention_heads
        model_config.max_position_embeddings = config.max_length
        model_config.vocab_size = tokenizer.vocab_size  # vocab_size and type_vocab_size determine model embedding sizes
        model_config.type_vocab_size = tokenizer.gene_token_type_offset + config.num_top_genes  # TODO: less hardcoding
        model_config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        model_kwargs = {"attn_implementation": "flash_attention_2"} if config.use_flash_attn else dict()

        with open("data/phenotype_tokens_freqs.json", "r") as f:  # TODO: don't hardcode
            phenotype_tokens_freqs = json.load(f)

        # Load relevant `model`, `data_collator`, and ` eval_metric`
        if config.subcommand == "mlm":
            if len(config.included_phenotypes) > 0:
                weights_per_class = torch.zeros(tokenizer.vocab_size)
                # On average, a sequence will have `len(config.included_phenotypes) * config.phenotype_mask_prob` masked
                # phenotype tokens and approximately `config.max_length * config.gene_mask_prob` masked genes
                # (approximation because there are special/phenotype tokens, like padding)
                GENE_WEIGHT = 1 / (config.max_length * config.gene_mask_prob) if config.gene_mask_prob > 0 else 0
                PHENOTYPE_WEIGHT = 1 / (len(config.included_phenotypes) * config.phenotype_mask_prob)
                weights_per_class[tokenizer.get_phenotypic_tokens_mask(torch.arange(tokenizer.vocab_size))] \
                    = PHENOTYPE_WEIGHT
                weights_per_class[tokenizer.get_gene_tokens_mask(torch.arange(tokenizer.vocab_size))] = GENE_WEIGHT
                mlm_loss_fct = nn.CrossEntropyLoss(weight=weights_per_class, label_smoothing=0.5)
            else:
                mlm_loss_fct = nn.CrossEntropyLoss(label_smoothing=0.5)

            model = GeneBertForPhenotypicMLM._from_config(model_config, mlm_loss_fct=mlm_loss_fct, torch_dtype=torch.float16, **model_kwargs)
            data_collator = DataCollatorForPhenotypicMLM(
                tokenizer=tokenizer, mlm=True, mlm_probability=config.gene_mask_prob,
                phenotype_mask_prob=config.phenotype_mask_prob
            )

            eval_metric = mlm_metrics_wrapper(tokenizer, working_dir)
        elif config.subcommand == "cls":
            model_config.num_labels = len(set(train_dataset.label2id.values()))
            model_config.problem_type = "single_label_classification"  # should be inferred anyway

            phenotype_cat_freqs = phenotype_tokens_freqs[config.phenotype_category]
            pos_freq = phenotype_cat_freqs[config.binary_label]
            if config.binary_label is not None:
                cls_loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1 / (1 - pos_freq), 1 / pos_freq],
                                                                       dtype=torch.float), label_smoothing=0.1 * pos_freq)
            else:
                cls_loss_fct = nn.CrossEntropyLoss(
                    weight=torch.tensor([1 / phenotype_cat_freqs[label]
                                         for label in train_dataset.phenotype_category_labels],
                                        dtype=torch.float), label_smoothing=0.1 * pos_freq
                )

            model = GeneBertForClassification._from_config(model_config, cls_loss_fct=cls_loss_fct, **model_kwargs)

            # Trainer expects a `DataCollator`, but it seems that any callable works, so using a function instead
            # TODO: implement `GeneTokenizer.pad()` and pass to Trainer to get default as DataCollatorWithPadding?
            data_collator = collate_fn_wrapper(tokenizer)
            eval_metric = cls_metrics_wrapper(tokenizer, working_dir)
        else:
            raise NotImplementedError

        # TODO: shuffle train, add shard_sizes arg, diff shard_size for eval

        model_total_params = sum(p.numel() for p in model.parameters())
        if rank == 0:  # accelerator.print() would be nice, but don't have accelerator outside of Trainer
            print(f"Total number of parameters: {model_total_params}")

        training_args = transformers.TrainingArguments(
            output_dir=working_dir,
            overwrite_output_dir=True,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            logging_dir=working_dir,
            dataloader_num_workers=config.dataloader_num_workers,
            logging_steps=50,
            save_strategy="steps",
            save_steps=config.save_steps,
            save_total_limit=5,
            evaluation_strategy="steps",
            eval_steps=config.eval_steps,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            eval_accumulation_steps=25,  # TODO: don't hardcode
            logging_first_step=True,
            load_best_model_at_end=True,
            metric_for_best_model="f1_overall" if config.subcommand == "mlm" else "f1",
            report_to=["wandb"],
            auto_find_batch_size=config.auto_find_batch_size,
            accelerator_config={"dispatch_batches": False},
            dataloader_pin_memory=True,
            dataloader_persistent_workers=True,
            fp16=True,
        )

        trainer = ShardedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=eval_metric,
            preprocess_logits_for_metrics=preprocess_logits_argmax,
        )

        # Train the model
        trainer.train()

        # Evaluate the model
        trainer.evaluate()

        wandb.finish()  # does Trainer not handle this?
    else:
        raise NotImplementedError
