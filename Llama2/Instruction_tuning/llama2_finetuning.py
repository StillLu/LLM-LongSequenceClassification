import os
import string
import logging
from transformers import LlamaTokenizerFast
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, GenerationConfig, LlamaForSequenceClassification
import torch
import datasets
from peft import LoraConfig,prepare_model_for_kbit_training,get_peft_model
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,TrainingArguments
from trl import SFTTrainer

import argparse
import numpy as np
import pandas as pd
import pickle


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)





def main(args):

    train = pd.read_csv(os.path.join(args.data_dir, "train_2.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test_2.csv"))

    train_data = train["text"]
    train_labels = train["label"]

    test_data = test["text"]
    test_labels = test["label"]

    prompt = "Record:{sentence} Diagnose:{label}"
    train_instructions = []

    for text, label in zip(train_data, train_labels):
        example = prompt.format(sentence=text, label=label)
        train_instructions.append(example)

    train_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(
            data={
                "instructions": train_instructions,
                "labels": train_labels,
            }
        )
    )

    prompt = "Record:{sentence} Diagnose:"
    test_instructions = []
    for text, label in zip(test_data, test_labels):
        example = prompt.format(sentence=text, )
        test_instructions.append(example)

    test_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(
            data={
                "instructions": test_instructions,
                "labels": test_labels,
            }
        )
    )

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_ckpt,
        quantization_config=bnb_config,
        use_cache=False,
        device_map="auto",
    )
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_ckpt)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=args.dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    results_dir = args.data_dir

    training_args = TrainingArguments(
        output_dir=results_dir,
        logging_dir=f"{results_dir}/logs",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=6,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=100,
        learning_rate=5e-4,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="none",
        gradient_checkpointing_kwargs={"use_reentrant": True}

        # disable_tqdm=True # disable tqdm since with packing values are in correct
    )

    max_seq_length = 1500  # max sequence length for model and packing of the dataset

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        args=training_args,
        dataset_text_field="instructions",
    )

    trainer_stats = trainer.train()
    train_loss = trainer_stats.training_loss

    logger.info("training loss = %f", train_loss)

    peft_model_id = f"{results_dir}/assets"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--data_dir", default="datasets")
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)

    args = parser.parse_args()
    main(args)
