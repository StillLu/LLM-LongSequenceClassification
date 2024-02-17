import argparse
import torch
import os
import pandas as pd
import pickle
from tqdm import tqdm
import datasets

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)



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


    peft_model_id = f"{args.data_dir}/assets"


    # load base LLM model and tokenizer
    model = AutoPeftModelForCausalLM.from_pretrained(
        peft_model_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

    results = []
    oom_examples = []
    instructions, labels = test_dataset["instructions"], test_dataset["labels"]

    for instruct, label in tqdm(zip(instructions, labels)):

        input_ids = tokenizer(instruct, return_tensors="pt", truncation=True)

        input_ids.to('cuda')

        with torch.inference_mode():
            try:
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=20,
                    do_sample=True,
                    top_p=0.95,
                    temperature=1e-3,
                )
                result = tokenizer.batch_decode(
                   outputs.detach().cpu().numpy(), skip_special_tokens=True
                )[0]
                result = result[len(instruct) :]
                print(result)
            except:
                result = ""
                oom_examples.append(input_ids.shape[-1])

            results.append(result)

    metrics = {
        "micro_f1": f1_score(labels, results, average="micro"),
        "macro_f1": f1_score(labels, results, average="macro"),
        "precision": precision_score(labels, results, average="micro"),
        "recall": recall_score(labels, results, average="micro"),
        "accuracy": accuracy_score(labels, results),
        "oom_examples": oom_examples,
    }
    print(metrics)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="datasets",
    )

    args = parser.parse_args()
    main(args)