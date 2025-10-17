from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    XLMRobertaForSequenceClassification,
    ModernBertForSequenceClassification,
)
from datasets import load_dataset, ClassLabel
import numpy as np
import evaluate
import argparse
import os
import wandb
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import random
import json


def compute_metrics(eval_pred):
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred
    preds = np.round(logits.squeeze()).clip(0, 5).astype(int)
    labels = np.round(labels.squeeze()).astype(int)
    precision = precision_metric.compute(
        predictions=preds, references=labels, average="macro"
    )["precision"]
    recall = recall_metric.compute(
        predictions=preds, references=labels, average="macro"
    )["recall"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]

    report = classification_report(labels, preds)
    cm = confusion_matrix(labels, preds)
    print("Validation Report:\n" + report)
    print("Confusion Matrix:\n" + str(cm))

    return {
        "precision": precision,
        "recall": recall,
        "f1_macro": f1,
        "accuracy": accuracy,
    }


label_min_max_map = {
    "fw_edu_score": (0, 5),
    "dclm_score": (0, 5),
    "ocr_quality_score": (0, 3),
    "fw_edu_score_2": (0, 5),
}


def parse_sample_per_class(arg_value, min_label, max_label):
    """
    Accepts an int (uniform samples per class) or a comma-separated string like "10,20,30".
    Returns a dict {class_label: k_samples}.
    """
    n_classes = max_label - min_label + 1
    if isinstance(arg_value, int):
        return {c: int(arg_value) for c in range(min_label, max_label + 1)}
    # It's a string — try to parse "a,b,c"
    if isinstance(arg_value, str):
        arg_value = arg_value.strip()
        # allow numeric strings for uniform sampling too
        if arg_value.isdigit():
            k = int(arg_value)
            return {c: k for c in range(min_label, max_label + 1)}
        parts = [p.strip() for p in arg_value.split(",") if p.strip() != ""]
        counts = list(map(int, parts))
        if len(counts) != n_classes:
            raise ValueError(
                f"--sample_per_class provided {len(counts)} counts, "
                f"but there are {n_classes} classes (from {min_label} to {max_label})."
            )
        return {min_label + i: counts[i] for i in range(n_classes)}
    raise ValueError("--sample_per_class must be an int or a comma-separated string.")


def main(args):
    # Load dataset with subset if specified
    if args.subset:
        dataset = load_dataset(args.dataset_name, name=args.subset, split="train", num_proc=8)
    else:
        dataset = load_dataset(args.dataset_name, split="train", num_proc=8)

    min_label, max_label = label_min_max_map[args.target_column]

    # Filter unusable rows
    dataset = dataset.filter(
        lambda x: x[args.target_column] is not None
        and x[args.target_column] != -1
        and x["text"] is not None
        and x["text"].strip() != ""
    )

    # Clip labels to range and cast to ClassLabel
    dataset = dataset.map(
        lambda x: {
            args.target_column: np.clip(
                int(x[args.target_column]), min_label, max_label
            )
        },
        num_proc=10,
    )
    dataset = dataset.cast_column(
        args.target_column,
        ClassLabel(names=[str(i) for i in range(max_label - min_label + 1)]),
    )

    # Split BEFORE any sampling/balancing
    dataset = dataset.train_test_split(
        train_size=0.96, seed=42, stratify_by_column=args.target_column
    )

    # -------- Model init --------
    kwargs = {}
    if "snowflake" in args.base_model_name:
        kwargs["hidden_dropout_prob"] = 0.0
        kwargs["classifier_dropout"] = None

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_name,
        num_labels=1,  # regression-style head; metrics round to class indices
        **kwargs,
        output_hidden_states=False,
    )

    if isinstance(model, ModernBertForSequenceClassification):
        for param in model.model.parameters():
            param.requires_grad = False
        if args.transformer_layers_unfrozen > 0:
            for layer in model.model.layers[-args.transformer_layers_unfrozen:]:
                for param in layer.parameters():
                    param.requires_grad = True

    if isinstance(model, XLMRobertaForSequenceClassification):
        for param in model.roberta.parameters():
            param.requires_grad = False
        if args.transformer_layers_unfrozen > 0:
            for layer in model.roberta.encoder.layer[-args.transformer_layers_unfrozen:]:
                for param in layer.parameters():
                    param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    # -------- Tokenizer & preprocessing --------
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name,
        model_max_length=min(
            getattr(model.config, "max_position_embeddings", 2048), 2048
        ),
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(examples):
        batch = tokenizer(examples["text"], truncation=True)
        batch["labels"] = np.float32(examples[args.target_column])
        return batch

    dataset = dataset.map(preprocess, batched=True, num_proc=10)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # -------- Prepare one-epoch expanded training set via per-class sampling-with-replacement --------
    if args.sample_per_class is None:
        raise ValueError(
            "Please provide --sample_per_class (int or comma-separated string)."
        )

    # Build class -> indices map on tokenized train split
    train_full = dataset["train"]
    labels_full = np.array(train_full["labels"]).astype(int)
    class_to_indices = {
        c: [i for i, y in enumerate(labels_full) if y == c]
        for c in range(min_label, max_label + 1)
    }

    # Parse desired sample counts
    target_counts = parse_sample_per_class(args.sample_per_class, min_label, max_label)
    print(f"Target samples per class: {target_counts}")

    rng = random.Random(args.sample_seed)
    selected = []
    for c in range(min_label, max_label + 1):
        k = int(target_counts.get(c, 0))
        pool = class_to_indices.get(c, [])
        if k <= 0:
            continue
        if len(pool) == 0:
            print(f"Warning: no samples available for class {c}; requesting {k} -> 0")
            continue
        # sample with replacement
        chosen = [rng.choice(pool) for _ in range(k)]
        selected.extend(chosen)

    rng.shuffle(selected)

    expanded_train = train_full.select(selected)
    print(
        f"Expanded training set size: {len(expanded_train)} "
        f"(sum of requested per-class samples)"
    )

    # Replace training set with expanded one; train for ONE epoch
    dataset["train"] = expanded_train

    # Now randomly sample test set so that we have at most 20k samples
    test_dataset = dataset["test"]
    if len(test_dataset) > 20000:
        # Randomly sample 20k samples from test set
        rng_test = random.Random(args.sample_seed)
        test_indices = list(range(len(test_dataset)))
        rng_test.shuffle(test_indices)
        sampled_test_indices = test_indices[:20000]
        dataset["test"] = test_dataset.select(sampled_test_indices)
        print(f"Sampled test set size: {len(dataset['test'])} (from original {len(test_dataset)})")
    else:
        print(f"Test set size: {len(test_dataset)} (no sampling needed)")

    # -------- Training setup --------
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")

    train_batch_sizes = {
        "Snowflake/snowflake-arctic-embed-m": 128,
        "Snowflake/snowflake-arctic-embed-m-v2.0": 128,
        "Snowflake/snowflake-arctic-embed-l-v2.0": 128,
        "answerdotai/ModernBERT-base": 128,
        "answerdotai/ModernBERT-large": 128,
        "mmbert-colab/mmBERT-base": 128,
    }
    eval_batch_sizes = {
        "Snowflake/snowflake-arctic-embed-m": 256,
        "Snowflake/snowflake-arctic-embed-m-v2.0": 256,
        "Snowflake/snowflake-arctic-embed-l-v2.0": 256,
        "answerdotai/ModernBERT-base": 256,
        "answerdotai/ModernBERT-large": 256,
        "mmbert-colab/mmBERT-base": 256,
    }
    
    # Build run name with optional prefix
    base_run_name = f"{args.base_model_name}-{args.target_column}-spc-{args.sample_per_class}-{args.transformer_layers_unfrozen}-{args.lr}-{args.max_steps}"
    if args.prefix_run_name:
        run_name = f"{args.prefix_run_name}-{base_run_name}"
    else:
        run_name = base_run_name

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.max_steps//10,
        save_steps=args.max_steps//10,
        max_steps=args.max_steps,
        logging_steps=100,
        learning_rate=args.lr,
        num_train_epochs=1,  # <-- exactly one epoch
        seed=args.sample_seed,
        report_to="wandb",
        per_device_train_batch_size=train_batch_sizes[args.base_model_name],
        per_device_eval_batch_size=eval_batch_sizes[args.base_model_name],
        eval_on_start=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        bf16=True,
        run_name=run_name,
        push_to_hub=False,
    )

    wandb.init(
        project="fine-pdfs-classification",
        config={
            "trainer_epochs": training_args.num_train_epochs,  # will be 1
            "sample_per_class": str(args.sample_per_class),
            "expanded_train_size": len(expanded_train),
            "batch_size": train_batch_sizes[args.base_model_name],
            "learning_rate": training_args.learning_rate,
            "trainable_params": trainable_params,
            "transformer_layers_unfrozen": args.transformer_layers_unfrozen,
            "total_params": total_params,
            "seed": args.sample_seed,
        },
        name=run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "final"))

    # Final evaluation
    result = trainer.evaluate(dataset["test"])

    # Save the final evaluation results
    eval_results_dir = os.path.join(training_args.output_dir, "evaluation_results")
    os.makedirs(eval_results_dir, exist_ok=True)
    final_results_path = os.path.join(eval_results_dir, "final.json")
    with open(final_results_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Final evaluation results saved to: {final_results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_name", type=str, default="Snowflake/snowflake-arctic-embed-m"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="HuggingFaceFW-Dev/fine-pdfs-classification-1-chunk-tb-teacher-1M-eng_Latn-Qwen_Qwen3-235B-A22B-Instruct-2507",
    )
    parser.add_argument(
        "--subset",
        type=str,
        help="Subset of the dataset",
    )
    parser.add_argument("--target_column", type=str, default="fw_edu_score")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./trained_models",
    )
    parser.add_argument(
        "--sample_per_class",
        type=str,
        required=True,
        help='Either an integer (uniform samples per class), or a comma-separated list like "10,20,30" '
             "giving counts for each class in order from min_label to max_label.",
    )
    parser.add_argument(
        "--transformer_layers_unfrozen",
        type=int,
        default=0,
        help="Number of transformer layers to unfreeze",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--max_steps", type=int, help="Maximum number of steps"
    )
    parser.add_argument(
        "--sample_seed", type=int, default=42, help="RNG seed for sampling-with-replacement"
    )
    parser.add_argument(
        "--prefix_run_name", type=str, help="Prefix for run name"
    )
    args = parser.parse_args()

    # Cast numeric strings to int for convenience
    try:
        if args.sample_per_class.isdigit():
            args.sample_per_class = int(args.sample_per_class)
    except AttributeError:
        pass  # already an int

    main(args)
