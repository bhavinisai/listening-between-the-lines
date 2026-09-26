#!/usr/bin/env python3
"""
Fine-tune xlm-roberta-base into the 6-class question-type sequence classifier
that question_type_analysis.py loads via --model_path.

Training data:
    An LLM-labeled seed set built from a stratified sample of the questions
    already extracted by `question_type_analysis.py --extract_only`
    (results/question_classification.csv -> results/question_type_llm_seed_sample.csv,
    labeled and merged into results/question_type_seed_labeled.csv with columns
    question_id, text, label). See documentation/question_type_labeling.md for
    how the seed labels were produced (LLM-labeled, not hand-annotated) --
    this is a methodological choice that should be disclosed in the writeup.

Labels (must match CLASS_LABELS in question_type_analysis.py):
    closed, open, leading, personal, professional, challenge

Usage:
    python src/finetune_question_type_xlmr.py \
        --labeled_csv results/question_type_seed_labeled.csv \
        --out_dir models/xlmr_question_type \
        --epochs 6
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

CLASS_LABELS = ["closed", "open", "leading", "personal", "professional", "challenge"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled_csv", default="results/question_type_seed_labeled.csv")
    ap.add_argument("--base_model", default="xlm-roberta-base")
    ap.add_argument("--out_dir", default="models/xlmr_question_type")
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--class_weighted", action="store_true",
        help="Weight the training loss by inverse class frequency. Use this when the "
             "seed labels are imbalanced (e.g. few 'challenge' examples) so the model "
             "isn't just optimized for the majority classes.",
    )
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from datasets import Dataset
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, DataCollatorWithPadding,
    )

    df = pd.read_csv(args.labeled_csv)
    df["label"] = df["label"].str.strip().str.lower()
    bad = ~df["label"].isin(CLASS_LABELS)
    if bad.any():
        raise SystemExit(f"{bad.sum()} rows have a label outside {CLASS_LABELS}: "
                          f"{df.loc[bad, 'label'].unique().tolist()}")

    label2id = {l: i for i, l in enumerate(CLASS_LABELS)}
    id2label = {i: l for l, i in label2id.items()}
    df["label_id"] = df["label"].map(label2id)

    train_df, val_df = train_test_split(
        df, test_size=args.val_size, random_state=args.seed, stratify=df["label_id"],
    )
    print(f"[OK] train={len(train_df)} val={len(val_df)}")
    print(train_df["label"].value_counts())

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = Dataset.from_pandas(train_df[["text", "label_id"]].rename(columns={"label_id": "label"}))
    val_ds = Dataset.from_pandas(val_df[["text", "label_id"]].rename(columns={"label_id": "label"}))
    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=len(CLASS_LABELS), id2label=id2label, label2id=label2id,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
        }

    training_args = TrainingArguments(
        output_dir=os.path.join(args.out_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=20,
        seed=args.seed,
        report_to=[],
    )

    trainer_cls = Trainer
    trainer_kwargs = {}
    if args.class_weighted:
        counts = train_df["label_id"].value_counts().reindex(range(len(CLASS_LABELS)), fill_value=0)
        weights = (counts.sum() / (len(CLASS_LABELS) * counts.clip(lower=1))).values
        class_weights = torch.tensor(weights, dtype=torch.float)
        print(f"[OK] class weights (inverse frequency): "
              f"{dict(zip(CLASS_LABELS, weights.round(3)))}")

        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
                loss = loss_fct(logits, labels)
                return (loss, outputs) if return_outputs else loss

        trainer_cls = WeightedTrainer

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        **trainer_kwargs,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("[OK] Final validation metrics:", metrics)

    val_preds = np.argmax(trainer.predict(val_ds).predictions, axis=-1)
    val_labels = val_ds["label"]
    report = classification_report(
        val_labels, val_preds, target_names=CLASS_LABELS, zero_division=0,
    )
    print(report)

    final_dir = args.out_dir
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    with open(os.path.join(final_dir, "val_classification_report.txt"), "w") as f:
        f.write(report)
    print(f"[OK] Saved fine-tuned model to {final_dir}")


if __name__ == "__main__":
    main()
