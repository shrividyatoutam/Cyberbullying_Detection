import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from config import (
    CLEAN_DATA_PATH,
    MAX_LEN,
    MODEL_NAME,
    RANDOM_STATE,
    TEST_SIZE,
    TRANSFORMER_CHECKPOINT_DIR,
    TRANSFORMER_FINAL_DIR,
)


def load_data():
    df = pd.read_csv(CLEAN_DATA_PATH)
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label_id"]
    )
    return train_df, test_df, le


def tokenize_datasets(train_df, test_df, tokenizer):
    train_ds = Dataset.from_pandas(train_df[["clean_text", "label_id"]])
    test_ds = Dataset.from_pandas(test_df[["clean_text", "label_id"]])

    def tok(batch):
        return tokenizer(batch["clean_text"], truncation=True, padding="max_length", max_length=MAX_LEN)

    train_ds = train_ds.map(tok, batched=True)
    test_ds = test_ds.map(tok, batched=True)
    train_ds = train_ds.rename_column("label_id", "labels")
    test_ds = test_ds.rename_column("label_id", "labels")
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return train_ds, test_ds


class WeightedTrainer(Trainer):
    """Trainer subclass that applies class weights to the loss (handles imbalance)."""

    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    macro_f1 = f1_score(labels, preds, average="macro")
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    return {"macro_f1": macro_f1, "precision": precision, "recall": recall}


def main():
    train_df, test_df, le = load_data()
    print("Classes:", list(le.classes_))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds, test_ds = tokenize_datasets(train_df, test_df, tokenizer)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["label_id"]),
        y=train_df["label_id"],
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(le.classes_)
    )

    args = TrainingArguments(
        output_dir=TRANSFORMER_CHECKPOINT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Final eval metrics:", metrics)

    trainer.save_model(TRANSFORMER_FINAL_DIR)
    tokenizer.save_pretrained(TRANSFORMER_FINAL_DIR)
    print(f"Model saved to {TRANSFORMER_FINAL_DIR}")
    print("\nCompare this macro_f1 against 02_baseline_model.py's score for your writeup.")


if __name__ == "__main__":
    main()
