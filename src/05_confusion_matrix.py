"""
Step 5: Confusion matrix + per-class error breakdown.

Run this after 04_error_analysis_bias_check.py. It quantifies patterns spotted
by eye (e.g. one class over-triggering) across the full test set rather than
just the top confident errors -- turning "I noticed this" into "I measured
this," which is what you actually want to say in interviews.

Usage:
    python src/05_confusion_matrix.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    CLEAN_DATA_PATH,
    CONFUSION_MATRIX_IMG,
    FULL_TEST_PREDICTIONS_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    TRANSFORMER_FINAL_DIR,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def predict(texts, tokenizer, model, batch_size=32):
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
    return np.array(all_preds)


def main():
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_FINAL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_FINAL_DIR).to(device)
    model.eval()

    df = pd.read_csv(CLEAN_DATA_PATH)
    le = LabelEncoder()
    le.fit(df["label"])
    df["label_id"] = le.transform(df["label"])
    _, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label_id"])

    preds = predict(test_df["clean_text"].tolist(), tokenizer, model)
    test_df = test_df.copy()
    test_df["pred_label"] = le.inverse_transform(preds)

    # --- Confusion matrix ---
    cm = confusion_matrix(test_df["label"], test_df["pred_label"], labels=le.classes_)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print("Confusion matrix (rows = true label, cols = predicted):\n")
    print(cm_df)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_IMG)
    print(f"\nSaved {CONFUSION_MATRIX_IMG}")

    # --- Quantify a specific over-triggering pattern for one class, e.g. "religion" ---
    target_class = "religion"
    print(f"\n--- Over-triggering check: what gets misclassified as '{target_class}'? ---")
    false_positive = test_df[
        (test_df["label"] != target_class) & (test_df["pred_label"] == target_class)
    ]
    print(f"{len(false_positive)} examples from OTHER true classes were predicted as '{target_class}'")
    print(false_positive["label"].value_counts())

    class_recall = cm_df.loc[target_class, target_class] / cm_df.loc[target_class].sum()
    class_precision = cm_df.loc[target_class, target_class] / cm_df[target_class].sum()
    print(f"\n'{target_class}' class -- precision: {class_precision:.3f}, recall: {class_recall:.3f}")
    print("Low precision here indicates over-triggering: the model predicts this class")
    print("more often than it should, possibly keying on surface-level keywords rather")
    print("than actual hostile intent -- worth digging into further.")

    test_df.to_csv(FULL_TEST_PREDICTIONS_PATH, index=False)
    print(f"\nSaved {FULL_TEST_PREDICTIONS_PATH} ({len(test_df)} predictions, for further digging)")


if __name__ == "__main__":
    main()
