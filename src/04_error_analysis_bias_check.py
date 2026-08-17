import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    BIAS_CHECK_PATH,
    CLEAN_DATA_PATH,
    MISCLASSIFIED_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    TRANSFORMER_FINAL_DIR,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_FINAL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_FINAL_DIR).to(device)
    model.eval()
    return tokenizer, model


def predict(texts, tokenizer, model, batch_size=32):
    """Run inference in batches, return predicted class indices and confidence scores."""
    all_preds, all_confs = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_confs.extend(confs.cpu().numpy())
    return np.array(all_preds), np.array(all_confs)

def run_error_analysis(tokenizer, model, le):
    df = pd.read_csv(CLEAN_DATA_PATH)
    df["label_id"] = le.transform(df["label"])
    _, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label_id"])

    preds, confs = predict(test_df["clean_text"].tolist(), tokenizer, model)
    test_df = test_df.copy()
    test_df["pred_label"] = le.inverse_transform(preds)
    test_df["confidence"] = confs

    wrong = test_df[test_df["label"] != test_df["pred_label"]].sort_values(
        "confidence", ascending=False
    )

    print(f"Total test examples: {len(test_df)}")
    print(f"Misclassified: {len(wrong)} ({len(wrong) / len(test_df):.1%})")
    print("\nMost confident WRONG predictions (worth reading closely -- these are the")
    print("interesting failure cases, not just borderline/ambiguous ones):\n")

    cols = ["text", "label", "pred_label", "confidence"]
    print(wrong[cols].head(20).to_string(index=False))

    wrong[cols].to_csv(MISCLASSIFIED_PATH, index=False)
    print(f"\nSaved all {len(wrong)} misclassified examples to {MISCLASSIFIED_PATH}")
    print("\n--> Manually read through 15-20 of these and tag each with a failure")
    print("    category: sarcasm, reclaimed language, coded/slang, too short, mislabeled")
    print("    in the original dataset, etc. This categorization is what you'll")
    print("    actually talk about in interviews.")


BIAS_PROBE_SENTENCES = [
    "I'm so proud to be part of the Black community today.",
    "Just got back from a great Pride parade with my LGBTQ friends.",
    "My grandma is Jewish and she makes the best food.",
    "Excited to celebrate Eid with my Muslim family this weekend.",
    "I'm a woman in tech and I love my job.",
    "My friend is trans and she's one of the kindest people I know.",
    "Growing up Latino, food was always at the center of family gatherings.",
    "As a gay man, I'm grateful for how far things have come.",
    "My dad is elderly but he still beats me at chess every time.",
    "I'm autistic and I think differently, and that's okay.",
]


def run_bias_check(tokenizer, model, le):
    preds, confs = predict(BIAS_PROBE_SENTENCES, tokenizer, model)
    pred_labels = le.inverse_transform(preds)

    results = pd.DataFrame({
        "text": BIAS_PROBE_SENTENCES,
        "predicted_label": pred_labels,
        "confidence": confs,
    })

    flagged = results[results["predicted_label"] != "not_cyberbullying"]

    print("\nBias check results:")
    print(results.to_string(index=False))

    if len(flagged) > 0:
        print(f"\n[!] {len(flagged)}/{len(results)} neutral, non-abusive sentences mentioning")
        print("    identity/group terms were flagged as cyberbullying. This is a real")
        print("    finding worth discussing: the model may be picking up on identity")
        print("    terms as a signal rather than actual abusive intent.")
    else:
        print(f"\nNone of the {len(results)} neutral probe sentences were flagged. Good sign,")
        print("but this is a small, hand-written probe set -- not a rigorous fairness audit.")
        print("Say that explicitly in your writeup rather than overclaiming 'no bias found.'")

    results.to_csv(BIAS_CHECK_PATH, index=False)
    print(f"\nSaved {BIAS_CHECK_PATH}")


def main():
    tokenizer, model = load_model()
    df = pd.read_csv(CLEAN_DATA_PATH)
    le = LabelEncoder()
    le.fit(df["label"])
    print("Classes:", list(le.classes_))

    run_error_analysis(tokenizer, model, le)
    run_bias_check(tokenizer, model, le)


if __name__ == "__main__":
    main()
