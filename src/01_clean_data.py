import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import CLASS_DIST_IMG, CLEAN_DATA_PATH, LENGTH_DIST_IMG, RAW_DATA_PATH


def clean_tweet(text: str) -> str:
    """Basic tweet cleaning: strip URLs, mentions, extra whitespace, lowercase."""
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)              # URLs
    text = re.sub(r"@\w+", " ", text)                           # mentions
    text = re.sub(r"#(\w+)", r"\1", text)                       # hashtags -> keep the word
    text = re.sub(r"[^A-Za-z0-9\s'.,!?]", " ", text)            # strip weird symbols/emojis
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def load_and_clean(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"tweet_text": "text", "cyberbullying_type": "label"})
    df = df.dropna(subset=["text", "label"]).drop_duplicates(subset=["text"])
    df["clean_text"] = df["text"].apply(clean_tweet)
    df = df[df["clean_text"].str.len() > 3]  # drop empty/near-empty tweets after cleaning
    return df.reset_index(drop=True)


def explore(df: pd.DataFrame):
    print("Class distribution:")
    print(df["label"].value_counts())
    print("\nTotal rows after cleaning/dedup:", len(df))

    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, y="label", order=df["label"].value_counts().index)
    plt.title("Class distribution")
    plt.tight_layout()
    plt.savefig(CLASS_DIST_IMG)
    print(f"Saved {CLASS_DIST_IMG}")

    df["length"] = df["clean_text"].str.split().apply(len)
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x="length", hue="label", element="step", bins=30)
    plt.title("Tweet length (words) by class")
    plt.tight_layout()
    plt.savefig(LENGTH_DIST_IMG)
    print(f"Saved {LENGTH_DIST_IMG}")


if __name__ == "__main__":
    cleaned = load_and_clean()
    explore(cleaned)
    cleaned.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"\nSaved cleaned data to {CLEAN_DATA_PATH}")
