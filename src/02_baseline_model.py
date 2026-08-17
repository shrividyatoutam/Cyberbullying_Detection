import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from config import (
    BASELINE_MODEL_PATH,
    BASELINE_VECTORIZER_PATH,
    CLEAN_DATA_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)


def main():
    df = pd.read_csv(CLEAN_DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"],
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_vec, y_train)

    preds = clf.predict(X_test_vec)

    print(classification_report(y_test, preds))
    macro_f1 = f1_score(y_test, preds, average="macro")
    print(f"Macro F1 (baseline): {macro_f1:.4f}")

    joblib.dump(clf, BASELINE_MODEL_PATH)
    joblib.dump(vectorizer, BASELINE_VECTORIZER_PATH)
    print(f"Saved baseline model + vectorizer to {BASELINE_MODEL_PATH}")
    print("Use this macro-F1 as the number the transformer needs to beat.")


if __name__ == "__main__":
    main()
