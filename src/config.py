"""
Central configuration for file paths and constants shared across the pipeline.

All data/model/output artifacts live under one directory so the scripts work
the same way locally, in Colab, or in CI. Override the location with the
PROJECT_DIR environment variable, e.g.:

    export PROJECT_DIR=/content/drive/MyDrive/SentimentAnalysisProject   # Colab
    export PROJECT_DIR=./artifacts                                       # local (default)
"""
import os

PROJECT_DIR = os.environ.get(
    "PROJECT_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts"),
)
os.makedirs(PROJECT_DIR, exist_ok=True)

# --- Data ---
RAW_DATA_PATH = os.path.join(PROJECT_DIR, "cyberbullying_tweets.csv")
CLEAN_DATA_PATH = os.path.join(PROJECT_DIR, "cyberbullying_clean.csv")

# --- Baseline model (TF-IDF + Logistic Regression) ---
BASELINE_VECTORIZER_PATH = os.path.join(PROJECT_DIR, "baseline_vectorizer.joblib")
BASELINE_MODEL_PATH = os.path.join(PROJECT_DIR, "baseline_logreg.joblib")

# --- Transformer model (DistilBERT) ---
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
TRANSFORMER_CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "distilbert_cyberbullying")
TRANSFORMER_FINAL_DIR = os.path.join(PROJECT_DIR, "distilbert_cyberbullying_final")

# --- Analysis outputs ---
CLASS_DIST_IMG = os.path.join(PROJECT_DIR, "class_distribution.png")
LENGTH_DIST_IMG = os.path.join(PROJECT_DIR, "length_distribution.png")
MISCLASSIFIED_PATH = os.path.join(PROJECT_DIR, "misclassified_examples.csv")
BIAS_CHECK_PATH = os.path.join(PROJECT_DIR, "bias_check_results.csv")
CONFUSION_MATRIX_IMG = os.path.join(PROJECT_DIR, "confusion_matrix.png")
FULL_TEST_PREDICTIONS_PATH = os.path.join(PROJECT_DIR, "full_test_predictions.csv")

# --- Zero-shot LLM baseline ---
ZERO_SHOT_MODEL = "gemini-3.5-flash-lite"
ZERO_SHOT_SAMPLE_SIZE = 30
ZERO_SHOT_REQUEST_DELAY_SECONDS = 4
ZERO_SHOT_RESULTS_PATH = os.path.join(PROJECT_DIR, "zero_shot_llm_results.csv")

# --- Shared ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
LABELS = ["age", "ethnicity", "gender", "not_cyberbullying", "other_cyberbullying", "religion"]
