import os
import time

import pandas as pd
from google import genai
from google.genai import types
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from config import (
    CLEAN_DATA_PATH,
    LABELS,
    RANDOM_STATE,
    TEST_SIZE,
    ZERO_SHOT_MODEL,
    ZERO_SHOT_REQUEST_DELAY_SECONDS,
    ZERO_SHOT_RESULTS_PATH,
    ZERO_SHOT_SAMPLE_SIZE,
)

SYSTEM_PROMPT = f"""You are a content moderation classifier. Classify the given tweet
into exactly one of these categories: {', '.join(LABELS)}.

- age, ethnicity, gender, religion: use these if the tweet contains cyberbullying
  targeting someone based on that specific attribute.
- other_cyberbullying: use this if it's bullying/harassment but not clearly tied to
  age, ethnicity, gender, or religion.
- not_cyberbullying: use this if the tweet is not bullying/harassment at all.

Respond with ONLY the category label, nothing else. No explanation, no punctuation."""


def classify_tweet(client, text: str, max_retries: int = 3) -> tuple[str, bool]:
    safety_settings = [
        types.SafetySetting(category=cat, threshold="BLOCK_NONE")
        for cat in [
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        ]
    ]

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=ZERO_SHOT_MODEL,
                contents=text,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=20,  # headroom in case thinking can't be fully disabled
                    safety_settings=safety_settings,
                ),
            )

            if response.text is None:
                finish_reason = response.candidates[0].finish_reason if response.candidates else None
                print(f"  Blocked/empty response (finish_reason={finish_reason}) -- "
                      f"defaulting to 'other_cyberbullying'")
                return "other_cyberbullying", True

            raw = response.text.strip().lower()
          
            matched = next((lbl for lbl in LABELS if lbl in raw), None)
            label = matched if matched else "not_cyberbullying"
            return label, False
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower() or "RESOURCE_EXHAUSTED" in str(e):
                wait = 60 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            elif "503" in str(e) or "UNAVAILABLE" in str(e) or "overloaded" in str(e).lower():
                wait = 30 * (attempt + 1)
                print(f"  Model overloaded (503), waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    return "not_cyberbullying", True  # fallback if retries exhausted


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set the GEMINI_API_KEY environment variable before running this script "
            "(see the setup instructions in this file's docstring)."
        )

    client = genai.Client(api_key=api_key)

    df = pd.read_csv(CLEAN_DATA_PATH)
    # Use the SAME split logic as the other scripts so this is a fair comparison
    _, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label"])
    sample = test_df.sample(n=min(ZERO_SHOT_SAMPLE_SIZE, len(test_df)), random_state=RANDOM_STATE).reset_index(drop=True)

    preds = []
    blocked_count = 0
    start = time.time()
    for i, row in sample.iterrows():
        pred, was_blocked = classify_tweet(client, row["text"])
        preds.append(pred)
        if was_blocked:
            blocked_count += 1
        time.sleep(ZERO_SHOT_REQUEST_DELAY_SECONDS)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(sample)} done...")
    elapsed = time.time() - start

    sample["pred_label"] = preds

    print("\n" + classification_report(sample["label"], sample["pred_label"]))
    macro_f1 = f1_score(sample["label"], sample["pred_label"], average="macro")
    avg_latency = elapsed / len(sample)

    print(f"Zero-shot Gemini macro F1: {macro_f1:.4f}")
    print(f"Total time: {elapsed:.1f}s for {len(sample)} tweets ({avg_latency:.3f}s/tweet avg)")
    print(f"Blocked by safety filters (defaulted to 'other_cyberbullying'): {blocked_count}/{len(sample)}")
    if blocked_count > 0:
        print("  ^ Worth noting in your writeup: some tweets triggered Gemini's safety")
        print("    filters even with relaxed thresholds, since they contain genuine hate")
        print("    speech/slurs by dataset design. This is itself a real deployment")
        print("    consideration for using general-purpose LLMs as content moderators.")
    print("Compare this against the DistilBERT macro F1 and note:")
    print("  - DistilBERT: near-instant inference once loaded, no per-call cost, but")
    print("    needed a labeled dataset + training time to get there.")
    print("  - Zero-shot LLM: no training needed and free on this tier, but slower")
    print("    per-prediction and rate-limited, and accuracy depends entirely on")
    print("    prompt quality rather than task-specific training.")

    sample.to_csv(ZERO_SHOT_RESULTS_PATH, index=False)
    print(f"\nSaved {ZERO_SHOT_RESULTS_PATH}")


if __name__ == "__main__":
    main()
