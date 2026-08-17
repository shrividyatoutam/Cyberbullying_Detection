import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "distilbert_cyberbullying_final"  # local path after downloading from Drive
LABELS = ["age", "ethnicity", "gender", "not_cyberbullying", "other_cyberbullying", "religion"]

st.set_page_config(page_title="Cyberbullying Detector", page_icon="🛡️")


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


def classify(text, tokenizer, model):
    inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]
    return probs


st.title("🛡️ Cyberbullying Detector")
st.caption("Fine-tuned DistilBERT classifying tweets into 6 categories")

tokenizer, model = load_model()

text = st.text_area("Enter a tweet to classify:", height=100, placeholder="Type or paste text here...")

if st.button("Classify", type="primary") and text.strip():
    probs = classify(text, tokenizer, model)
    top_idx = torch.argmax(probs).item()
    top_label = LABELS[top_idx]
    top_conf = probs[top_idx].item()

    if top_label == "not_cyberbullying":
        st.success(f"**{top_label}** ({top_conf:.1%} confidence)")
    else:
        st.error(f"**{top_label}** ({top_conf:.1%} confidence)")

    st.subheader("All class probabilities")
    for label, prob in sorted(zip(LABELS, probs.tolist()), key=lambda x: -x[1]):
        st.progress(prob, text=f"{label}: {prob:.1%}")

st.divider()
