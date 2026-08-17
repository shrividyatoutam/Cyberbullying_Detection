# Cyberbullying Detection using DistilBERT

##  Project Overview

This project uses Natural Language Processing (NLP) to classify tweets into different cyberbullying-related categories.

The goal is to use a transformer-based model to analyze tweet text and predict its category. A Streamlit application provides a simple interface for testing the trained model.

##  Classification Classes

The model classifies tweets into six categories:

* **age** — cyberbullying related to age
* **ethnicity** — cyberbullying related to ethnicity
* **gender** — cyberbullying related to gender
* **religion** — cyberbullying related to religion
* **not_cyberbullying** — tweets that are not classified as cyberbullying
* **other_cyberbullying** — other forms of cyberbullying

##  Dataset

The project uses a labeled Twitter dataset containing approximately 47,000 tweets across the six categories.

The dataset is used to train and evaluate the text classification model.

##  Project Structure

```text
cyberbullying-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
└── src/
    ├── config.py                        # shared paths/constants
    ├── 01_clean_data.py                 # load, clean, explore
    ├── 02_baseline_model.py             # TF-IDF + Logistic Regression
    ├── 03_train_transformer.py          # fine-tune DistilBERT
    ├── 04_error_analysis_bias_check.py  # misclassifications + bias probes
    ├── 05_confusion_matrix.py           # confusion matrix + over-triggering check
    └── 06_zero_shot_llm_baseline.py     # zero-shot Gemini comparison
```

##  Installation and Running

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```
