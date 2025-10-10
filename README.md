# Indonesian Tweet Sentiment — Final Enhanced Pipeline

End-to-end sentiment analysis for Indonesian tweets with a **clean preprocessing pipeline**, **model comparison**, and **auto-selection by Macro F1**. Outputs are saved to `./SentimentFinalEnhanced` (or to your Google Drive folder if you set it).

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Features](#features)
- [Models](#models)
- [Training & Evaluation Flow](#training--evaluation-flow)
- [Outputs](#outputs)
- [How to Run (Colab + Google Drive)](#how-to-run-colab--google-drive)
- [How to Run (Local)](#how-to-run-local)
- [Repository Structure](#repository-structure)
- [Reproducibility](#reproducibility)
- [Extending / Next Steps](#extending--next-steps)
- [License](#license)

---

## Overview
This project is part of my AI Bootcamp assignment, especially in NLP area. It builds a sentiment classifier for Indonesian tweets and compares **Logistic Regression** vs **Multinomial Naive Bayes**, selecting the **best model by Macro F1**. The notebook also includes:
- A **cleaning pipeline** tuned for social text (preserving `@mentions` and `#hashtags`)
- **Indonesian stopword removal** and **Sastrawi stemming**
- **TF-IDF word n-grams (1,2)**
- **Macro F1 bar chart** and **confusion matrix for the best model**
- A **sample inference cell** to predict on new tweets

---

## Dataset
Provide a `tweet.csv` with:
- `sentimen` (labels: e.g., `positif`, `negatif`, `netral`)
- `tweet` (raw tweet text)
- Optional index columns (e.g., `Unnamed: 0`) are dropped automatically

---

## Preprocessing
Steps applied in order:

1. Lowercasing  
2. Remove URLs (`http(s)://…`, `www…`)  
3. Keep mentions & hashtags (`@user`, `#topic`)  
4. Normalize Indonesian slang/shortcuts (custom dictionary)  
5. Limit repeated characters (e.g., “baguuuuuus” → “baguuus”)  
6. Remove standalone numbers  
7. Remove Indonesian stopwords (non-@/# tokens only)  
8. **Sastrawi stemming** (non-@/# tokens only)  

This ensures tweets are normalized while **preserving entity/topic signals**.

---

## Features
- **TF-IDF word n-grams**: `(1,2)`  
- `min_df=2`, `max_df=0.95`  

---

## Models
Two linear baselines are trained and evaluated:
- **Logistic Regression** (`class_weight="balanced"`, `max_iter=1000`)  
- **Multinomial Naive Bayes** (`alpha=1.0`)  

**Selection criterion**: **Macro F1** (higher is better).  

---

## Training & Evaluation Flow
1. Clean + preprocess text → `tweet_clean.csv`  
2. Train/test split (`test_size=0.2`, `random_state=42`, stratify labels)  
3. Fit TF-IDF (1,2), train LogReg & NB  
4. Evaluate both models → Accuracy, Macro F1, Per-class F1  
5. Pick the **best model by Macro F1**  
6. Save artifacts + plots  
7. Show **Macro F1 bar chart** (also saved as `f1_bar_chart.png`) and **confusion matrix for the best model**  
8. Run **sample inference** on new tweets  

> Note: Metrics and per-class F1 are **displayed inside the notebook**. No `summary_metrics.csv` file is written.

---

## Outputs
All artifacts are written to **`./SentimentFinalEnhanced`** (or to your Drive folder if set):

- `tweet_clean.csv` — cleaned dataset  
- `tfidf.joblib` — TF-IDF vectorizer  
- `logreg.joblib`, `nb.joblib` — trained models  
- `best_model.joblib` — best model (by Macro F1)  
- `label_encoder.joblib` — encoder for labels  
- `report_logreg.txt`, `report_nb.txt` — detailed classification reports  
- `f1_bar_chart.png` — Macro F1 bar chart  
- `confusion_matrix_best.png` — confusion matrix for the best model  

---

## How to Run (Colab + Google Drive)

**1) Mount Google Drive** (so your data and outputs persist):

```python
from google.colab import drive
drive.mount('/content/drive')
```

**2) Set your data and output paths**  
You said your data is in **`MyDrive/Proyek/Data`**. In Colab, that resolves to:

```python
from pathlib import Path

# Point to your dataset in Drive
DATA_PATH = "/content/drive/MyDrive/Proyek/Data/tweet.csv"

# Save outputs back to Drive (recommended)
OUTPUT_DIR = Path("/content/drive/MyDrive/Proyek/SentimentFinalEnhanced")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
```

> Tip: In the notebook (`sentiment_final_enhanced.ipynb`), you can either **edit the Config cell** to use these values or **add a small cell after the Config** to override `DATA_PATH` and `OUTPUT_DIR` with the snippet above.

**3) Run all cells**  
The notebook will install dependencies, preprocess, train/evaluate both models, pick the best by Macro F1, and write artifacts to `OUTPUT_DIR`.

**4) Sample predictions**  
Use the final cell to test the best model on new text. It will also show which model was selected and can re-plot the confusion matrix on demand.

---

## How to Run (Local)
```bash
# setup environment
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# install dependencies
pip install sastrawi scikit-learn joblib matplotlib pandas numpy nltk jupyter

# run jupyter
jupyter notebook
```

Open `sentiment_final_enhanced.ipynb` and run all cells.  
> First run may download NLTK stopwords.

---

## Repository Structure
```
.
├─ notebook
|  ├─ sentiment_final_enhanced.ipynb
├─ data
|  ├─ tweet.csv                  # dataset (not pushed if sensitive)
├─ SentimentFinalEnhanced/    # created by the notebook (or in Drive if you set OUTPUT_DIR there)
│  ├─ tweet_clean.csv
│  ├─ tfidf.joblib
│  ├─ logreg.joblib
│  ├─ nb.joblib
│  ├─ best_model.joblib
│  ├─ label_encoder.joblib
│  ├─ report_logreg.txt
│  ├─ report_nb.txt
│  ├─ f1_bar_chart.png
│  └─ confusion_matrix_best.png
└─ README.md
```

---

## Reproducibility
- Fixed random seed (`random_state=42`)  
- Deterministic sklearn splits  
- Same preprocessing pipeline reused at inference  

---

## Extending / Next Steps
- Hyperparameter tuning (`C` for LogReg, `alpha` for NB`)  
- Try **Linear SVM** (LinearSVC)  
- Enrich slang dictionary & stopwords  
- Experiment with **IndoBERT** for deep contextual features  

---

## License
MIT (or your choice)
