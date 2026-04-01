# 🎬 Netflix Movie Recommendation System (SVD)

A collaborative filtering recommendation engine built on the [Netflix Prize Dataset](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) using **Singular Value Decomposition (SVD)** via the `scikit-surprise` library.

---

## 📌 Project Overview

This project predicts how a user would rate a movie they haven't watched yet, and uses those predictions to generate personalised top-N recommendations.

**Approach:** Matrix Factorisation using SVD  
**Library:** [scikit-surprise](https://surpriselib.com/)  
**Dataset:** Netflix Prize Data (`combined_data_1.txt` + `movie_titles.csv`)

---

## 🗂️ Project Structure

```
netflix-svd-recommendation/
│
├── netflix_svd_recommendation.py   # Main script (all steps)
├── requirements.txt                # Python dependencies
├── .gitignore                      # Excludes data, model, checkpoints
└── README.md
```

---

## ⚙️ How It Works

1. **Load & Parse** raw Netflix rating data (movie IDs embedded as rows with NaN ratings)
2. **EDA** — Visualise rating distribution across 1–5 stars
3. **Filter** sparse movies & users using 60th-percentile benchmarks
4. **Train SVD** model on the full filtered dataset with proper train/test split (80/20)
5. **Evaluate** using RMSE & MAE on holdout test set + 3-fold cross-validation
6. **Save** trained model as `.pkl` for reuse
7. **Recommend** top-N movies for a given user (excluding already-rated ones)

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Test RMSE | ~0.97 |
| Test MAE  | ~0.76 |

> Scores may vary slightly depending on data subset used.

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/kritikkaaa/Netflix_Recommendation_SVD.git
cd Netflix_Recommendation_SVD
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
Download the Netflix Prize Dataset and place these files inside a `netflix_dataset/` folder in your Google Drive:
- `combined_data_1.txt`
- `movie_titles.csv`

> ⚠️ The raw data files are **not included** in this repo due to licensing restrictions.

### 4. Run in Google Colab
Open `netflix_svd_recommendation.py` in Google Colab and run all cells.

To get recommendations for a different user, change:
```python
USER_ID = 1331154  # Replace with any valid customer ID
```

---

## 📦 Requirements

See `requirements.txt`. Key libraries:
- `numpy`
- `pandas`
- `scikit-surprise`
- `matplotlib`
- `seaborn`

---

## 🔒 .gitignore

The following are excluded from the repo:
```
*.csv
*.txt
*.pkl
*.npy
*.h5
.env
__pycache__/
.ipynb_checkpoints/
```

---

## 🙋 Author

**Kritika Kamboj**  

---
