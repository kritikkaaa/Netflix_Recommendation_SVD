# ============================================================
# Netflix SVD Recommendation System
# ============================================================
# Description: Collaborative Filtering using SVD (Singular
#              Value Decomposition) on the Netflix Prize Dataset
# Author: Kritika Kamboj
# ============================================================

# ── 0. Dependencies ──────────────────────────────────────────
!pip uninstall -y numpy
!pip install numpy==1.26.4
!pip install scikit-surprise 

# ── 1. Imports ───────────────────────────────────────────────
from google.colab import drive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy

# ── 2. Mount Google Drive ────────────────────────────────────
drive.mount('/content/drive')

# ── 3. Load Raw Dataset ──────────────────────────────────────
netflix_dataset = pd.read_csv(
    '/content/drive/MyDrive/netflix_dataset/combined_data_1.txt',
    header=None,
    names=['Cust_Id', 'Rating'],
    usecols=[0, 1]
)

print("Raw Dataset Shape:", netflix_dataset.shape)
print(netflix_dataset.head())

# ── 4. Exploratory Data Analysis ─────────────────────────────

print("\nData Types:\n", netflix_dataset.dtypes)
print("\nMissing Values:\n", netflix_dataset.isnull().sum())

# Count movies (rows where Rating is NaN contain movie IDs)
movie_count = netflix_dataset['Rating'].isnull().sum()
print(f"\nTotal Movies: {movie_count}")

# Count unique customers (excluding movie ID rows)
customer_count = netflix_dataset['Cust_Id'].nunique() - movie_count
print(f"Total Unique Customers: {customer_count}")

# Total number of ratings
total_ratings = netflix_dataset['Cust_Id'].count() - movie_count
print(f"Total Ratings: {total_ratings}")

# Star rating distribution
stars = netflix_dataset.groupby('Rating')['Rating'].agg(['count'])
ax = stars.plot(kind='barh', legend=False, figsize=(9, 6))
plt.title(
    f'Total pool: {movie_count} Movies, {customer_count} Customers, {total_ratings} Ratings',
    fontsize=16
)
plt.xlabel("Count")
plt.ylabel("Star Rating")
plt.grid(True)
plt.tight_layout()
plt.show()

# ── 5. Feature Engineering: Extract Movie IDs ────────────────
netflix_dataset['Movie_id'] = netflix_dataset['Cust_Id'].apply(
    lambda x: x[:-1] if isinstance(x, str) and ':' in x else None
)
netflix_dataset['Movie_id'] = netflix_dataset['Movie_id'].ffill()

# Drop rows where Rating is NaN (those were movie ID rows)
netflix_dataset.dropna(inplace=True)

# Fix data types
netflix_dataset["Cust_Id"] = netflix_dataset["Cust_Id"].astype(int)
netflix_dataset["Movie_id"] = netflix_dataset["Movie_id"].astype(int)

print("\nCleaned Dataset Shape:", netflix_dataset.shape)
print(netflix_dataset.head())

# ── 6. Filter Sparse Movies & Users (Benchmarking) ───────────

# Movies with fewer than 60th percentile ratings are dropped
dataset_movie_summary = netflix_dataset.groupby('Movie_id')['Rating'].agg(['count'])
movie_benchmark = round(dataset_movie_summary['count'].quantile(0.6), 0)
drop_movie_list = dataset_movie_summary[dataset_movie_summary['count'] < movie_benchmark].index
print(f"\nMovie Benchmark: {movie_benchmark} | Movies to drop: {len(drop_movie_list)}")

# Customers with fewer than 60th percentile ratings are dropped
dataset_cust_summary = netflix_dataset.groupby('Cust_Id')['Rating'].agg(['count'])
cust_benchmark = round(dataset_cust_summary['count'].quantile(0.6), 0)
cust_to_drop = dataset_cust_summary[dataset_cust_summary['count'] < cust_benchmark].index
print(f"Customer Benchmark: {cust_benchmark} | Customers to drop: {len(cust_to_drop)}")

# Apply filters
netflix_dataset = netflix_dataset[~netflix_dataset['Movie_id'].isin(drop_movie_list)]
netflix_dataset = netflix_dataset[~netflix_dataset['Cust_Id'].isin(cust_to_drop)]
print(f"\nAfter trimming, dataset shape: {netflix_dataset.shape}")

# ── 7. Load Movie Titles ──────────────────────────────────────
df_title = pd.read_csv(
    '/content/drive/MyDrive/netflix_dataset/movie_titles.csv',
    encoding='latin',
    header=None,
    usecols=[0, 1, 2],
    names=['Movie_id', 'Year', 'Movie_Title']
)
print("\nMovie Titles Sample:\n", df_title.head())

# ── 8. Prepare Data for Surprise Library ─────────────────────
reader = Reader()

# FIX: Use full filtered dataset instead of just 100k rows
data = Dataset.load_from_df(
    netflix_dataset[['Cust_Id', 'Movie_id', 'Rating']],
    reader
)

# ── 9. Train / Test Split ─────────────────────────────────────
# FIX: Added proper holdout evaluation instead of only cross-validation
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# ── 10. Train SVD Model ───────────────────────────────────────
model = SVD(random_state=42)
model.fit(trainset)
print("\nModel trained successfully.")

# Evaluate on holdout test set
predictions = model.test(testset)
print(f"\nTest Set RMSE: {accuracy.rmse(predictions):.4f}")
print(f"Test Set MAE : {accuracy.mae(predictions):.4f}")

# Additional cross-validation for robust performance estimate
print("\nRunning 3-fold Cross Validation...")
cv_results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

# ── 11. Save Trained Model ────────────────────────────────────
# FIX: Persist model so you don't retrain every time
MODEL_PATH = '/content/drive/MyDrive/netflix_dataset/svd_model.pkl'
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)
print(f"\nModel saved to: {MODEL_PATH}")

# To load the model later:
# with open(MODEL_PATH, 'rb') as f:
#     model = pickle.load(f)

# ── 12. Generate Recommendations for a User ──────────────────
# FIX: Made user ID a variable + filter out already-rated movies (not just sparse ones)

USER_ID = 1331154  # Change this to recommend for any user

user_ratings = netflix_dataset[netflix_dataset['Cust_Id'] == USER_ID]
movies_rated_by_user = user_ratings['Movie_id'].nunique()
print(f"\nUser {USER_ID} has rated {movies_rated_by_user} unique movies.")

# FIX: Exclude movies already rated by the user (not drop_movie_list)
already_rated = set(user_ratings['Movie_id'].tolist())

# Filter title list to only movies NOT yet rated by user
candidate_movies = df_title[~df_title['Movie_id'].isin(already_rated)].copy()

# Predict scores for all candidate movies
candidate_movies['Estimate_Score'] = candidate_movies['Movie_id'].apply(
    lambda x: model.predict(USER_ID, x).est
)

# Sort by predicted score
candidate_movies_sorted = candidate_movies.sort_values('Estimate_Score', ascending=False)

# Top 10 recommendations
top10 = candidate_movies_sorted.head(10)
print(f"\nTop 10 Movie Recommendations for User {USER_ID}:\n")
print(top10[['Movie_id', 'Year', 'Movie_Title', 'Estimate_Score']].to_string(index=False))

# ── 13. Visualise Top 10 Recommendations ─────────────────────
plt.figure(figsize=(10, 6))
sns.barplot(
    data=top10,
    x='Estimate_Score',
    y='Movie_Title',
    palette='viridis'
)
plt.title(f'Top 10 Predicted Ratings for User {USER_ID}', fontsize=14)
plt.xlabel('Estimated Rating')
plt.ylabel('Movie Title')
plt.xlim(0, 5)
plt.tight_layout()
plt.show()
