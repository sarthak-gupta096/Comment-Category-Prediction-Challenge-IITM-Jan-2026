# Comment Category Prediction Challenge IITM Jan_2026
This repository contains a machine learning solution for predicting the category of user comments. The project involves extensive Exploratory Data Analysis (EDA), feature engineering, text preprocessing, and an ensemble modeling approach to handle highly imbalanced multiclass data.

IIT Madras – Kaggle Competition

🏆 Final Rank: 492 / 2,744 participants

📅 Duration: 3 Months

🎯 Objective: Determine the final category assigned to each comment.

## 🚀 Project Overview
The goal of this project is to classify comments into specific categories (labels) based on their content and associated metadata (e.g., upvotes, downvotes, gender, race, etc.). The dataset is derived from a Kaggle-style competition.

## 📊 Dataset Insights
 - Imbalanced Classes: The target variable is highly imbalanced, with Class 0 representing ~57.6% of the data.
 - Skewed Features: Numerical features like upvote and downvote exhibit heavy right-skewness.
 - Categorical Diversity: Features include race, religion, and gender, which are used to capture contextual metadata.

## 🛠️ Key Features & Preprocessing
1. Text Preprocessing
 - Removal of URLs and special characters.
 - Tokenization and case normalization.
 - TF-IDF Vectorization: Utilized 1-2 n-grams with a limit of 25,000 features and sublinear TF scaling.

2. Feature Engineering
 - Engagement Metrics: Calculated vote_ratio and engagement scores.
 - Writing Style Features: Extracted exclamation_count, question_count, caps_ratio, and avg_word_length.
 - Text Stats: Derived comment_length and word_count.
 - Handling Outliers: Applied log transformation to votes and clipped extreme comment_length values.

3. Data Pipeline
 - Imputation: Missing categorical values filled with "unknown".
 - Scaling: Standard Scaling for numerical features.
 - Encoding: One-Hot Encoding for categorical features.
 - Merging: Combined sparse text features with dense numerical/categorical vectors using scipy.sparse.hstack.

## 🤖 Modeling Approach
The final model is a Soft Voting Ensemble combining:
 - Logistic Regression: Balanced class weights and liblinear solver for robust baseline performance.
 - LightGBM Classifier: GPU-accelerated gradient boosting with optimized hyperparameters (450 estimators, 128 leaves).
This ensemble approach leverages the linear strengths of Logistic Regression and the non-linear capture of LightGBM.

## 📈 Results
 - **Evaluation Metric:** Macro F1-Score.
 - Handles class imbalance effectively through class_weight='balanced' in base models.
 - Generates a submission.csv ready for competition ranking.

## 📁 Repository Structure
 - `ML_Notebook_IITM.ipynb` – Full end-to-end pipeline
 - `README.md` – Project overview

## ⚠️ Disclaimer
Completed as part of an academic Kaggle competition. Data used solely for learning and evaluation purposes.
