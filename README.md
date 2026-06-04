# Comment-Category-Prediction-Challenge
This project delivers an end-to-end Machine Learning pipeline designed to solve a complex, highly imbalanced multi-class text classification problem. The objective is to automatically categorize user comments into one of four distinct classes based on textual content and metadata.

Project Overview: Comment Category Prediction Challenge
This project delivers an end-to-end Machine Learning pipeline designed to solve a complex, highly imbalanced multi-class text classification problem. The objective is to automatically categorize user comments into one of four distinct classes based on textual content and metadata.

Key Technical Implementations:
Advanced Text Representation: Leveraged a dual-vectorization strategy combining Word-level and Character-level TF-IDF TfidfVectorizer to capture both semantic meaning and stylistic writing nuances (such as typos or specific punctuation habits).

Feature Engineering: Extracted and scaled 24 diverse numerical and categorical features, including engagement metrics (upvotes/downvotes), temporal patterns (hour, dayofweek), and structural text properties (capitalization ratios, punctuation densities, and unique word ratios).

Ensemble Modeling Strategy: Built a robust Weighted Soft-Voting Ensemble combining three structurally diverse classifiers optimized via RandomizedSearchCV:

LightGBM: To capture non-linear feature interactions and metadata patterns.

Logistic Regression (Saga Solver): For stable, high-dimensional text feature scaling.

SGD Classifier (Log Loss): To inject a fast, regularized linear perspective.

Imbalance Mitigation & Threshold Tuning: Addressed severe data sparsity (where the rarest minority class comprised less than 3% of the dataset) by incorporating balanced class weights and developing a customized post-prediction threshold tuning algorithm to maximize minority class recall.
