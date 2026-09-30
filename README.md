# Credit Card Fraud Detection & Financial Data Preprocessing

## Overview
Financial datasets are notoriously imbalanced—fraudulent transactions make up only a tiny fraction of the total data. In this project, I cleaned and preprocessed transaction records, handled extreme outliers, and analyzed class imbalance to better understand fraud patterns.

## What I Did
- **Data Preprocessing & Scaling:** Cleaned missing values, treated extreme transaction outliers, and scaled feature values for analysis.
- **Class Imbalance Analysis:** Examined the distribution between normal and fraudulent transactions to see how skewed the data really is.
- **Exploratory Data Analysis (EDA):** Generated correlation heatmaps and distribution plots (`fraud_by_hour.png`, `top_correlations.png`) to spot time-based and feature-based risk patterns.

## Tech Stack
- **Python**
- **Pandas & NumPy** (Data manipulation and cleaning)
- **Matplotlib & Seaborn** (Data visualization)

## Key Takeaways
- Fraudulent transactions are heavily concentrated in specific time windows and transaction amounts.
- Standardizing features and isolating rare fraud instances is essential before feeding this data into any ML classification model.
