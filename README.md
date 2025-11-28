# 📋 Customer Lifetime Value (CLV) Prediction System

<div align="center">

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://clv-prediction-system.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**End-to-End ML Solution for E-Commerce Customer Value Prediction**

*Predicting customer lifetime value with 85% accuracy for data-driven marketing optimization*

</div>

---

## Overview

A complete machine learning system that predicts customer lifetime value (CLV) using RFM analysis, advanced feature engineering, and multiple ML models. Enables e-commerce businesses to:

- 🇨 Identify high-value customers for targeted marketing
- 💵 Optimize customer acquisition & retention budgets
- 🏑 Segment customers for personalized strategies
- 📋 Forecast revenue impact of marketing campaigns

**Business Impact:**
- ✅ **85% Accuracy** on CLV predictions
- ✅ **Top 20%** high-value customers identified
- ✅ **3 ML Models** compared (Linear, RF, XGBoost)
- ✅ **ROI Calculator** included in dashboard

---

## Problem Statement

E-commerce companies struggle to identify which customers will generate the most revenue over time. Without proper CLV prediction, businesses waste marketing budgets on low-value customers while missing opportunities with high-value prospects.

**Challenge:** Predicting CLV accurately from transaction history with handling of right-skewed revenue distribution.

---

## Key Features

- **RFM Analysis:** Recency, Frequency, Monetary value segmentation
- **Advanced Feature Engineering:** Customer behavioral & temporal features
- **Multiple Models:** Linear Regression, Random Forest, XGBoost comparison
- **Business Dashboard:** Interactive Streamlit UI with insights
- **ROI Calculator:** Estimate marketing campaign returns
- **Customer Segmentation:** A/B testing framework for marketing strategies
- **Production Ready:** Scalable architecture with deployment configs

---

## Dataset

- **Source:** E-commerce transaction data
- **Records:** 100,000+ customer transactions
- **Features:** Purchase frequency, recency, monetary value, behavioral patterns
- **Target:** Customer lifetime value (continuous regression)

---

## Model Performance

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Linear Regression | 0.72 | 850 | 620 |
| Random Forest | 0.81 | 450 | 280 |
| **XGBoost** | **0.85** | **380** | **220** |

---

## Business Applications

🏑 **Customer Segmentation:** High/Medium/Low value groups  
💵 **Marketing Spend:** Allocate budget based on CLV potential  
📋 **Retention Programs:** Target high-value at-risk customers  
🎲 **A/B Testing:** Compare strategies by CLV segments  

---

## Installation

```bash
git clone https://github.com/Jaimin-prajapati-ds/clv-prediction-system.git
cd clv-prediction-system
pip install -r requirements.txt
```

---

## Usage

```bash
# Run predictions
python src/predict.py

# Launch dashboard
streamlit run app.py
```

---

## Technologies

Python, Pandas, Scikit-learn, XGBoost, Streamlit, SQLite

---

## License

MIT License
