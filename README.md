# Customer-churn-ltv-engine-proj1
production level data analytics project -customer churn prediction and lifetime value engine
# Customer Churn Prediction & Lifetime Value (LTV) Engine

## Project Overview

A predictive analytics system designed for telecommunications or subscription-based businesses.

The system uses historical customer demographic data, billing information, and service usage metrics to:

- Identify customers at high risk of cancellation (churn)
- Predict Customer Lifetime Value (LTV)
- Help marketing teams prioritize high-value retention campaigns

## Data Source

**Telco Customer Churn Dataset**

The dataset contains over 7,000 rows of customer data, including:

- Tenure
- Monthly charges
- Contract types
- Internet service details
- Churn status

## Tech Stack

- Python
- SQL
- PostgreSQL
- SQLAlchemy
- Pandas
- Scikit-Learn
- XGBoost
- SHAP
- FastAPI
- Apache Superset / Metabase

## Project Timeline

### Week 1 — Data Ingestion & Exploratory Data Analysis
- Set up PostgreSQL database
- Load the Telco dataset
- Perform EDA
- Handle missing values
- Encode categorical variables
- Prepare baseline analytics report

### Week 2 — Feature Engineering & Predictive Modeling
- Engineer useful features
- Train classification models
- Evaluate using Precision, Recall and F1-Score
- Implement SHAP for model explainability

### Week 3 — LTV Calculation & API Development
- Forecast expected lifetime revenue
- Develop regression models
- Build FastAPI prediction service
- Support single-customer and batch predictions

### Week 4 — Visualization & Deployment
- Connect dashboard to database/API
- Build interactive dashboards
- Show churn risk and LTV segments
- Containerize the application using Docker
- Complete technical documentation

## Expected Impact

- Reduce customer acquisition costs through proactive retention
- Optimize marketing budgets
- Identify high-value customer segments
- Support data-driven retention decisions



## Project Status

🚧 Development in Progress
