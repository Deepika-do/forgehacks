# AI Customer Churn Predictor & Retention Advisor

## Problem Statement
Subscription businesses lose revenue when customers cancel, and teams don't know
whom to contact first. This project predicts which customers are likely to churn,
explains why, estimates the revenue at risk, and recommends a specific retention
action for each customer.

## Solution
1. Customer data (usage, billing, support history) is loaded (sample data or CSV).
2. A Random Forest model predicts churn probability per customer.
3. Feature importance shows the main reasons for churn.
4. Revenue at risk = churn probability x yearly charges.
5. Rule-based recommendations give an action per customer.
6. A Streamlit dashboard shows everything and exports an action list.

## Run
```
pip install pandas numpy scikit-learn streamlit
python churn_core.py        # quick terminal demo
streamlit run app.py        # dashboard
```

## Tech
Python, pandas, scikit-learn, Streamlit

## Future Scope
Real company data, SHAP explanations, email automation, LLM-written retention messages.
