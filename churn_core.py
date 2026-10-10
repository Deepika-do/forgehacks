"""Core logic: data -> model -> insights -> recommendations."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

FEATURES = ["tenure_months", "monthly_charges", "support_tickets",
            "late_payments", "logins_per_month", "contract_yearly"]


def generate_data(n=2000, seed=42):
    """Create realistic sample customer data for a subscription business."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n)],
        "tenure_months": rng.integers(1, 61, n),
        "monthly_charges": rng.normal(60, 20, n).clip(15, 130).round(2),
        "support_tickets": rng.poisson(1.5, n),
        "late_payments": rng.poisson(0.8, n),
        "logins_per_month": rng.poisson(12, n),
        "contract": rng.choice(["Monthly", "Yearly"], n, p=[0.65, 0.35]),
    })
    z = (-0.5 - 0.05 * df.tenure_months + 0.02 * df.monthly_charges
         + 0.35 * df.support_tickets + 0.45 * df.late_payments
         - 0.08 * df.logins_per_month
         - 1.2 * (df.contract == "Yearly") + 1.8)
    p = 1 / (1 + np.exp(-z))
    df["churned"] = (rng.random(n) < p).astype(int)
    return df


def prepare(df):
    X = df.copy()
    X["contract_yearly"] = (X["contract"] == "Yearly").astype(int)
    return X[FEATURES]


def train(df):
    X, y = prepare(df), df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                              random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    metrics = {"accuracy": accuracy_score(y_te, proba > 0.5),
               "auc": roc_auc_score(y_te, proba)}
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    return model, metrics, importance


def recommend(row):
    """Rule-based action for each at-risk customer."""
    if row["late_payments"] >= 2:
        return "Offer flexible payment plan / auto-pay reminder"
    if row["support_tickets"] >= 3:
        return "Assign priority support call to fix open issues"
    if row["logins_per_month"] <= 6:
        return "Send re-engagement email with feature tips"
    if row["contract"] == "Monthly":
        return "Offer discount to upgrade to yearly plan"
    return "Send loyalty reward / thank-you offer"


def score_customers(model, df):
    out = df.copy()
    out["churn_risk"] = model.predict_proba(prepare(df))[:, 1].round(3)
    out["risk_level"] = pd.cut(out["churn_risk"], [-0.01, 0.4, 0.7, 1.0],
                               labels=["Low", "Medium", "High"])
    out["revenue_at_risk"] = (out["churn_risk"] * out["monthly_charges"] * 12).round(2)
    out["recommended_action"] = out.apply(
        lambda r: recommend(r) if r["risk_level"] != "Low" else "No action needed", axis=1)
    return out.sort_values("revenue_at_risk", ascending=False)


if __name__ == "__main__":
    data = generate_data()
    model, metrics, imp = train(data)
    scored = score_customers(model, data)
    print("Metrics:", metrics)
    print("\nTop churn drivers:\n", imp)
    print("\nTop 5 customers to save:\n",
          scored[["customer_id", "churn_risk", "revenue_at_risk", "recommended_action"]].head())
    print("\nTotal annual revenue at risk: ₹/$", round(scored.revenue_at_risk.sum(), 2))
