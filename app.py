import pandas as pd
import streamlit as st
from churn_core import generate_data, train, score_customers

st.set_page_config(page_title="Churn Insights AI", layout="wide")
st.title("📉 AI Customer Churn Predictor & Retention Advisor")
st.caption("Turns customer data into churn predictions and recommended actions.")

uploaded = st.sidebar.file_uploader("Upload customer CSV (optional)", type="csv")
st.sidebar.write("No file? Sample data is used.")
st.sidebar.caption("CSV columns: customer_id, tenure_months, monthly_charges, "
                   "support_tickets, late_payments, logins_per_month, contract, churned")

df = pd.read_csv(uploaded) if uploaded else generate_data()
model, metrics, importance = train(df)
scored = score_customers(model, df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", len(scored))
c2.metric("High-risk customers", int((scored.risk_level == "High").sum()))
c3.metric("Annual revenue at risk", f"{scored.revenue_at_risk.sum():,.0f}")
c4.metric("Model AUC", f"{metrics['auc']:.2f}")

left, right = st.columns(2)
with left:
    st.subheader("Why do customers leave? (top drivers)")
    st.bar_chart(importance)
with right:
    st.subheader("Risk distribution")
    st.bar_chart(scored["risk_level"].value_counts())

st.subheader("🎯 Customers to save first")
level = st.multiselect("Filter risk level", ["High", "Medium", "Low"], default=["High"])
view = scored[scored.risk_level.isin(level)]
st.dataframe(view[["customer_id", "churn_risk", "risk_level", "revenue_at_risk",
                   "recommended_action"]], use_container_width=True)
st.download_button("Download action list (CSV)", view.to_csv(index=False),
                   "retention_actions.csv")

st.subheader("💡 Key insight")
top = importance.index[0].replace("_", " ")
st.info(f"The strongest churn driver is **{top}**. Focus retention on "
        f"{int((scored.risk_level == 'High').sum())} high-risk customers to protect "
        f"{scored[scored.risk_level == 'High'].revenue_at_risk.sum():,.0f} in yearly revenue.")
