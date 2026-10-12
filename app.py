import joblib, pandas as pd, streamlit as st

bundle = joblib.load("heart_model.joblib")
lr, cols = bundle["model"], bundle["columns"]
scaler, coef = lr[0], lr[-1].coef_[0]
HIDE = {"sex"}

st.title("Heart Risk Explainer (learning project)")
st.warning("Not a medical diagnosis. Trained on a 303-row public dataset for learning only.")

defaults = {"age": 50, "sex": 0, "cp": 0, "trestbps": 120, "chol": 200, "fbs": 0,
            "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 1.0,
            "slope": 1, "ca": 0, "thal": 2}

vals = {}
for c in cols:
    d = defaults[c]
    if isinstance(d, float):
        vals[c] = st.number_input(c, value=d)
    else:
        vals[c] = st.number_input(c, value=d, step=1)

if st.button("Check"):
    r = pd.DataFrame([vals], columns=cols)
    p = lr.predict_proba(r)[0][1]
    contrib = pd.Series(coef * scaler.transform(r)[0], index=cols).drop(HIDE)
    up = contrib[contrib > 0.1].sort_values(ascending=False).head(3)
    down = contrib[contrib < -0.1].sort_values().head(3)
    st.subheader(f"Risk score: {p:.0%}")
    st.write("Factors pushing risk up:", up.round(2).to_dict())
    st.write("Factors pushing risk down:", down.round(2).to_dict())