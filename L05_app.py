import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "./models/model_heart_cv.joblib"
DATASET_PATH = "./datasets/heart.csv"
SCALER_PATH = "./models/scaler_heart.joblib"

df = pd.read_csv(DATASET_PATH)
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
features = X.columns.tolist()

st.title("🩺Heart Disease Prediction🩺")
mode = st.sidebar.radio("Choose input mode:", ["Sample from dataset", "Manual input"])

if mode == "Sample from dataset":
    max_index = len(X) - 1
    idx = st.number_input(
        "Sample index",
        min_value=0,
        max_value=max_index,
        value=0,
        step=1,
        help=f"Select a row from the dataset (0 to {max_index})"
    )
    if st.button("Predict"):
        row = X.iloc[[idx]]
        scaled = scaler.transform(row)
        prob = model.predict_proba(scaled)[0]
        pred = model.predict(scaled)[0]
        st.write("Input Features:")
        st.dataframe(row)
        st.write("Prediction:")
        if pred == 1:
            st.error(f"Heart Disease (probability: %{prob[1]*100:.2f})")
        else:
            st.success(f"No Heart Disease (probability: %{prob[0]*100:.2f})")
        st.write("Actual:", "Heart Disease" if y.iloc[idx] == 1 else "No Heart Disease")

else:
    user_input = {}
    for f in features:
        min_val = X[f].min()
        max_val = X[f].max()
        mean_val = X[f].mean()
        dtype = X[f].dtype
        help_text = f"Type: {dtype}, Range: [{min_val} - {max_val}]"
        if pd.api.types.is_integer_dtype(dtype):
            user_input[f] = st.number_input(f, int(min_val), int(max_val), int(mean_val), step=1, help=help_text)
        else:
            user_input[f] = st.number_input(f, float(min_val), float(max_val), float(mean_val), format="%.2f", help=help_text)

    if st.button("Predict"):
        row = pd.DataFrame([user_input])
        scaled = scaler.transform(row)
        prob = model.predict_proba(scaled)[0]
        pred = model.predict(scaled)[0]
        st.write("Input Features:")
        st.dataframe(row)
        st.write("Prediction:")
        if pred == 1:
            st.error(f"Heart Disease (probability: %{prob[1]*100:.2f})")
        else:
            st.success(f"No Heart Disease (probability: %{prob[0]*100:.2f})")
