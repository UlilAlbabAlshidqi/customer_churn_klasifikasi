import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Load artifacts ===
model = joblib.load("xgb_best_model.joblib")
preprocessor = joblib.load("preprocessor.joblib")
numerical_columns = joblib.load("numerical_columns.joblib")   # kolom numerik asli
scaled_columns = joblib.load("scaled_columns.joblib")         # kolom numerik setelah scaling
ohe_columns = joblib.load("ohe_columns.joblib")               # kolom kategori setelah one-hot

# === Streamlit UI ===
st.title("Customer Churn Prediction")

states = [
    'KS','OH','NJ','OK','AL','MA','MO','LA','WV','IN','RI','IA','MT','NY','ID','VT','VA','TX',
    'FL','CO','AZ','SC','NE','WY','HI','IL','NH','GA','AK','MD','AR','WI','OR','MI','DE','UT',
    'CA','MN','SD','NC','WA','NM','NV','DC','KY','ME','MS','TN','PA','CT','ND'
]

state = st.selectbox("State", sorted(states))
account_length = st.number_input("Account Length", min_value=1, max_value=300, value=100)
area_code = st.selectbox("Area Code", [408, 415, 510])
intl_plan = st.selectbox("Int'l Plan", ["yes", "no"])
vmail_plan = st.selectbox("VMail Plan", ["yes", "no"])
vmail_message = st.number_input("VMail Message", min_value=0, value=0)

day_mins = st.number_input("Day Mins", min_value=0.0, value=100.0)
day_calls = st.number_input("Day Calls", min_value=0, value=100)
day_charge = round(day_mins * 0.17, 2)

eve_mins = st.number_input("Eve Mins", min_value=0.0, value=200.0)
eve_calls = st.number_input("Eve Calls", min_value=0, value=100)
eve_charge = round(eve_mins * 0.085, 2)

night_mins = st.number_input("Night Mins", min_value=0.0, value=200.0)
night_calls = st.number_input("Night Calls", min_value=0, value=100)
night_charge = round(night_mins * 0.045, 2)

intl_mins = st.number_input("Intl Mins", min_value=0.0, value=10.0)
intl_calls = st.number_input("Intl Calls", min_value=0, value=1)
intl_charge = round(intl_mins * 0.27, 2)

custserv_calls = st.number_input("CustServ Calls", min_value=0, value=2)

# === Bungkus ke DataFrame ===
data = pd.DataFrame([{
    "State": state,
    "Account Length": account_length,
    "Area Code": area_code,
    "Int'l Plan": intl_plan,
    "VMail Plan": vmail_plan,
    "VMail Message": vmail_message,
    "Day Mins": day_mins,
    "Day Calls": day_calls,
    "Day Charge": day_charge,
    "Eve Mins": eve_mins,
    "Eve Calls": eve_calls,
    "Eve Charge": eve_charge,
    "Night Mins": night_mins,
    "Night Calls": night_calls,
    "Night Charge": night_charge,
    "Intl Mins": intl_mins,
    "Intl Calls": intl_calls,
    "Intl Charge": intl_charge,
    "CustServ Calls": custserv_calls
}])

# === Predict button ===
if st.button("Predict"):
    # 1️⃣ Log transform untuk kolom yang dibutuhkan
    data['Vmail Message_log'] = np.log1p(data['VMail Message'])
    data['Intl Calls_log'] = np.log1p(data['Intl Calls'])

    # 2️⃣ Ambil kolom numerik untuk preprocessor
    numeric_for_preprocessor = numerical_columns.copy()
    numeric_for_preprocessor.remove('VMail Message')
    numeric_for_preprocessor.remove('Intl Calls')
    numeric_for_preprocessor += ['Vmail Message_log', 'Intl Calls_log']

    # 3️⃣ Transform numerik
    X_numeric_scaled = preprocessor.transform(data[numeric_for_preprocessor])
    df_numeric_scaled = pd.DataFrame(X_numeric_scaled, columns=scaled_columns)

    # 4️⃣ One-hot encoding kategori
    data_categorical = data.drop(columns=numerical_columns)
    data_categorical_encoded = pd.get_dummies(data_categorical)
    data_categorical_encoded = data_categorical_encoded.reindex(columns=ohe_columns, fill_value=0)

    # 5️⃣ Gabungkan numerik + kategori
    X_final = pd.concat([df_numeric_scaled, data_categorical_encoded], axis=1)

    # 6️⃣ Predict
    pred = model.predict(X_final)[0]
    proba = model.predict_proba(X_final)[0, 1]

    st.subheader("Hasil Prediksi")
    st.write("Churn Prediction:", "Yes" if pred == 1 else "No")
    st.write("Probability of Churn:", f"{round(proba*100,2)} %")
