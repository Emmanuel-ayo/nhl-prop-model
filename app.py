import streamlit as st
import pandas as pd
import joblib

def toi_to_minutes(toi):
    if pd.isna(toi):
        return None
    if isinstance(toi, (int, float)):
        return float(toi)
    if isinstance(toi, str) and ":" in toi:
        m, s = toi.split(":")
        return int(m) + int(s) / 60
    return None

# Load data and model
df = pd.read_excel("nhl-stats-7-seasons (1).xlsx")
df["TOI_min"] = df["TOI"].apply(toi_to_minutes)

model = joblib.load("sog_model.pkl")

st.set_page_config(layout="wide")
st.title("🏒 NHL SOG Prop Model")

date = st.selectbox("Date", sorted(df["Date"].dropna().unique()))
df_date = df[df["Date"] == date]

player = st.selectbox("Player", df_date["Name"].unique())
row = df_date[df_date["Name"] == player].iloc[0]

features = ["L5 Avg", "L10 Avg", "TOI_min"]
X = row[features].values.reshape(1, -1)
projection = model.predict(X)[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Projected SOG", round(projection, 2))
c2.metric("Season Avg", row["Season Avg"])
c3.metric("L5 Avg", row["L5 Avg"])
c4.metric("L10 Avg", row["L10 Avg"])

line = st.number_input("Shot Line", value=2.5, step=0.5)

basis = st.radio("Hit Rate Based On", ["Season Avg", "L5 Avg", "L10 Avg"])
rate = round((row[basis] / line) * 100, 1) if line > 0 else 0

st.metric("Hit Rate", f"{rate}%")
