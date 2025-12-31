import streamlit as st
import pandas as pd
import joblib
import math



# ---------------------------
# Helper function
# ---------------------------
def toi_to_minutes(toi):
    if pd.isna(toi):
        return None
    if isinstance(toi, (int, float)):
        return float(toi)
    if isinstance(toi, str) and ":" in toi:
        m, s = toi.split(":")
        return int(m) + int(s) / 60
    return None

# ---------------------------
# Load data & models
# ---------------------------
df = pd.read_excel("nhl_stats.xlsx")
df["TOI_min"] = df["TOI"].apply(toi_to_minutes)

sog_model = joblib.load("sog_model.pkl")
goals_model = joblib.load("goals_model.pkl")

# ---------------------------
# UI setup
# ---------------------------
st.set_page_config(layout="wide")
st.title("🏒 NHL Player Prop Model")

stat = st.radio("Select Stat", ["Shots on Goal", "Goals"], horizontal=True)

date = st.selectbox("Game Date", sorted(df["Date"].dropna().unique()))
df_date = df[df["Date"] == date].copy()

# ---------------------------
# Player selection (case-insensitive & stripped)
# ---------------------------
df_date["Name_clean"] = df_date["Name"].str.strip().str.lower()
players_available = sorted(df_date["Name"].dropna().unique())
player = st.selectbox("Select Player", players_available)

row = df_date[df_date["Name_clean"] == player.strip().lower()].iloc[0]

# ---------------------------
# Projection
# ---------------------------
features = ["L5 Avg", "L10 Avg", "TOI_min"]
X_player = row[features].astype(float).values.reshape(1, -1)

if stat == "Shots on Goal":
    proj = sog_model.predict(X_player)[0]
else:
    proj = goals_model.predict(X_player)[0]

c1, c2, c3 = st.columns(3)
c1.metric("Projection", round(proj, 2))
c2.metric("Season Avg", round(row.get("Season Avg", 0), 2))
c3.metric("Opponent", row.get("Opponent", "N/A"))

# ---------------------------
# Hit Rate (safe calculation)
# ---------------------------
st.markdown("### 🎯 Hit Rate")

line = st.number_input(
    "Prop Line",
    value=2.5 if stat == "Shots on Goal" else 0.5,
    step=0.5
)

basis = st.radio("Hit Rate Based On", ["Season Avg", "L5 Avg", "L10 Avg"])

basis_value = row.get(basis, 0)
if stat == "Shots on Goal":
    if line > 0 and basis_value is not None:
        rate = round((basis_value / line) * 100, 1)
    else:
        rate = 0
else:
    lam = float(proj)
    rate = round((1 - math.exp(-lam)) * 100, 1)

st.metric("Estimated Hit Rate", f"{rate}%")

# ---------------------------
# Projection Table
# ---------------------------
st.markdown("### 📊 All Player Projections")

table_df = df_date.dropna(subset=features).copy()
X_pred = table_df[features].astype(float).values

if stat == "Shots on Goal":
    table_df["Projection"] = sog_model.predict(X_pred)
else:
    table_df["Projection"] = goals_model.predict(X_pred)

st.dataframe(
    table_df[
        ["Name", "Team", "Opponent", "Projection", "Season Avg", "L5 Avg", "L10 Avg", "TOI_min"]
    ].sort_values("Projection", ascending=False),
    use_container_width=True
)
