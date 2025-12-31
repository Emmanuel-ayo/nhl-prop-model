import streamlit as st
import pandas as pd
import joblib
import math

# -----------------------------
# Helpers
# -----------------------------
def toi_to_minutes(toi):
    if pd.isna(toi):
        return None
    if isinstance(toi, (int, float)):
        return float(toi)
    if isinstance(toi, str) and ":" in toi:
        m, s = toi.split(":")
        return int(m) + int(s) / 60
    return None


# -----------------------------
# Load data & models
# -----------------------------
df = pd.read_excel("nhl_stats.xlsx")
df["TOI_min"] = df["TOI"].apply(toi_to_minutes)

sog_model = joblib.load("sog_model.pkl")
goals_model = joblib.load("goals_model.pkl")

FEATURES = ["L5 Avg", "L10 Avg", "TOI_min"]

# -----------------------------
# UI Config
# -----------------------------
st.set_page_config(layout="wide")
st.title("🏒 NHL Player Prop Model")

stat = st.radio(
    "Select Stat",
    ["Shots on Goal", "Goals"],
    horizontal=True
)

# -----------------------------
# Date Filter
# -----------------------------
date = st.selectbox("Select Date", sorted(df["Date"].dropna().unique()))
df_date = df[df["Date"] == date]

# -----------------------------
# PLAYER SEARCH (MAIN)
# -----------------------------
st.subheader("🔍 Player Search")

player = st.selectbox(
    "Select Player",
    sorted(df_date["Name"].dropna().unique())
)

row = df_date[df_date["Name"] == player].iloc[0]

X_player = row[FEATURES].astype(float).values.reshape(1, -1)

if stat == "Shots on Goal":
    proj = sog_model.predict(X_player)[0]
else:
    proj = goals_model.predict(X_player)[0]

# -----------------------------
# Metrics
# -----------------------------
c1, c2, c3, c4 = st.columns(4)

label = "Projected SOG" if stat == "Shots on Goal" else "Projected Goals"
c1.metric(label, round(proj, 2))
c2.metric("Season Avg", round(row["Season Avg"], 2))
c3.metric("L5 Avg", round(row["L5 Avg"], 2))
c4.metric("L10 Avg", round(row["L10 Avg"], 2))

# -----------------------------
# Hit Rate
# -----------------------------
line = st.number_input(
    "Prop Line",
    value=2.5 if stat == "Shots on Goal" else 0.5,
    step=0.5
)

basis = st.radio("Hit Rate Based On", ["Season Avg", "L5 Avg", "L10 Avg"])

if stat == "Shots on Goal":
    rate = round((row[basis] / line) * 100, 1) if line > 0 else 0
else:
    # Poisson probability of scoring ≥1 goal
    lam = proj
    rate = round((1 - math.exp(-lam)) * 100, 1)

st.metric("Estimated Hit Rate", f"{rate}%")

# -----------------------------
# Projection Table
# -----------------------------
st.subheader("📊 Player Projections Table")

table_df = df_date.dropna(subset=FEATURES).copy()
X_pred = table_df[FEATURES].astype(float).values

if stat == "Shots on Goal":
    table_df["Projection"] = sog_model.predict(X_pred)
else:
    table_df["Projection"] = goals_model.predict(X_pred)

display_cols = [
    "Name", "Team", "Opponent",
    "Projection",
    "Season Avg", "L5 Avg", "L10 Avg", "TOI_min"
]

st.dataframe(
    table_df[display_cols]
    .sort_values("Projection", ascending=False),
    use_container_width=True
)

# -----------------------------
# Player History
# -----------------------------
st.subheader("📊 Recent Games")

player_hist = (
    df[df["Name"] == player]
    .sort_values("Date", ascending=False)
)

st.dataframe(
    player_hist[
        ["Date", "Opponent", "Season Avg", "L5 Avg", "L10 Avg", "TOI_min"]
    ],
    use_container_width=True
)
