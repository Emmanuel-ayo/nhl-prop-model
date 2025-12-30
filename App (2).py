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
df = pd.read_excel("nhl_stats.xlsx")
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
st.subheader("📊 Player Projections Table")

table_df = df_date.copy()

# Generate projections for all players

features = ["L5 Avg", "L10 Avg", "TOI_min"]

table_df = table_df.dropna(subset=features)

X_pred = table_df[features].astype(float).values  # 👈 IMPORTANT

table_df["Projected SOG"] = model.predict(X_pred)

# Select columns to display
display_cols = [
    "Name", "Team", "Opponent",
    "Projected SOG",
    "Season Avg", "L5 Avg", "L10 Avg", "TOI_min"
]

st.dataframe(
    table_df[display_cols]
    .sort_values("Projected SOG", ascending=False),
    use_container_width=True
)

st.subheader("🔍 Player Search")

player_names = df["Name"].dropna().unique()
selected_player = st.selectbox(
    "Select a player",
    sorted(player_names)
)

player_df = df[df["Name"] == selected_player].sort_values("Date", ascending=False)

latest = player_df.iloc[0]

features = ["L5 Avg", "L10 Avg", "TOI_min"]
X_player = latest[features].astype(float).values.reshape(1, -1)
proj = model.predict(X_player)[0]

c1, c2, c3 = st.columns(3)
c1.metric("Projected SOG", round(proj, 2))
c2.metric("Season Avg", round(latest["Season Avg"], 2))
c3.metric("Opponent", latest["Opponent"])

st.markdown("### 📊 Recent Games")
st.dataframe(
    player_df[
        ["Date", "Opponent", "Season Avg", "L5 Avg", "L10 Avg", "TOI_min"]
    ],
    use_container_width=True
)



