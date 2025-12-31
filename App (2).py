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


sog_model = joblib.load("sog_model.pkl")
goals_model = joblib.load("goals_model.pkl")


stat = st.radio(
    "Select Stat",
    ["Shots on Goal", "Goals"],
    horizontal=True
)

st.set_page_config(layout="wide")
st.title("🏒 NHL SOG Prop Model")

date = st.selectbox("Date", sorted(df["Date"].dropna().unique()))
df_date = df[df["Date"] == date]



line = st.number_input("Shot Line", value=2.5, step=0.5)

basis = st.radio("Hit Rate Based On", ["Season Avg", "L5 Avg", "L10 Avg"])
if stat == "Shots on Goal":
    rate = round((row[basis] / line) * 100, 1)
else:
    # Goals probability (Poisson approx)
    import math
    lam = proj
    rate = round((1 - math.exp(-lam)) * 100, 1)


st.metric("Hit Rate", f"{rate}%")
st.subheader("📊 Player Projections Table")

table_df = df_date.copy()

# Generate projections for all players

features = ["L5 Avg", "L10 Avg", "TOI_min"]

table_df = table_df.dropna(subset=features)

X_pred = table_df[features].astype(float).values  # 👈 IMPORTANT

if stat == "Shots on Goal":
    table_df["Projection"] = sog_model.predict(X_pred)
else:
    table_df["Projection"] = goals_model.predict(X_pred)


# Select columns to display
display_cols = [
    "Name", "Team", "Opponent",
    "Projection",
    "Season Avg", "L5 Avg", "L10 Avg", "TOI_min"
]


st.dataframe(
    table_df[display_cols]
    .sort_values("Projected SOG", ascending=False),
    use_container_width=True
)
st.write("PLAYER SEARCH LOADED")

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
if stat == "Shots on Goal":
    proj = sog_model.predict(X_player)[0]
else:
    proj = goals_model.predict(X_player)[0]


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

st.markdown("---")
st.subheader("🔍 Player Search")

player_names = sorted(df["Name"].dropna().unique())
selected_player = st.selectbox("Search player", player_names)

player_df = df[df["Name"] == selected_player].sort_values("Date", ascending=False)
latest = player_df.iloc[0]

features = ["L5 Avg", "L10 Avg", "TOI_min"]
X_player = latest[features].astype(float).values.reshape(1, -1)
if stat == "Shots on Goal":
    proj = sog_model.predict(X_player)[0]
else:
    proj = goals_model.predict(X_player)[0]

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
