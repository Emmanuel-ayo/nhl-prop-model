import streamlit as st
import pandas as pd
import joblib
import math
import requests

st.set_page_config(layout="wide")
st.title("🏒 NHL Player Prop Model (Hybrid API + Dataset)")

# ---------------------------
# Load models
# ---------------------------
sog_model = joblib.load("sog_model.pkl")
goals_model = joblib.load("goals_model.pkl")

# ---------------------------
# Load local dataset fallback
# ---------------------------
df = pd.read_excel("nhl_stats.xlsx")
df["TOI_min"] = df["TOI"].apply(lambda x: int(x.split(':')[0]) + int(x.split(':')[1])/60 if isinstance(x, str) and ':' in x else 20.0)

# ---------------------------
# Helper Functions
# ---------------------------
def toi_to_minutes(toi):
    if not toi or ":" not in toi:
        return 20.0  # default
    m, s = toi.split(":")
    return int(m) + int(s)/60

def get_player_id(player_name):
    """Search NHL API for player ID by full name"""
    url = f"https://statsapi.web.nhl.com/api/v1/people?fullName={player_name}"
    try:
        resp = requests.get(url, timeout=5).json()
        people = resp.get("people", [])
        if people:
            return people[0]["id"], people[0]["fullName"], people[0]["currentTeam"]["name"]
    except requests.exceptions.RequestException:
        return None, None, None
    return None, None, None

def get_season_stats(player_id, season="20242025"):
    """Fetch season stats for a player from NHL API"""
    url = f"https://statsapi.web.nhl.com/api/v1/people/{player_id}/stats?stats=statsSingleSeason&season={season}"
    try:
        resp = requests.get(url, timeout=5).json()
        stats = resp["stats"][0]["splits"]
        if stats:
            stat = stats[0]["stat"]
            return {
                "games": stat.get("games", 0),
                "goals": stat.get("goals", 0),
                "shots": stat.get("shots", 0),
                "assists": stat.get("assists", 0),
                "timeOnIcePerGame": stat.get("timeOnIcePerGame", "20:00")
            }
    except requests.exceptions.RequestException:
        return None
    return None

# ---------------------------
# User Input
# ---------------------------
player_name = st.text_input("Enter NHL Player Name")

if player_name:
    # Try NHL API first
    player_id, full_name, team = get_player_id(player_name)
    if player_id:
        season_stats = get_season_stats(player_id)
        if season_stats:
            # Convert TOI to minutes
            TOI_min = toi_to_minutes(season_stats.get("timeOnIcePerGame", "20:00"))
            
            # Compute L5/L10 averages (currently using season avg as proxy)
            L5_avg = season_stats.get("shots",0) / max(season_stats.get("games",1),1)
            L10_avg = L5_avg
            
            features = [L5_avg, L10_avg, TOI_min]
            X_player = [features]
            
            data_source = "API"
        else:
            st.warning("Stats not found via NHL API. Using local dataset if available.")
            player_id = None  # fallback to dataset
    else:
        st.warning("Player not found via NHL API. Using local dataset if available.")
        player_id = None  # fallback to dataset

    # ---------------------------
    # Fallback to local dataset
    # ---------------------------
    if player_id is None:
        df["Name_clean"] = df["Name"].str.strip().str.lower()
        player_clean = player_name.strip().lower()
        if player_clean in df["Name_clean"].values:
            row = df[df["Name_clean"] == player_clean].iloc[0]
            full_name = row["Name"]
            team = row.get("Team", "N/A")
            L5_avg = row.get("L5 Avg", 0)
            L10_avg = row.get("L10 Avg", 0)
            TOI_min = row.get("TOI_min", 20)
            X_player = [[L5_avg, L10_avg, TOI_min]]
            data_source = "Dataset"
        else:
            st.error("Player not found in NHL API or local dataset.")
            st.stop()

    # ---------------------------
    # Stat selection
    # ---------------------------
    stat = st.radio("Select Stat", ["Shots on Goal", "Goals"], horizontal=True)
    
    if stat == "Shots on Goal":
        proj = sog_model.predict(X_player)[0]
    else:
        proj = goals_model.predict(X_player)[0]

    # ---------------------------
    # Display Metrics
    # ---------------------------
    c1, c2, c3 = st.columns(3)
    c1.metric("Player", full_name)
    c2.metric("Team", team)
    c3.metric("Projection", round(proj,2))
    st.caption(f"Data source: {data_source}")

    # ---------------------------
    # Hit Rate
    # ---------------------------
    st.markdown("### 🎯 Hit Rate")
    line = st.number_input("Prop Line", value=2.5 if stat=="Shots on Goal" else 0.5, step=0.5)
    basis_value = L5_avg  # default
    if stat == "Shots on Goal":
        rate = round((basis_value / line) * 100, 1) if line>0 else 0
    else:
        rate = round((1 - math.exp(-proj)) * 100, 1)
    st.metric("Estimated Hit Rate", f"{rate}%")
