import streamlit as st
import requests
import pandas as pd
import joblib
import math
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🏒 NHL Player Prop Model (Live API)")

# ---------------------------
# Load models
# ---------------------------
sog_model = joblib.load("sog_model.pkl")
goals_model = joblib.load("goals_model.pkl")

# ---------------------------
# Helper Functions
# ---------------------------

def get_player_id(player_name):
    """Search NHL API for player ID by full name"""
    url = f"https://statsapi.web.nhl.com/api/v1/people?fullName={player_name}"
    resp = requests.get(url).json()
    people = resp.get("people", [])
    if people:
        return people[0]["id"], people[0]["fullName"], people[0]["currentTeam"]["name"]
    return None, None, None

def get_season_stats(player_id, season="20242025"):
    """Fetch season stats for a player"""
    url = f"https://statsapi.web.nhl.com/api/v1/people/{player_id}/stats?stats=statsSingleSeason&season={season}"
    resp = requests.get(url).json()
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
    return None

def toi_to_minutes(toi):
    if not toi or ":" not in toi:
        return 20.0  # default
    m, s = toi.split(":")
    return int(m) + int(s)/60

# ---------------------------
# User Input
# ---------------------------
player_name = st.text_input("Enter NHL Player Name")

if player_name:
    player_id, full_name, team = get_player_id(player_name)
    if player_id:
        season_stats = get_season_stats(player_id)
        if season_stats:
            # Convert TOI to minutes
            TOI_min = toi_to_minutes(season_stats.get("timeOnIcePerGame", "20:00"))
            
            # Compute features for model
            # For simplicity, we assume L5 Avg = shots/games (or goals/games) as proxy
            # In a real implementation, you would fetch game logs and compute last 5/10 averages
            L5_avg = season_stats.get("shots",0) / max(season_stats.get("games",1),1)
            L10_avg = L5_avg  # fallback: use same value if no game log
            
            features = [L5_avg, L10_avg, TOI_min]
            X_player = [features]
            
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
            
            # Hit Rate
            st.markdown("### 🎯 Hit Rate")
            line = st.number_input("Prop Line", value=2.5 if stat=="Shots on Goal" else 0.5, step=0.5)
            basis_value = L5_avg  # for simplicity, you can also allow L10_avg or user selection
            if stat == "Shots on Goal":
                rate = round((basis_value / line) * 100, 1) if line>0 else 0
            else:
                rate = round((1 - math.exp(-proj)) * 100, 1)
            st.metric("Estimated Hit Rate", f"{rate}%")
            
        else:
            st.error("Season stats not found for this player.")
    else:
        st.error("Player not found in NHL API.")
