import streamlit as st
import pandas as pd
import joblib
import math
import plotly.express as px

st.set_page_config(layout="wide")
st.title("🏒 NHL Player Prop Model (Polished Version)")

# ---------------------------
# Load dataset and models
# ---------------------------
df = pd.read_excel("nhl_stats.xlsx")
df["TOI_min"] = df["TOI"].apply(
    lambda x: int(x.split(':')[0]) + int(x.split(':')[1])/60 if isinstance(x, str) and ':' in x else 20.0
)

sog_model = joblib.load("sog_model.pkl")
goals_model = joblib.load("goals_model.pkl")

# ---------------------------
# Helper function
# ---------------------------
def toi_to_minutes(toi):
    if not toi or ":" not in toi:
        return 20.0
    m, s = toi.split(":")
    return int(m) + int(s)/60

# ---------------------------
# Player input (any name in dataset)
# ---------------------------
df["Name_clean"] = df["Name"].str.strip().str.lower()
player_input = st.text_input("Enter Player Name (from dataset)")

if player_input:
    player_clean = player_input.strip().lower()
    if player_clean in df["Name_clean"].values:
        row = df[df["Name_clean"] == player_clean].iloc[0]
        
        # ---------------------------
        # Projection
        # ---------------------------
        features = ["L5 Avg", "L10 Avg", "TOI_min"]
        X_player = row[features].astype(float).values.reshape(1, -1)

        stat = st.radio("Select Stat", ["Shots on Goal", "Goals"], horizontal=True)
        if stat == "Shots on Goal":
            proj = sog_model.predict(X_player)[0]
        else:
            proj = goals_model.predict(X_player)[0]

        # ---------------------------
        # Display Metrics
        # ---------------------------
        c1, c2, c3 = st.columns(3)
        c1.metric("Player", row["Name"])
        c2.metric("Team", row.get("Team", "N/A"))
        c3.metric("Projection", round(proj,2))

        # ---------------------------
        # Hit Rate
        # ---------------------------
        st.markdown("### 🎯 Hit Rate")
        line = st.number_input("Prop Line", value=2.5 if stat=="Shots on Goal" else 0.5, step=0.5)
        basis = st.radio("Hit Rate Based On", ["Season Avg", "L5 Avg", "L10 Avg"])
        basis_value = row.get(basis, 0)
        
        if stat == "Shots on Goal":
            rate = round((basis_value / line) * 100, 1) if line>0 else 0
        else:
            lam = float(proj)
            rate = round((1 - math.exp(-lam)) * 100, 1)

        st.metric("Estimated Hit Rate", f"{rate}%")

        # ---------------------------
        # Projection Table
        # ---------------------------
        st.markdown("### 📊 All Player Projections")
        table_df = df.dropna(subset=features).copy()
        X_pred = table_df[features].astype(float).values

        if stat == "Shots on Goal":
            table_df["Projection"] = sog_model.predict(X_pred)
        else:
            table_df["Projection"] = goals_model.predict(X_pred)

        # Highlight selected player
        def highlight_player(x):
            return ['background-color: yellow' if x['Name'].lower() == player_clean else '' for i in x]

        st.dataframe(
            table_df[
                ["Name", "Team", "Opponent", "Projection", "Season Avg", "L5 Avg", "L10 Avg", "TOI_min"]
            ].sort_values("Projection", ascending=False).style.apply(highlight_player, axis=1)
        )

        # ---------------------------
        # Visualizations
        # ---------------------------
        st.markdown("### 📊 Top 10 Player Projections")
        top10 = table_df.sort_values("Projection", ascending=False).head(10)
        fig = px.bar(top10, x="Name", y="Projection", color="Projection",
                     color_continuous_scale="Viridis", title="Top 10 Projections")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"### 📈 L5 vs L10 Avg for {row['Name']}")
        avg_df = pd.DataFrame({
            "Metric": ["L5 Avg", "L10 Avg"],
            "Value": [row["L5 Avg"], row["L10 Avg"]]
        })
        fig2 = px.line(avg_df, x="Metric", y="Value", markers=True,
                       title=f"L5 vs L10 Avg for {row['Name']}")
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.error("Player not found in dataset. Make sure the name matches exactly.")
