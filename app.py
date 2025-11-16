import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="FIFA Ranking Stability Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Loading dataset & model
@st.cache_data
def load_data():
    return pd.read_csv("fifa_ranking_2022-10-06.csv")

@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

df = load_data()
model = load_model()

# ------------------------------
# Helper Function
def predict(rank, previous_rank, points, previous_points):
    X = np.array([[rank, previous_rank, points, previous_points]])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    return pred, prob


# ------------------------------
# Sidebar Navigation
st.sidebar.title("⚽ Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Team Insights", "Compare Two Teams", "Ranking Trend Simulation"]
)


# ------------------------------
# Page 1: Team insights
if page == "Team Insights":
    st.title("🔍 Team Insights")
    st.write("Select a team to view ranking data and prediction.")

    team_list = df["team"].sort_values().unique()
    team = st.selectbox("Select a Team", team_list)

    team_data = df[df["team"] == team].iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Current Rank", int(team_data["rank"]))
    with col2:
        st.metric("Previous Rank", int(team_data["previous_rank"]))
    with col3:
        movement = (
            "↑ Improved" if team_data["previous_rank"] > team_data["rank"]
            else "↓ Dropped" if team_data["previous_rank"] < team_data["rank"]
            else "→ Stable"
        )
        st.metric("Movement", movement)

    st.subheader("Prediction")
    pred, prob = predict(
        team_data["rank"],
        team_data["previous_rank"],
        team_data["points"],
        team_data["previous_points"]
    )

    if pred == 0:
        st.success(f"✅ Stable (Probability: {prob:.2f})")
    else:
        st.warning(f"⚠️ Changed (Probability: {prob:.2f})")

    st.info(f"**Association:** {team_data['association']}")
    st.write("---")


# ------------------------------
# Page 2: compare 2 teams
elif page == "Compare Two Teams":
    st.title("⚔️ Compare Two Teams")

    teams = df["team"].sort_values().unique()

    colA, colB = st.columns(2)

    with colA:
        team1 = st.selectbox("Team A", teams)
    with colB:
        team2 = st.selectbox("Team B", teams)

    data1 = df[df["team"] == team1].iloc[0]
    data2 = df[df["team"] == team2].iloc[0]

    st.subheader("📊 Ranking Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(f"{team1} Rank", int(data1["rank"]))
        st.metric("Points", float(data1["points"]))
    with col2:
        st.metric(f"{team2} Rank", int(data2["rank"]))
        st.metric("Points", float(data2["points"]))

    st.subheader("🧠 ML Predictions")
    pred1, prob1 = predict(data1["rank"], data1["previous_rank"], data1["points"], data1["previous_points"])


# ------------------------------------------
# Page 3: Rank trend simulation
elif page == "Ranking Trend Simulation":
    st.title("📈 Ranking Trend Simulation")
    st.write("Simulate how a team's rank changes based on points adjustments.")

    teams = df["team"].sort_values().unique()
    team = st.selectbox("Select a Team", teams)

    team_data = df[df["team"] == team].iloc[0]
    current_rank = int(team_data["rank"])
    current_points = float(team_data["points"])

    st.subheader(f"📌 Current Rank: **{current_rank}**")
    st.subheader(f"📌 Current Points: **{current_points}**")

    # adjusting the team's points
    new_points = st.slider(
        "Adjust Team Points",
        min_value=int(current_points - 100),
        max_value=int(current_points + 200),
        value=int(current_points),
        step=5
    )

    # simulation df
    sim_df = df.copy()
    sim_df.loc[sim_df["team"] == team, "points"] = new_points

    # Re-rank teams
    sim_df = sim_df.sort_values(by="points", ascending=False).reset_index(drop=True)
    sim_df["new_rank"] = sim_df.index + 1

    new_rank = int(sim_df[sim_df["team"] == team]["new_rank"].iloc[0])

    # Displaying results
    st.subheader(f"🏆 New Simulated Rank: **{new_rank}**")
    rank_change = current_rank - new_rank

    if rank_change > 0:
        st.success(f"✅ Team improved by **{rank_change}** positions!")
    elif rank_change < 0:
        st.error(f"❌ Team dropped by **{-rank_change}** positions!")
    else:
        st.info("⚪ No change in ranking.")

    # display Chart
    st.write("### Rank Comparison")
    st.bar_chart(
        pd.DataFrame({
            "Rank": [current_rank, new_rank]
        }, index=["Current Rank", "Simulated Rank"])
    )

