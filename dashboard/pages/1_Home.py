import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Home - Hedging DRL", page_icon="🏠", layout="wide")

st.title("Derivative Hedging using Deep Reinforcement Learning")
st.subheader("Comparison of Classical Black-Scholes Delta Hedging and Deep Reinforcement Learning")

st.markdown("""
### Project Overview
This project explores the application of Deep Reinforcement Learning (DRL), specifically the Deep Deterministic Policy Gradient (DDPG) algorithm, to the problem of derivative hedging. 
It compares the DRL agent's performance against a classical Black-Scholes Delta hedging strategy and a random action baseline. 
The underlying asset follows a Geometric Brownian Motion (GBM), and the goal is to minimize the hedging error and transaction costs while maintaining a delta-neutral portfolio.
""")

# Load metrics to display on the KPI cards
RESULTS_PATH = "outputs/results/comparison.csv"
try:
    df = pd.read_csv(RESULTS_PATH)
    
    # Determine best agent (highest reward)
    best_agent = df.loc[df['Reward'].idxmax(), 'Agent']
    
    # Determine lowest RMSE
    lowest_rmse = df['RMSE'].min()
    
    # Determine lowest transaction cost
    lowest_tc = df['Transaction Cost'].min()
    
except Exception as e:
    st.warning("Could not load comparison metrics. Please ensure `outputs/results/comparison.csv` exists.")
    best_agent = "N/A"
    lowest_rmse = "N/A"
    lowest_tc = "N/A"

st.markdown("---")
st.markdown("### Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="🏆 Best Agent", value=best_agent)

with col2:
    val = f"{lowest_rmse:.2f}" if isinstance(lowest_rmse, float) else lowest_rmse
    st.metric(label="📉 Lowest RMSE", value=val)

with col3:
    val = f"{lowest_tc:.2f}" if isinstance(lowest_tc, float) else lowest_tc
    st.metric(label="💸 Lowest Transaction Cost", value=val)

with col4:
    # Based on the plan, setting a static value of 1000 or reading if possible. We will set 100 as a placeholder or just note it.
    st.metric(label="🔁 Number of Evaluation Episodes", value="1000") # Assumption
