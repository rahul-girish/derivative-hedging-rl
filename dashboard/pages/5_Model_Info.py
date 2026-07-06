import streamlit as st
import pandas as pd

st.set_page_config(page_title="Model Information", page_icon="🤖", layout="wide")

st.title("🤖 Model Information")
st.markdown("Overview of the Deep Deterministic Policy Gradient (DDPG) model hyperparameters used for training.")

# Define the hyperparameters specified in the project requirements
hyperparameters = {
    "Parameter": [
        "Algorithm",
        "Policy",
        "Training timesteps",
        "Replay Buffer Size",
        "Batch Size",
        "Learning Rate",
        "Discount Factor (Gamma)",
        "Soft Update Coefficient (Tau)",
        "Observation Space",
        "Action Space",
        "Reward Function"
    ],
    "Value": [
        "DDPG (Deep Deterministic Policy Gradient)",
        "MlpPolicy",
        "500,000",
        "100,000",
        "256",
        "3e-4",
        "0.99",
        "0.005",
        "Box(low=-inf, high=inf, shape=(4,)) - [Stock Price, Time to Maturity, Current Delta, Option Price]",
        "Box(low=-1.0, high=1.0, shape=(1,)) - [Target Delta / Number of Options]",
        "Change in Portfolio Value - Transaction Costs - Hedging Penalty"
    ]
}

df_hyper = pd.DataFrame(hyperparameters)

st.subheader("Hyperparameters Table")
# Present them in a professional table
st.dataframe(df_hyper, width="stretch", hide_index=True)

st.markdown("---")
st.markdown("""
### Architecture Overview
The model uses an **MlpPolicy**, which consists of multi-layer perceptrons (feed-forward neural networks) for both the Actor (policy) and the Critic (Q-value function). 
- **Actor:** Maps the current observation to an action (the target hedge ratio or number of stocks to hold).
- **Critic:** Evaluates the chosen action given the current observation by estimating the expected future reward.

DDPG is an off-policy, actor-critic algorithm that uses a replay buffer and target networks (updated softly via parameter Tau) to stabilize training in environments with continuous action spaces.
""")
