import streamlit as st

st.set_page_config(
    page_title="Derivative Hedging DRL Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Streamlit native multipage handles the navigation through the `pages/` folder.
# This app.py file serves as the entry point. We can simply redirect or display a welcome message.
# Since we have `1_Home.py`, we can also just put some common configuration here.

st.title("Welcome to the Derivative Hedging DRL Dashboard")
st.markdown("""
Please use the **sidebar navigation** to explore different aspects of the project:

- 🏠 **Home**: Overview and Key Performance Indicators
- 📊 **Agent Comparison**: Compare the performance of Random, Delta, and DDPG agents
- 🖼️ **Figures**: View the generated plots from training and evaluation
- 🎛️ **Environment Explorer**: Interactively explore the Black-Scholes environment parameters
- 🤖 **Model Info**: View hyperparameters of the trained DDPG agent
- 🏆 **Performance Summary**: See the summary and radar chart of agent performances
- ℹ️ **About**: Learn about the theoretical concepts and math behind the project
""")
