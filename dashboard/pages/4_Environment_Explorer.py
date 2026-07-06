import streamlit as st

st.set_page_config(page_title="Environment Explorer", page_icon="🎛️", layout="wide")

st.title("🎛️ Environment Explorer")
st.markdown("Interactively explore and configure the Black-Scholes environment parameters.")
st.info("💡 Note: These settings are for exploration and visualization only. Retraining is not performed here.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Asset & Option Parameters")
    
    initial_stock = st.slider(
        "Initial Stock Price (S0)",
        min_value=50.0, max_value=150.0, value=100.0, step=1.0,
        help="The initial price of the underlying asset."
    )
    
    strike_price = st.slider(
        "Strike Price (K)",
        min_value=50.0, max_value=150.0, value=100.0, step=1.0,
        help="The strike price of the European option."
    )
    
    maturity = st.slider(
        "Maturity (T in years)",
        min_value=0.1, max_value=3.0, value=1.0, step=0.1,
        help="Time to expiration of the option."
    )

with col2:
    st.subheader("Market Parameters")
    
    volatility = st.slider(
        "Volatility (σ)",
        min_value=0.01, max_value=1.00, value=0.20, step=0.01,
        help="The annualized volatility of the underlying asset."
    )
    
    interest_rate = st.slider(
        "Risk-Free Interest Rate (r)",
        min_value=0.0, max_value=0.20, value=0.05, step=0.01,
        help="The continuous risk-free interest rate."
    )

st.markdown("---")
st.subheader("Current Environment Configuration")

config_data = {
    "Parameter": [
        "Initial Stock Price ($S_0$)",
        "Strike Price ($K$)",
        "Maturity ($T$)",
        r"Volatility ($\sigma$)",
        "Interest Rate ($r$)"
    ],
    "Selected Value": [
        f"${initial_stock:.2f}",
        f"${strike_price:.2f}",
        f"{maturity:.2f} years",
        f"{volatility:.2%}",
        f"{interest_rate:.2%}"
    ]
}

st.table(config_data)
