import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

st.title("ℹ️ About the Project")
st.markdown("Learn about the theoretical concepts and algorithms used in this project.")

st.markdown("---")

st.subheader("Black-Scholes Model")
st.markdown("""
The Black-Scholes model is a mathematical model for the dynamics of a financial market containing derivative investment instruments. 
The formula provides a theoretical estimate of the price of European-style options.
""")
st.latex(r"""
C = S_0 \Phi(d_1) - K e^{-rT} \Phi(d_2)
""")
st.latex(r"""
d_1 = \frac{\ln(S_0 / K) + (r + \frac{\sigma^2}{2})T}{\sigma \sqrt{T}}
""")
st.latex(r"""
d_2 = d_1 - \sigma \sqrt{T}
""")
st.markdown(r"Where: $C$ = Call Option Price, $S_0$ = Current Stock Price, $K$ = Strike Price, $r$ = Risk-Free Interest Rate, $T$ = Time to Maturity, $\sigma$ = Volatility, and $\Phi$ = Cumulative Standard Normal Distribution.")

st.markdown("---")

st.subheader("Geometric Brownian Motion (GBM)")
st.markdown("""
GBM is a continuous-time stochastic process in which the logarithm of the randomly varying quantity follows a Brownian motion with drift. 
It is commonly used to model stock prices in the Black-Scholes model.
""")
st.latex(r"""
dS_t = \mu S_t dt + \sigma S_t dW_t
""")
st.markdown(r"Where: $S_t$ = Stock Price at time $t$, $\mu$ = Drift (expected return), $\sigma$ = Volatility, $W_t$ = Wiener process (Standard Brownian Motion).")

st.markdown("---")

st.subheader("Delta Hedging")
st.markdown(r"""
Delta hedging is an options strategy that aims to reduce, or hedge, the risk associated with price movements in the underlying asset, by offsetting long and short positions.
For a call option, the Delta ($\Delta$) is given by:
""")
st.latex(r"""
\Delta = \Phi(d_1)
""")
st.markdown(r"A delta-neutral portfolio requires holding $-\Delta$ shares of the underlying stock for every option sold.")

st.markdown("---")

st.subheader("Deep Deterministic Policy Gradient (DDPG)")
st.markdown("""
DDPG is an algorithm which concurrently learns a Q-function and a policy. It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.
""")
st.latex(r"""
J(\theta^\mu) = \mathbb{E}_{s \sim \rho^\beta} \left[ Q(s, \mu(s|\theta^\mu) | \theta^Q) \right]
""")
st.markdown(r"Where: $\mu$ is the actor (policy) network with weights $\theta^\mu$, and $Q$ is the critic network with weights $\theta^Q$.")

st.markdown("---")

st.subheader("Environment & Reward Function")
st.markdown("""
The RL environment is framed as a Markov Decision Process (MDP) where:
- **State ($S_t$):** Contains current stock price, time to maturity, option price, and current holdings.
- **Action ($a_t$):** The number of shares to hold (or target delta) for the next time step.
- **Reward ($R_t$):** Formulated to maximize the portfolio value while minimizing transaction costs and hedging errors.
""")
st.latex(r"""
R_t = \Delta V_t - c |a_t - a_{t-1}| - \lambda | V_t - V_{target} |
""")
st.markdown(r"Where: $\Delta V_t$ is the change in portfolio value, $c$ is the transaction cost coefficient, and $\lambda$ is a penalty for deviation from delta neutrality.")
