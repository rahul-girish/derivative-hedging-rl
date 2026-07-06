import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Performance Summary", page_icon="🏆", layout="wide")

st.title("🏆 Performance Summary")
st.markdown("High-level summary of agent performance and a comparative radar chart.")

RESULTS_PATH = "outputs/results/comparison.csv"

@st.cache_data
def load_data():
    if os.path.exists(RESULTS_PATH):
        return pd.read_csv(RESULTS_PATH)
    return None

df = load_data()

if df is not None:
    st.subheader("Best Overall Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_reward = df['Reward'].max()
        best_reward_agent = df.loc[df['Reward'].idxmax(), 'Agent']
        st.metric(label="Best Reward", value=f"{best_reward:.2f}", delta=best_reward_agent, delta_color="off")
        
    with col2:
        lowest_rmse = df['RMSE'].min()
        lowest_rmse_agent = df.loc[df['RMSE'].idxmin(), 'Agent']
        st.metric(label="Lowest RMSE", value=f"{lowest_rmse:.2f}", delta=lowest_rmse_agent, delta_color="inverse")
        
    with col3:
        lowest_error = df['Mean Hedging Error'].min()
        lowest_error_agent = df.loc[df['Mean Hedging Error'].idxmin(), 'Agent']
        st.metric(label="Lowest Hedging Error", value=f"{lowest_error:.2f}", delta=lowest_error_agent, delta_color="inverse")
        
    with col4:
        lowest_cost = df['Transaction Cost'].min()
        lowest_cost_agent = df.loc[df['Transaction Cost'].idxmin(), 'Agent']
        st.metric(label="Lowest Transaction Cost", value=f"{lowest_cost:.2f}", delta=lowest_cost_agent, delta_color="inverse")

    st.markdown("---")
    st.subheader("Agent Performance Radar Chart")
    
    # We will normalize the metrics for the radar chart so they can be plotted on the same scale (0 to 1).
    # For Reward and Portfolio, higher is better (Min-Max normalization)
    # For Errors and Costs, lower is better (Max-Min normalization, so 1 is best)
    
    categories = ['Reward', 'Portfolio', 'RMSE', 'Transaction Cost', 'Mean Hedging Error']
    
    # Create a normalized dataframe for the radar chart
    df_norm = df.copy()
    
    for col in ['Reward', 'Portfolio']:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        if max_val != min_val:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 1.0
            
    for col in ['RMSE', 'Transaction Cost', 'Mean Hedging Error']:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        if max_val != min_val:
            # Invert so higher value on radar means better performance (lower error/cost)
            df_norm[col] = (max_val - df_norm[col]) / (max_val - min_val)
        else:
            df_norm[col] = 1.0

    fig = go.Figure()

    for i, row in df_norm.iterrows():
        agent = row['Agent']
        values = [row[cat] for cat in categories]
        # Close the loop
        values.append(values[0])
        closed_categories = categories + [categories[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=closed_categories,
            fill='toself',
            name=agent
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Normalized Performance (Outer Edge = Better)"
    )

    st.plotly_chart(fig, width="stretch")

else:
    st.error(f"Results file not found at `{RESULTS_PATH}`. Please run the evaluation first.")
