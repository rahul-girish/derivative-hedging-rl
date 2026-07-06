import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Agent Comparison", page_icon="📊", layout="wide")

st.title("📊 Agent Comparison")
st.markdown("Compare the performance of the agents using different metrics.")

RESULTS_PATH = "outputs/results/comparison.csv"

@st.cache_data
def load_data():
    if os.path.exists(RESULTS_PATH):
        return pd.read_csv(RESULTS_PATH)
    return None

df = load_data()

if df is not None:
    st.subheader("Metrics Dataframe")
    st.dataframe(df, width="stretch")
    
    st.markdown("---")
    st.subheader("Interactive Visualizations")
    
    # Exclude non-numeric columns like 'Agent' for the selectbox
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if numeric_columns:
        selected_metric = st.selectbox("Select a metric to visualize:", numeric_columns)
        
        # Determine if higher or lower is better
        # For Reward and Portfolio, higher is better. For others (Error, Cost, RMSE), lower is better.
        higher_is_better = selected_metric in ["Reward", "Portfolio"]
        
        if higher_is_better:
            best_idx = df[selected_metric].idxmax()
        else:
            best_idx = df[selected_metric].idxmin()
            
        best_agent = df.loc[best_idx, 'Agent']
        st.success(f"🏆 Best performing agent for **{selected_metric}**: **{best_agent}**")
        
        # Create bar chart
        fig = px.bar(
            df, 
            x='Agent', 
            y=selected_metric, 
            color='Agent',
            title=f"{selected_metric} by Agent",
            text_auto='.2f',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        fig.update_layout(
            xaxis_title="Agent Strategy",
            yaxis_title=selected_metric,
            showlegend=False
        )
        
        st.plotly_chart(fig, width="stretch")
        
    else:
        st.warning("No numeric columns found to plot.")
        
else:
    st.error(f"Results file not found at `{RESULTS_PATH}`. Please run the evaluation first.")
