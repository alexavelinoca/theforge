import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Model Health Monitor")

# --- Main Title and Description ---
st.title("Model Health Monitor: Continuous Evaluation & Drift Debugging")
st.write("Welcome to the dashboard for observing the impact of data drift on LLM performance.")

# --- Sidebar (Optional for future controls) ---
st.sidebar.header("Controls & Navigation")
# You can add filters or selectors here later, e.g., st.sidebar.selectbox(...)

# --- Simulate Performance & Drift Data (This will come from Thiago's backend later) ---
sim_current_quality = 85.5 # Simulated current LLM quality
sim_baseline_quality = 90.0 # Simulated baseline LLM quality
sim_change_perc = ((sim_current_quality - sim_baseline_quality) / sim_baseline_quality) * 100

# Simulated drift metrics for individual features
sim_drift_summary = pd.DataFrame({
    'feature': ['prompt_length', 'prompt_topic', 'response_coherence_score'],
    'type': ['numerical', 'categorical', 'numerical'],
    'psi': [0.18, 0.07, 0.25], # PSI for numerical
    'p_value': [0.001, 0.005, 0.01], # p-value for KS/Chi2
    'drift_alert': ['CRITICAL', 'WARNING', 'CRITICAL'] # Based on thresholds
})


st.header("1. Model Health Overview")
st.write("Quick glance at current LLM quality and key alerts.")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Current LLM Quality (Simulated %)", value=f"{sim_current_quality:.1f}%")
with col2:
    st.metric(label="Baseline LLM Quality (Simulated %)", value=f"{sim_baseline_quality:.1f}%")
with col3:
    st.metric(label="Change vs. Baseline", value=f"{sim_change_perc:.1f}%", delta=f"{sim_change_perc:.1f}%", delta_color="inverse")
    # 'delta_color="inverse"' makes negative deltas red and positive deltas green

# Logic to display overall performance alert
if sim_change_perc < -5: # If quality drops more than 5%
    st.error("🚨 CRITICAL Alert! LLM quality has dropped significantly. Immediate attention required.")
elif sim_change_perc < 0: # If quality has slightly decreased
    st.warning("⚠️ Warning: LLM quality has slightly decreased. Possible data or concept drift.")
else: # If quality is stable or improved
    st.success("✅ Model Operating Healthy. Quality stable or improved.")

st.header("2. LLM Performance Trend (Output)")
st.write("Observe how the simulated model quality has evolved across different data periods.")

# Prepare data for the comparative bar chart
performance_data_for_plot = {
    'Period': ['Baseline', 'Current'],
    'Simulated Quality (%)': [sim_baseline_quality, sim_current_quality]
}
df_performance_plot = pd.DataFrame(performance_data_for_plot)

# Create the bar chart with Plotly Express
fig_performance = px.bar(df_performance_plot, x='Period', y='Simulated Quality (%)',
                         title='Simulated LLM Quality: Baseline vs. Current',
                         color='Period', # Color bars by period
                         color_discrete_map={'Baseline': 'lightgray', 'Current': 'teal'}) # Custom colors
fig_performance.update_layout(yaxis_range=[0,100]) # Ensure Y-axis is from 0 to 100%
st.plotly_chart(fig_performance, use_container_width=True) # Display the chart in Streamlit

st.header("3. Input Data Drift Analysis")
st.write("Identify which of your input features have changed most significantly compared to the baseline, potentially impacting the LLM.")

st.subheader("Drift Summary by Feature")
# Display the simulated drift summary table
st.dataframe(sim_drift_summary.style.applymap(
    lambda x: 'background-color: #ffe0b2' if 'WARNING' in str(x) else ('background-color: #ffcdd2' if 'CRITICAL' in str(x) else ''),
    subset=['drift_alert'] # Apply style only to the 'drift_alert' column
))


st.subheader("Detailed Distribution Drift Visualization")
# Dropdown to select the feature to visualize
selected_feature = st.selectbox(
    "Select an input feature to view its distribution change:",
    sim_drift_summary['feature'].tolist()
)

# Simulate raw data for histograms/bar charts based on selected feature
# In a real scenario, these would come from Thiago's analysis_results['hist_data']
if selected_feature == 'prompt_length':
    base_dist_data = np.random.normal(loc=100, scale=20, size=500) # Baseline: average length 100
    current_dist_data = np.random.normal(loc=150, scale=25, size=500) # Current: shifted to average length 150
    df_hist = pd.DataFrame({
        'Prompt Length': np.concatenate([base_dist_data, current_dist_data]),
        'Dataset': ['Baseline'] * len(base_dist_data) + ['Current'] * len(current_dist_data)
    })
    fig_hist = px.histogram(df_hist, x='Prompt Length', color='Dataset',
                            title=f"Distribution of '{selected_feature}': Baseline vs. Current",
                            barmode='overlay', opacity=0.7, histnorm='percent')
    st.plotly_chart(fig_hist, use_container_width=True)

elif selected_feature == 'prompt_topic':
    base_topics = np.random.choice(['finance', 'tech', 'health'], size=500, p=[0.4, 0.4, 0.2])
    current_topics = np.random.choice(['finance', 'tech', 'quantum_ai', 'health'], size=500, p=[0.3, 0.3, 0.2, 0.2]) # New topic introduced

    base_counts = pd.Series(base_topics).value_counts().reset_index()
    base_counts.columns = ['Topic', 'Count']
    base_counts['Dataset'] = 'Baseline'

    current_counts = pd.Series(current_topics).value_counts().reset_index()
    current_counts.columns = ['Topic', 'Count']
    current_counts['Dataset'] = 'Current'

    df_bar = pd.concat([base_counts, current_counts])

    fig_bar = px.bar(df_bar, x='Topic', y='Count', color='Dataset',
                     title=f"Frequency of '{selected_feature}': Baseline vs. Current",
                     barmode='group')
    st.plotly_chart(fig_bar, use_container_width=True)

elif selected_feature == 'response_coherence_score':
    # Simulate a drop in coherence score
    base_scores = np.random.uniform(0.7, 0.9, size=500)
    current_scores = np.random.uniform(0.4, 0.7, size=500) # Scores are lower
    df_hist = pd.DataFrame({
        'Coherence Score': np.concatenate([base_scores, current_scores]),
        'Dataset': ['Baseline'] * len(base_scores) + ['Current'] * len(current_scores)
    })
    fig_hist = px.histogram(df_hist, x='Coherence Score', color='Dataset',
                            title=f"Distribution of '{selected_feature}': Baseline vs. Current",
                            barmode='overlay', opacity=0.7, histnorm='percent')
    st.plotly_chart(fig_hist, use_container_width=True)

st.header("4. Correlation: Input Drift vs. Output Performance")
st.write("This section helps you debug. Observe if a drop in LLM quality coincides with a significant drift in key inputs.")

# Simulate time-series data for the correlation plot
sim_time_points = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5']
sim_perf_trend = [sim_baseline_quality, sim_baseline_quality - 2, sim_baseline_quality - 5, # Performance starts dropping
                  sim_current_quality + 3, sim_current_quality, sim_current_quality - 2] # Continues to fluctuate lower
sim_drift_length_trend = [0.01, 0.03, 0.10, 0.15, 0.12, 0.20] # Drift increases when performance drops
sim_drift_topic_trend = [0.01, 0.02, 0.03, 0.08, 0.06, 0.15] # Drift also increases

df_correlation_sim = pd.DataFrame({
    'Period': sim_time_points,
    'Simulated Quality (%)': sim_perf_trend,
    'Drift (Prompt Length - PSI)': sim_drift_length_trend,
    'Drift (Prompt Topic - p-value)': sim_drift_topic_trend
})

# Create dual-axis plot
fig_corr = go.Figure()

# Add LLM Quality line (primary Y-axis)
fig_corr.add_trace(go.Scatter(x=df_correlation_sim['Period'], y=df_correlation_sim['Simulated Quality (%)'],
                             mode='lines+markers', name='Simulated LLM Quality', yaxis='y1', line=dict(color='blue')))

# Add Drift for Prompt Length line (secondary Y-axis)
fig_corr.add_trace(go.Scatter(x=df_correlation_sim['Period'], y=df_correlation_sim['Drift (Prompt Length - PSI)'],
                             mode='lines+markers', name='Drift (Prompt Length - PSI)', yaxis='y2', line=dict(color='red')))

# Add Drift for Prompt Topic line (secondary Y-axis)
fig_corr.add_trace(go.Scatter(x=df_correlation_sim['Period'], y=df_correlation_sim['Drift (Prompt Topic - p-value)'],
                             mode='lines+markers', name='Drift (Prompt Topic - p-value)', yaxis='y2', line=dict(color='green')))


fig_corr.update_layout(
    title='Correlation: LLM Performance vs. Input Data Drift',
    yaxis=dict(title='Simulated Quality (%)', side='left', showgrid=False, range=[0,100]), # Primary Y-axis for performance
    yaxis2=dict(title='Drift Metric', side='right', overlaying='y', showgrid=True, range=[0, df_correlation_sim[['Drift (Prompt Length - PSI)', 'Drift (Prompt Topic - p-value)']].max().max() * 1.2]), # Secondary Y-axis for drift
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(0,0,0,0.2)'),
    hovermode="x unified" # Shows info for all traces on hover
)
st.plotly_chart(fig_corr, use_container_width=True)

st.info("💡 **How to use this:** If the 'Simulated LLM Quality' line drops while a 'Drift' line for an input feature rises, it's a strong signal that specific data drift is impacting model performance.")