import streamlit as st
import pandas as pd
import numpy as np
import lime
import lime.lime_tabular
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

st.set_page_config(page_title="LIME Analysis", page_icon="🍋", layout="wide")

st.title("🍋 LIME Analysis")
st.markdown("Local Interpretable Model-agnostic Explanations for individual predictions.")

# Check if model and data are loaded
if 'model' not in st.session_state or st.session_state.model is None:
    st.warning("⚠️ Please upload a model first using the Model Upload page.")
    st.stop()

if 'X' not in st.session_state or st.session_state.X is None:
    st.warning("⚠️ Please analyze your model first using the Model Upload page.")
    st.stop()

# Get data from session state
model = st.session_state.model
X = st.session_state.X
y = st.session_state.y
feature_names = st.session_state.feature_names
X_test = st.session_state.get('X_test', X)
is_classification = st.session_state.get('is_classification', True)

# LIME Configuration
st.sidebar.header("LIME Configuration")

# Select mode
mode = st.sidebar.selectbox(
    "Analysis Mode:",
    ["Single Instance", "Batch Analysis", "Custom Instance"]
)

# LIME explainer parameters
num_features = st.sidebar.slider(
    "Number of features to show:",
    min_value=3,
    max_value=len(feature_names),
    value=min(10, len(feature_names))
)

num_samples = st.sidebar.slider(
    "Number of samples for explanation:",
    min_value=100,
    max_value=5000,
    value=1000,
    help="More samples provide more accurate explanations but take longer"
)

# Initialize LIME explainer
@st.cache_resource
def get_lime_explainer(_X, _feature_names, _is_classification):
    """Create LIME explainer"""
    if _is_classification:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            _X.values,
            feature_names=_feature_names,
            class_names=['Class 0', 'Class 1'] if len(np.unique(y)) == 2 else [f'Class {i}' for i in range(len(np.unique(y)))],
            mode='classification',
            discretize_continuous=True
        )
    else:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            _X.values,
            feature_names=_feature_names,
            mode='regression',
            discretize_continuous=True
        )
    return explainer

# Create LIME explainer
with st.spinner("Initializing LIME explainer..."):
    lime_explainer = get_lime_explainer(X, feature_names, is_classification)

if mode == "Single Instance":
    st.header("🎯 Single Instance Explanation")
    
    # Select instance
    if len(X_test) > 0:
        instance_idx = st.selectbox(
            "Select instance to explain:",
            range(len(X_test)),
            format_func=lambda x: f"Instance {x+1}"
        )
        instance = X_test.iloc[instance_idx].values
        instance_df = X_test.iloc[instance_idx:instance_idx+1]
    else:
        instance_idx = st.selectbox(
            "Select instance to explain:",
            range(len(X)),
            format_func=lambda x: f"Instance {x+1}"
        )
        instance = X.iloc[instance_idx].values
        instance_df = X.iloc[instance_idx:instance_idx+1]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📋 Instance Details")
        
        # Display instance values
        instance_info = pd.DataFrame({
            'Feature': feature_names,
            'Value': instance
        })
        st.dataframe(instance_info, use_container_width=True)
        
        # Model prediction
        prediction = model.predict(instance_df)[0]
        
        if is_classification:
            st.metric("Predicted Class", int(prediction))
            
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(instance_df)[0]
                st.subheader("🎲 Class Probabilities")
                
                prob_df = pd.DataFrame({
                    'Class': [f"Class {i}" for i in range(len(probabilities))],
                    'Probability': probabilities
                })
                
                fig = px.bar(
                    prob_df,
                    x='Class',
                    y='Probability',
                    title="Prediction Probabilities"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.metric("Predicted Value", f"{prediction:.3f}")
    
    with col2:
        st.subheader("🔍 LIME Explanation")
        
        if st.button("Generate Explanation", type="primary"):
            with st.spinner("Generating LIME explanation..."):
                # Generate explanation
                if is_classification:
                    explanation = lime_explainer.explain_instance(
                        instance,
                        model.predict_proba,
                        num_features=num_features,
                        num_samples=num_samples
                    )
                else:
                    explanation = lime_explainer.explain_instance(
                        instance,
                        model.predict,
                        num_features=num_features,
                        num_samples=num_samples
                    )
                
                # Extract explanation data
                exp_data = explanation.as_list()
                
                # Create visualization
                features = [item[0] for item in exp_data]
                importances = [item[1] for item in exp_data]
                
                # Color coding for positive/negative importance
                colors = ['green' if imp > 0 else 'red' for imp in importances]
                
                fig = go.Figure(data=[
                    go.Bar(
                        y=features,
                        x=importances,
                        orientation='h',
                        marker_color=colors,
                        text=[f"{imp:.3f}" for imp in importances],
                        textposition='auto'
                    )
                ])
                
                fig.update_layout(
                    title="LIME Feature Importance",
                    xaxis_title="Importance",
                    yaxis_title="Features",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Explanation table
                st.subheader("📊 Detailed Explanation")
                
                exp_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': importances,
                    'Effect': ['Increases' if imp > 0 else 'Decreases' for imp in importances]
                })
                
                st.dataframe(exp_df.round(4), use_container_width=True)
                
                # Model fidelity
                st.subheader("🎯 Explanation Quality")
                
                score = explanation.score
                local_pred = explanation.local_pred
                
                col3, col4 = st.columns(2)
                with col3:
                    st.metric("R² Score", f"{score:.3f}")
                with col4:
                    st.metric("Local Prediction", f"{local_pred[0]:.3f}")

elif mode == "Batch Analysis":
    st.header("📊 Batch Analysis")
    st.markdown("Analyze multiple instances to understand pattern in explanations.")
    
    # Select batch size
    batch_size = st.slider(
        "Number of instances to analyze:",
        min_value=5,
        max_value=min(50, len(X_test)),
        value=min(10, len(X_test))
    )
    
    if st.button("Analyze Batch", type="primary"):
        batch_explanations = []
        batch_instances = X_test.iloc[:batch_size] if len(X_test) > 0 else X.iloc[:batch_size]
        
        progress_bar = st.progress(0)
        
        for i, (idx, instance) in enumerate(batch_instances.iterrows()):
            with st.spinner(f"Analyzing instance {i+1}/{batch_size}..."):
                # Generate explanation
                if is_classification:
                    explanation = lime_explainer.explain_instance(
                        instance.values,
                        model.predict_proba,
                        num_features=num_features,
                        num_samples=num_samples
                    )
                else:
                    explanation = lime_explainer.explain_instance(
                        instance.values,
                        model.predict,
                        num_features=num_features,
                        num_samples=num_samples
                    )
                
                batch_explanations.append(explanation.as_list())
                progress_bar.progress((i + 1) / batch_size)
        
        # Aggregate results
        st.subheader("📈 Aggregated Feature Importance")
        
        # Collect all feature importances
        all_features = {}
        for explanation in batch_explanations:
            for feature, importance in explanation:
                if feature not in all_features:
                    all_features[feature] = []
                all_features[feature].append(importance)
        
        # Calculate statistics
        feature_stats = []
        for feature, importances in all_features.items():
            feature_stats.append({
                'Feature': feature,
                'Mean_Importance': np.mean(importances),
                'Std_Importance': np.std(importances),
                'Frequency': len(importances) / batch_size
            })
        
        stats_df = pd.DataFrame(feature_stats).sort_values('Mean_Importance', key=abs, ascending=False)
        
        # Visualization
        fig = px.bar(
            stats_df,
            x='Mean_Importance',
            y='Feature',
            orientation='h',
            error_x='Std_Importance',
            title="Average Feature Importance Across Batch"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics table
        st.dataframe(stats_df.round(4), use_container_width=True)
        
        # Feature frequency analysis
        st.subheader("📊 Feature Selection Frequency")
        
        freq_fig = px.bar(
            stats_df,
            x='Feature',
            y='Frequency',
            title="How Often Each Feature Appears in Explanations"
        )
        freq_fig.update_xaxes(tickangle=45)
        st.plotly_chart(freq_fig, use_container_width=True)

else:  # Custom Instance
    st.header("⚙️ Custom Instance Analysis")
    st.markdown("Create your own instance and see how the model would explain it.")
    
    # Input fields for custom instance
    st.subheader("🔧 Configure Instance")
    
    custom_values = {}
    
    # Create input fields for each feature
    cols = st.columns(min(3, len(feature_names)))
    
    for i, feature in enumerate(feature_names):
        col_idx = i % len(cols)
        
        with cols[col_idx]:
            # Get feature statistics
            feature_min = float(X[feature].min())
            feature_max = float(X[feature].max())
            feature_mean = float(X[feature].mean())
            
            custom_values[feature] = st.slider(
                f"{feature}",
                min_value=feature_min,
                max_value=feature_max,
                value=feature_mean,
                help=f"Range: {feature_min:.3f} - {feature_max:.3f}"
            )
    
    # Create custom instance
    custom_instance = np.array([custom_values[feature] for feature in feature_names])
    custom_df = pd.DataFrame([custom_values])
    
    # Display custom instance
    st.subheader("📋 Custom Instance")
    st.dataframe(custom_df, use_container_width=True)
    
    # Prediction for custom instance
    custom_prediction = model.predict(custom_df)[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        if is_classification:
            st.metric("Predicted Class", int(custom_prediction))
            
            if hasattr(model, 'predict_proba'):
                custom_probabilities = model.predict_proba(custom_df)[0]
                
                prob_df = pd.DataFrame({
                    'Class': [f"Class {i}" for i in range(len(custom_probabilities))],
                    'Probability': custom_probabilities
                })
                
                fig = px.bar(
                    prob_df,
                    x='Class',
                    y='Probability',
                    title="Custom Instance Probabilities"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.metric("Predicted Value", f"{custom_prediction:.3f}")
    
    with col2:
        if st.button("Explain Custom Instance", type="primary"):
            with st.spinner("Generating explanation for custom instance..."):
                # Generate explanation
                if is_classification:
                    custom_explanation = lime_explainer.explain_instance(
                        custom_instance,
                        model.predict_proba,
                        num_features=num_features,
                        num_samples=num_samples
                    )
                else:
                    custom_explanation = lime_explainer.explain_instance(
                        custom_instance,
                        model.predict,
                        num_features=num_features,
                        num_samples=num_samples
                    )
                
                # Extract and display explanation
                exp_data = custom_explanation.as_list()
                
                features = [item[0] for item in exp_data]
                importances = [item[1] for item in exp_data]
                colors = ['green' if imp > 0 else 'red' for imp in importances]
                
                fig = go.Figure(data=[
                    go.Bar(
                        y=features,
                        x=importances,
                        orientation='h',
                        marker_color=colors,
                        text=[f"{imp:.3f}" for imp in importances],
                        textposition='auto'
                    )
                ])
                
                fig.update_layout(
                    title="Custom Instance LIME Explanation",
                    xaxis_title="Importance",
                    yaxis_title="Features"
                )
                
                st.plotly_chart(fig, use_container_width=True)

# Export functionality
st.sidebar.markdown("---")
if st.sidebar.button("📥 Export LIME Analysis"):
    # Create sample explanation for export
    sample_instance = X.iloc[0].values
    
    if is_classification:
        sample_explanation = lime_explainer.explain_instance(
            sample_instance,
            model.predict_proba,
            num_features=num_features,
            num_samples=num_samples
        )
    else:
        sample_explanation = lime_explainer.explain_instance(
            sample_instance,
            model.predict,
            num_features=num_features,
            num_samples=num_samples
        )
    
    exp_data = sample_explanation.as_list()
    
    export_df = pd.DataFrame({
        'Feature': [item[0] for item in exp_data],
        'Importance': [item[1] for item in exp_data]
    })
    
    st.sidebar.download_button(
        label="Download LIME Results",
        data=export_df.to_csv(index=False),
        file_name="lime_analysis_results.csv",
        mime="text/csv"
    )

# Information panel
with st.expander("ℹ️ About LIME Analysis"):
    st.markdown("""
    ### What is LIME?
    LIME (Local Interpretable Model-agnostic Explanations) is a technique that explains individual predictions
    by learning an interpretable model locally around the prediction.
    
    ### How LIME Works:
    1. **Perturbation**: Creates variations of the input by changing feature values
    2. **Prediction**: Gets model predictions for these variations
    3. **Weighting**: Weights samples by their proximity to the original instance
    4. **Local Model**: Fits a simple, interpretable model to explain the local behavior
    
    ### Analysis Modes:
    - **Single Instance**: Detailed explanation for one specific prediction
    - **Batch Analysis**: Patterns across multiple instances
    - **Custom Instance**: Test your own hypothetical scenarios
    
    ### Interpreting Results:
    - **Positive values**: Features that increase the prediction
    - **Negative values**: Features that decrease the prediction
    - **Magnitude**: Strength of the feature's local influence
    """)
