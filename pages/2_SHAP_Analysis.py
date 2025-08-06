import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Try to import SHAP, handle gracefully if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.warning("⚠️ SHAP library is not available in this environment. This page will show alternative explanations.")

st.set_page_config(page_title="SHAP Analysis", page_icon="📊", layout="wide")

st.title("📊 SHAP Analysis")
st.markdown("Understand your model's decisions using Shapley values for transparent AI explanations.")

# Check if SHAP is available
if not SHAP_AVAILABLE:
    st.error("❌ SHAP library is not available in this environment.")
    st.info("""
    ### Alternative Explainability Methods
    
    Since SHAP is not available, you can use:
    - **LIME Analysis**: Available on the LIME Analysis page for local explanations
    - **Model Feature Importance**: Built-in feature importance from tree-based models
    - **Examples**: Pre-built demonstrations on the Examples page
    
    These methods provide similar insights into model behavior and predictions.
    """)
    st.stop()

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

# SHAP Configuration
st.sidebar.header("SHAP Configuration")

# Select analysis type
analysis_type = st.sidebar.selectbox(
    "Analysis Type:",
    ["Global Feature Importance", "Local Explanations", "Partial Dependence", "Interaction Effects"]
)

# Sample size for SHAP (for performance)
max_samples = min(1000, len(X))
sample_size = st.sidebar.slider(
    "Sample Size for Analysis:",
    min_value=50,
    max_value=max_samples,
    value=min(200, max_samples),
    help="Larger samples provide more accurate results but take longer to compute"
)

# Initialize SHAP explainer
@st.cache_data(show_spinner=False)
def get_shap_explainer_and_values(_model, _X_sample):
    """Create SHAP explainer and compute values"""
    try:
        # Try TreeExplainer first (fastest for tree-based models)
        if hasattr(_model, 'estimators_') or 'forest' in str(type(_model)).lower():
            explainer = shap.TreeExplainer(_model)
            shap_values = explainer.shap_values(_X_sample)
        else:
            # Fall back to KernelExplainer for other models
            X_background = shap.sample(_X_sample, min(100, len(_X_sample)))
            explainer = shap.KernelExplainer(_model.predict, X_background)
            shap_values = explainer.shap_values(_X_sample)
            
        return explainer, shap_values
    except Exception as e:
        st.error(f"Error creating SHAP explainer: {str(e)}")
        return None, None

# Sample data for analysis
X_sample = X.sample(n=sample_size, random_state=42) if len(X) > sample_size else X

# Generate SHAP values
with st.spinner("Computing SHAP values... This may take a moment."):
    explainer, shap_values = get_shap_explainer_and_values(model, X_sample)

if explainer is None or shap_values is None:
    st.error("Unable to create SHAP explainer. Please check your model compatibility.")
    st.stop()

# Handle multi-class classification
if is_classification and isinstance(shap_values, list):
    if len(shap_values) == 2:  # Binary classification
        shap_values_display = shap_values[1]  # Use positive class
        class_names = ["Class 0", "Class 1"]
    else:  # Multi-class
        class_idx = st.sidebar.selectbox(
            "Select class for analysis:",
            range(len(shap_values)),
            format_func=lambda x: f"Class {x}"
        )
        shap_values_display = shap_values[class_idx]
        class_names = [f"Class {i}" for i in range(len(shap_values))]
else:
    shap_values_display = shap_values

# Analysis sections based on selection
if analysis_type == "Global Feature Importance":
    st.header("🌍 Global Feature Importance")
    st.markdown("Understanding which features are most important across all predictions.")
    
    # Summary plot
    st.subheader("Feature Importance Summary")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create summary plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values_display, X_sample, feature_names=feature_names, show=False)
        st.pyplot(fig, clear_figure=True)
        plt.close()
    
    with col2:
        # Feature importance bar chart
        mean_abs_shap = np.abs(shap_values_display).mean(axis=0)
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': mean_abs_shap
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Mean |SHAP Value|"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed importance metrics
    st.subheader("📋 Detailed Feature Rankings")
    
    detailed_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean |SHAP|': np.abs(shap_values_display).mean(axis=0),
        'Std SHAP': np.std(shap_values_display, axis=0),
        'Max |SHAP|': np.abs(shap_values_display).max(axis=0),
        'Range': np.abs(shap_values_display).max(axis=0) - np.abs(shap_values_display).min(axis=0)
    }).sort_values('Mean |SHAP|', ascending=False)
    
    st.dataframe(detailed_df.round(4), use_container_width=True)

elif analysis_type == "Local Explanations":
    st.header("🎯 Local Explanations")
    st.markdown("Understand individual predictions and their explanations.")
    
    # Select sample for explanation
    sample_idx = st.selectbox(
        "Select sample for explanation:",
        range(len(X_sample)),
        format_func=lambda x: f"Sample {x+1}"
    )
    
    selected_sample = X_sample.iloc[sample_idx:sample_idx+1]
    selected_shap = shap_values_display[sample_idx]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Sample Information")
        sample_info = selected_sample.T
        sample_info.columns = ['Value']
        st.dataframe(sample_info, use_container_width=True)
        
        # Prediction
        prediction = model.predict(selected_sample)[0]
        if hasattr(model, 'predict_proba') and is_classification:
            probabilities = model.predict_proba(selected_sample)[0]
            st.subheader("🎲 Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Class': [f"Class {i}" for i in range(len(probabilities))],
                'Probability': probabilities
            })
            fig = px.bar(prob_df, x='Class', y='Probability', title="Class Probabilities")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.metric("Prediction", f"{prediction:.3f}")
    
    with col2:
        st.subheader("🔍 SHAP Explanation")
        
        # Waterfall plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if hasattr(shap, 'waterfall_plot'):
            # Use new SHAP API if available
            expected_value = explainer.expected_value
            if isinstance(expected_value, np.ndarray):
                expected_value = expected_value[0] if is_classification else expected_value
            
            shap.waterfall_plot(
                shap.Explanation(
                    values=selected_shap,
                    base_values=expected_value,
                    data=selected_sample.iloc[0],
                    feature_names=feature_names
                ),
                show=False
            )
        else:
            # Fallback to force plot
            shap.force_plot(
                explainer.expected_value[0] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
                selected_shap,
                selected_sample,
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
        
        st.pyplot(fig, clear_figure=True)
        plt.close()
    
    # Feature contribution table
    st.subheader("📊 Feature Contributions")
    
    contrib_df = pd.DataFrame({
        'Feature': feature_names,
        'Value': selected_sample.iloc[0].values,
        'SHAP Value': selected_shap,
        'Contribution': ['Positive' if x > 0 else 'Negative' for x in selected_shap]
    }).sort_values('SHAP Value', key=abs, ascending=False)
    
    st.dataframe(contrib_df.round(4), use_container_width=True)

elif analysis_type == "Partial Dependence":
    st.header("📈 Partial Dependence Analysis")
    st.markdown("See how individual features affect predictions across their range.")
    
    # Select feature for partial dependence
    selected_feature = st.selectbox(
        "Select feature for analysis:",
        feature_names
    )
    
    feature_idx = feature_names.index(selected_feature)
    
    # Generate partial dependence plot
    feature_range = np.linspace(
        X[selected_feature].min(),
        X[selected_feature].max(),
        50
    )
    
    # Create samples for partial dependence
    X_pd = X_sample.copy()
    shap_values_pd = []
    
    with st.spinner(f"Computing partial dependence for {selected_feature}..."):
        for value in feature_range:
            X_temp = X_pd.copy()
            X_temp[selected_feature] = value
            
            # Get SHAP values for this configuration
            if hasattr(model, 'estimators_') or 'forest' in str(type(model)).lower():
                temp_shap = explainer.shap_values(X_temp)
            else:
                temp_shap = explainer.shap_values(X_temp)
            
            if isinstance(temp_shap, list):
                temp_shap = temp_shap[0] if is_classification else temp_shap
            
            shap_values_pd.append(temp_shap[:, feature_idx].mean())
    
    # Plot partial dependence
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=feature_range,
        y=shap_values_pd,
        mode='lines+markers',
        name=f'Partial Dependence'
    ))
    
    fig.update_layout(
        title=f"Partial Dependence: {selected_feature}",
        xaxis_title=selected_feature,
        yaxis_title="Average SHAP Value"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Min Value", f"{X[selected_feature].min():.3f}")
    with col2:
        st.metric("Mean Value", f"{X[selected_feature].mean():.3f}")
    with col3:
        st.metric("Max Value", f"{X[selected_feature].max():.3f}")

else:  # Interaction Effects
    st.header("🔗 Feature Interaction Effects")
    st.markdown("Discover how features interact with each other to influence predictions.")
    
    # Select two features for interaction analysis
    col1, col2 = st.columns(2)
    
    with col1:
        feature1 = st.selectbox(
            "Select first feature:",
            feature_names,
            key="feature1"
        )
    
    with col2:
        available_features = [f for f in feature_names if f != feature1]
        feature2 = st.selectbox(
            "Select second feature:",
            available_features,
            key="feature2"
        )
    
    if st.button("Analyze Interaction", type="primary"):
        with st.spinner("Computing interaction effects..."):
            # Get feature indices
            idx1 = feature_names.index(feature1)
            idx2 = feature_names.index(feature2)
            
            # Interaction plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            try:
                shap.dependence_plot(
                    idx1,
                    shap_values_display,
                    X_sample,
                    feature_names=feature_names,
                    interaction_index=idx2,
                    show=False
                )
                st.pyplot(fig, clear_figure=True)
                plt.close()
                
                # Correlation analysis
                st.subheader("📊 Interaction Statistics")
                
                correlation = X[feature1].corr(X[feature2])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Feature Correlation", f"{correlation:.3f}")
                with col2:
                    shap_corr = np.corrcoef(
                        shap_values_display[:, idx1],
                        shap_values_display[:, idx2]
                    )[0, 1]
                    st.metric("SHAP Correlation", f"{shap_corr:.3f}")
                with col3:
                    interaction_strength = np.abs(shap_corr - correlation)
                    st.metric("Interaction Strength", f"{interaction_strength:.3f}")
                
            except Exception as e:
                st.error(f"Error computing interaction plot: {str(e)}")

# Export results
st.sidebar.markdown("---")
if st.sidebar.button("📥 Export SHAP Analysis"):
    # Create export data
    export_data = {
        'feature_importance': pd.DataFrame({
            'Feature': feature_names,
            'Mean_SHAP': np.abs(shap_values_display).mean(axis=0)
        }),
        'sample_size': sample_size,
        'analysis_type': analysis_type
    }
    
    # Convert to JSON
    export_json = {
        'feature_importance': export_data['feature_importance'].to_dict('records'),
        'sample_size': export_data['sample_size'],
        'analysis_type': export_data['analysis_type']
    }
    
    st.sidebar.download_button(
        label="Download SHAP Results",
        data=pd.DataFrame(export_data['feature_importance']).to_csv(index=False),
        file_name="shap_analysis_results.csv",
        mime="text/csv"
    )

# Information panel
with st.expander("ℹ️ About SHAP Analysis"):
    st.markdown("""
    ### What is SHAP?
    SHAP (SHapley Additive exPlanations) is a game theory approach to explain machine learning models.
    It provides unified measure of feature importance that satisfies efficiency, symmetry, dummy, and additivity properties.
    
    ### Analysis Types:
    - **Global Feature Importance**: Shows which features are most important across all predictions
    - **Local Explanations**: Explains individual predictions in detail
    - **Partial Dependence**: Shows how features affect predictions across their range
    - **Interaction Effects**: Reveals how features interact with each other
    
    ### Interpreting Results:
    - **Positive SHAP values**: Push the prediction above the baseline
    - **Negative SHAP values**: Push the prediction below the baseline
    - **Magnitude**: Indicates the strength of the feature's influence
    """)
