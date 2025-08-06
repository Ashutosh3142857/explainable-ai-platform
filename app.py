import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine
try:
    from sklearn.datasets import load_diabetes  # Boston dataset is deprecated
except ImportError:
    load_diabetes = None
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import plotly.express as px
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="ExplainableAI Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<div class="main-header">🔍 ExplainableAI Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Making Black Box AI Models Transparent and Interpretable</div>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    ## Welcome to ExplainableAI Platform
    
    This platform helps you understand and interpret machine learning models through advanced explainability techniques.
    Our tools make complex AI decisions transparent and actionable for business stakeholders.
    """)
    
    # Features overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Key Features
        - **Model Upload & Analysis**: Support for scikit-learn, PyTorch, and TensorFlow models
        - **SHAP Integration**: Global and local feature importance using Shapley values
        - **LIME Explanations**: Local interpretable model-agnostic explanations
        - **Interactive Visualizations**: Rich, interactive charts and graphs
        - **Business-Ready Reports**: Non-technical explanations for stakeholders
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Supported Models
        - **Tree-based Models**: Random Forest, XGBoost, LightGBM
        - **Linear Models**: Logistic Regression, Linear Regression
        - **Neural Networks**: Basic feedforward networks
        - **Ensemble Methods**: Voting classifiers, bagging models
        - **Custom Models**: Upload your own trained models
        """)
    
    # Quick demo section
    st.markdown("## 🚀 Quick Demo")
    
    if st.button("Generate Sample Analysis", type="primary"):
        with st.spinner("Generating sample analysis..."):
            # Load sample data
            iris = load_iris()
            X, y = iris.data, iris.target
            feature_names = iris.feature_names
            
            # Train a simple model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Model Accuracy", f"{accuracy:.3f}")
            with col2:
                st.metric("Features Used", len(feature_names))
            with col3:
                st.metric("Test Samples", len(X_test))
            
            # Feature importance
            importances = model.feature_importances_
            
            fig = px.bar(
                x=feature_names,
                y=importances,
                title="Feature Importance (Random Forest - Iris Dataset)",
                labels={'x': 'Features', 'y': 'Importance'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("✅ Sample analysis complete! Explore the sidebar to access full features.")
    
    # Navigation guide
    st.markdown("""
    ## 📋 Getting Started
    
    1. **Upload Your Model**: Use the "Model Upload" page to upload your trained model
    2. **SHAP Analysis**: Get global and local explanations using Shapley values
    3. **LIME Analysis**: Generate local interpretable explanations for individual predictions
    4. **Examples**: Explore pre-built examples with sample datasets
    
    ### 💡 Tips for Best Results
    - Ensure your data is properly preprocessed
    - Use feature names for better interpretability
    - Start with our examples to understand the platform
    - Review explanations with domain experts
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>ExplainableAI Platform - Making AI Transparent and Trustworthy</p>
        <p>Built with ❤️ using Streamlit, SHAP, and LIME</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
