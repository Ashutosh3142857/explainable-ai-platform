import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import json
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.model_handler import ModelHandler
from utils.data_utils import DataProcessor

st.set_page_config(page_title="Model Upload", page_icon="📤", layout="wide")

st.title("📤 Model Upload & Analysis")
st.markdown("Upload your trained model and dataset to begin explainability analysis.")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'target_names' not in st.session_state:
    st.session_state.target_names = None

# Sidebar for upload options
st.sidebar.header("Upload Options")
upload_type = st.sidebar.selectbox(
    "Choose upload method:",
    ["Upload Files", "Use Example Data"]
)

if upload_type == "Upload Files":
    # Model upload section
    st.header("1. Upload Your Model")
    
    model_file = st.file_uploader(
        "Choose model file",
        type=['pkl', 'joblib', 'json'],
        help="Supported formats: .pkl (pickle), .joblib, .json"
    )
    
    # Data upload section
    st.header("2. Upload Your Dataset")
    
    data_file = st.file_uploader(
        "Choose data file",
        type=['csv', 'xlsx'],
        help="CSV or Excel file with your dataset"
    )
    
    if model_file and data_file:
        try:
            # Load model
            model_handler = ModelHandler()
            model = model_handler.load_model(model_file)
            
            # Load data
            data_processor = DataProcessor()
            df = data_processor.load_data(data_file)
            
            st.session_state.model = model
            st.session_state.data = df
            
            st.success("✅ Model and data loaded successfully!")
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Basic statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
                
        except Exception as e:
            st.error(f"Error loading files: {str(e)}")
            st.info("Please ensure your model is compatible and data is properly formatted.")

else:  # Use Example Data
    st.header("📊 Example Datasets")
    
    example_choice = st.selectbox(
        "Choose an example:",
        ["Iris Classification", "Wine Classification", "Boston Housing Regression"]
    )
    
    if st.button("Load Example", type="primary"):
        from data.sample_datasets import SampleDatasets
        
        sample_data = SampleDatasets()
        
        with st.spinner("Loading example data..."):
            if example_choice == "Iris Classification":
                model, data, feature_names, target_names = sample_data.get_iris_example()
            elif example_choice == "Wine Classification":
                model, data, feature_names, target_names = sample_data.get_wine_example()
            else:  # Boston Housing
                model, data, feature_names, target_names = sample_data.get_boston_example()
            
            st.session_state.model = model
            st.session_state.data = data
            st.session_state.feature_names = feature_names
            st.session_state.target_names = target_names
            
        st.success(f"✅ {example_choice} example loaded successfully!")
        st.rerun()

# Model analysis section
if st.session_state.model is not None and st.session_state.data is not None:
    st.header("3. Model Analysis")
    
    model = st.session_state.model
    df = st.session_state.data
    
    # Feature selection
    st.subheader("Feature Configuration")
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) == 0:
        st.error("No numeric columns found in the dataset.")
        st.stop()
    
    # Target column selection
    target_col = st.selectbox(
        "Select target column:",
        numeric_columns,
        index=len(numeric_columns)-1  # Default to last column
    )
    
    # Feature columns
    available_features = [col for col in numeric_columns if col != target_col]
    
    if len(available_features) == 0:
        st.error("No feature columns available after selecting target.")
        st.stop()
    
    selected_features = st.multiselect(
        "Select feature columns:",
        available_features,
        default=available_features
    )
    
    if len(selected_features) == 0:
        st.warning("Please select at least one feature column.")
        st.stop()
    
    # Prepare data
    X = df[selected_features]
    y = df[target_col]
    
    # Store feature names
    st.session_state.feature_names = selected_features
    
    # Model evaluation
    if st.button("Analyze Model Performance", type="primary"):
        try:
            # Split data for evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            # Make predictions
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_test)
                y_pred_train = model.predict(X_train)
            else:
                st.error("Model doesn't have a predict method.")
                st.stop()
            
            # Determine if classification or regression
            is_classification = len(np.unique(y)) <= 20 and y.dtype in ['int64', 'object', 'category']
            
            # Display metrics
            st.subheader("📈 Model Performance Metrics")
            
            col1, col2 = st.columns(2)
            
            if is_classification:
                # Classification metrics
                train_acc = accuracy_score(y_train, y_pred_train)
                test_acc = accuracy_score(y_test, y_pred)
                
                with col1:
                    st.metric("Training Accuracy", f"{train_acc:.3f}")
                    st.metric("Test Accuracy", f"{test_acc:.3f}")
                
                with col2:
                    st.metric("Overfitting", f"{train_acc - test_acc:.3f}")
                    st.metric("Unique Classes", len(np.unique(y)))
                
                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    title="Confusion Matrix",
                    labels=dict(x="Predicted", y="Actual")
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Classification Report
                if len(np.unique(y)) <= 10:  # Only show for reasonable number of classes
                    st.subheader("Classification Report")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.round(3), use_container_width=True)
            
            else:
                # Regression metrics
                train_mse = mean_squared_error(y_train, y_pred_train)
                test_mse = mean_squared_error(y_test, y_pred)
                train_rmse = np.sqrt(train_mse)
                test_rmse = np.sqrt(test_mse)
                
                with col1:
                    st.metric("Training RMSE", f"{train_rmse:.3f}")
                    st.metric("Test RMSE", f"{test_rmse:.3f}")
                
                with col2:
                    st.metric("Overfitting (RMSE)", f"{test_rmse - train_rmse:.3f}")
                    st.metric("R² Score", f"{model.score(X_test, y_test):.3f}")
                
                # Prediction vs Actual plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_test,
                    y=y_pred,
                    mode='markers',
                    name='Predictions',
                    opacity=0.7
                ))
                fig.add_trace(go.Scatter(
                    x=[y_test.min(), y_test.max()],
                    y=[y_test.min(), y_test.max()],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='red')
                ))
                fig.update_layout(
                    title="Predicted vs Actual Values",
                    xaxis_title="Actual Values",
                    yaxis_title="Predicted Values"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                st.subheader("🎯 Feature Importance")
                
                importances = model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'Feature': selected_features,
                    'Importance': importances
                }).sort_values('Importance', ascending=True)
                
                fig = px.bar(
                    feature_importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Feature Importance"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif hasattr(model, 'coef_'):
                st.subheader("🎯 Feature Coefficients")
                
                if len(model.coef_.shape) == 1:  # Single output
                    coefs = model.coef_
                else:  # Multi-class
                    coefs = np.abs(model.coef_).mean(axis=0)
                
                coef_df = pd.DataFrame({
                    'Feature': selected_features,
                    'Coefficient': coefs
                }).sort_values('Coefficient', ascending=True, key=abs)
                
                fig = px.bar(
                    coef_df,
                    x='Coefficient',
                    y='Feature',
                    orientation='h',
                    title="Feature Coefficients"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Store processed data for other pages
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.is_classification = is_classification
            
            st.success("✅ Model analysis complete! You can now proceed to SHAP or LIME analysis.")
            
        except Exception as e:
            st.error(f"Error during model analysis: {str(e)}")
            st.info("Please check your model compatibility and data format.")

# Instructions
if st.session_state.model is None:
    st.info("""
    ### 📝 Instructions
    1. **Upload your model**: Supported formats include pickle (.pkl), joblib, and JSON
    2. **Upload your dataset**: CSV or Excel files with your training/test data
    3. **Configure features**: Select target and feature columns
    4. **Analyze performance**: Review model metrics and feature importance
    
    ### 🔧 Model Requirements
    - Model must have a `predict()` method
    - Scikit-learn models are fully supported
    - Basic PyTorch/TensorFlow models with predict method
    - Feature names should match your dataset columns
    """)
