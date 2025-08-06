import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import load_iris, load_wine, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Import explanation libraries with error handling
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

st.set_page_config(page_title="Examples", page_icon="📚", layout="wide")

st.title("📚 Examples & Demonstrations")
st.markdown("Explore pre-built examples to understand explainable AI concepts and techniques.")

# Example selection
st.sidebar.header("Select Example")
example_type = st.sidebar.selectbox(
    "Choose an example:",
    ["Iris Classification", "Wine Classification", "Diabetes Regression", "Custom Comparison"]
)

def run_iris_example():
    st.header("🌸 Iris Classification Example")
    st.markdown("Classic iris dataset with Random Forest classification and explainability analysis.")
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Model performance
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Accuracy", f"{train_acc:.3f}")
    with col2:
        st.metric("Test Accuracy", f"{test_acc:.3f}")
    with col3:
        st.metric("Number of Features", len(feature_names))
    
    # Data visualization
    st.subheader("📊 Dataset Overview")
    
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = [target_names[i] for i in y]
    
    # Scatter plot matrix
    fig = px.scatter_matrix(
        df,
        dimensions=feature_names,
        color='species',
        title="Iris Dataset - Feature Relationships"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("🎯 Random Forest Feature Importance")
    
    importances = model.feature_importances_
    
    fig = px.bar(
        x=feature_names,
        y=importances,
        title="Feature Importance",
        labels={'x': 'Features', 'y': 'Importance'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # SHAP Analysis
    st.subheader("📊 SHAP Analysis")
    
    if not SHAP_AVAILABLE:
        st.warning("SHAP is not available in this environment. Please use the LIME analysis below.")
    elif st.button("Run SHAP Analysis", key="iris_shap"):
        with st.spinner("Computing SHAP values..."):
            # Create explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            # Summary plot
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values[0], X_test, feature_names=feature_names, show=False)
            st.pyplot(fig, clear_figure=True)
            
            # Feature importance comparison
            st.subheader("🔍 SHAP vs Random Forest Importance")
            
            shap_importance = np.abs(shap_values[0]).mean(axis=0)
            
            comparison_df = pd.DataFrame({
                'Feature': feature_names,
                'Random Forest': importances,
                'SHAP': shap_importance
            })
            
            fig = px.bar(
                comparison_df,
                x='Feature',
                y=['Random Forest', 'SHAP'],
                barmode='group',
                title="Feature Importance Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # LIME Analysis
    st.subheader("🍋 LIME Analysis")
    
    if st.button("Run LIME Analysis", key="iris_lime"):
        with st.spinner("Generating LIME explanations..."):
            # Create LIME explainer
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train,
                feature_names=feature_names,
                class_names=target_names,
                mode='classification'
            )
            
            # Explain a few instances
            instances_to_explain = [0, 1, 2]
            
            for i, idx in enumerate(instances_to_explain):
                instance = X_test[idx]
                prediction = model.predict([instance])[0]
                probabilities = model.predict_proba([instance])[0]
                
                explanation = lime_explainer.explain_instance(
                    instance,
                    model.predict_proba,
                    num_features=len(feature_names)
                )
                
                exp_data = explanation.as_list()
                
                st.write(f"**Instance {idx + 1}**: Predicted as {target_names[prediction]} (Confidence: {probabilities[prediction]:.3f})")
                
                features = [item[0] for item in exp_data]
                importances = [item[1] for item in exp_data]
                colors = ['green' if imp > 0 else 'red' for imp in importances]
                
                fig = go.Figure(data=[
                    go.Bar(
                        y=features,
                        x=importances,
                        orientation='h',
                        marker_color=colors,
                        name=f"Instance {idx + 1}"
                    )
                ])
                
                fig.update_layout(
                    title=f"LIME Explanation - Instance {idx + 1}",
                    xaxis_title="Importance",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)

def run_wine_example():
    st.header("🍷 Wine Classification Example")
    st.markdown("Wine dataset with multiple algorithms and explainability comparison.")
    
    # Load data
    wine = load_wine()
    X, y = wine.data, wine.target
    feature_names = wine.feature_names
    target_names = wine.target_names
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train multiple models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = {'model': model, 'accuracy': test_acc}
    
    # Model comparison
    st.subheader("🔍 Model Comparison")
    
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Test Accuracy': [results[name]['accuracy'] for name in results.keys()]
    })
    
    fig = px.bar(
        comparison_df,
        x='Model',
        y='Test Accuracy',
        title="Model Performance Comparison"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature analysis
    st.subheader("📊 Feature Analysis")
    
    selected_model = st.selectbox("Select model for analysis:", list(models.keys()))
    model = results[selected_model]['model']
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Top features
        top_features_idx = np.argsort(importances)[-10:]
        top_features = [feature_names[i] for i in top_features_idx]
        top_importances = importances[top_features_idx]
        
        fig = px.bar(
            x=top_importances,
            y=top_features,
            orientation='h',
            title=f"Top 10 Features - {selected_model}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif hasattr(model, 'coef_'):
        coefs = np.abs(model.coef_).mean(axis=0)
        
        # Top features
        top_features_idx = np.argsort(coefs)[-10:]
        top_features = [feature_names[i] for i in top_features_idx]
        top_coefs = coefs[top_features_idx]
        
        fig = px.bar(
            x=top_coefs,
            y=top_features,
            orientation='h',
            title=f"Top 10 Feature Coefficients - {selected_model}"
        )
        st.plotly_chart(fig, use_container_width=True)

def run_diabetes_example():
    st.header("🏥 Diabetes Regression Example")
    st.markdown("Diabetes progression prediction with regression models and explainability.")
    
    # Load data
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    feature_names = diabetes.feature_names
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = model.score(X_test, y_test)
        results[name] = {'model': model, 'rmse': rmse, 'r2': r2, 'predictions': predictions}
    
    # Model performance
    st.subheader("📈 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        rmse_df = pd.DataFrame({
            'Model': list(results.keys()),
            'RMSE': [results[name]['rmse'] for name in results.keys()]
        })
        
        fig = px.bar(rmse_df, x='Model', y='RMSE', title="RMSE Comparison")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        r2_df = pd.DataFrame({
            'Model': list(results.keys()),
            'R² Score': [results[name]['r2'] for name in results.keys()]
        })
        
        fig = px.bar(r2_df, x='Model', y='R² Score', title="R² Score Comparison")
        st.plotly_chart(fig, use_container_width=True)
    
    # Prediction analysis
    st.subheader("🎯 Prediction Analysis")
    
    selected_model = st.selectbox("Select model for analysis:", list(models.keys()), key="diabetes_model")
    model = results[selected_model]['model']
    predictions = results[selected_model]['predictions']
    
    # Actual vs Predicted
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test,
        y=predictions,
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
        title=f"Actual vs Predicted - {selected_model}",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        st.subheader("🎯 Feature Importance")
        
        importances = model.feature_importances_
        
        fig = px.bar(
            x=feature_names,
            y=importances,
            title="Feature Importance"
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    elif hasattr(model, 'coef_'):
        st.subheader("🎯 Feature Coefficients")
        
        coefs = model.coef_
        
        fig = px.bar(
            x=feature_names,
            y=coefs,
            title="Feature Coefficients"
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

def run_custom_comparison():
    st.header("⚖️ Custom Model Comparison")
    st.markdown("Compare different explainability methods on the same model and dataset.")
    
    # Dataset selection
    dataset_choice = st.selectbox(
        "Choose dataset:",
        ["Iris", "Wine", "Diabetes"]
    )
    
    # Load selected dataset
    if dataset_choice == "Iris":
        data = load_iris()
        is_classification = True
    elif dataset_choice == "Wine":
        data = load_wine()
        is_classification = True
    else:
        data = load_diabetes()
        is_classification = False
    
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Model selection
    if is_classification:
        model_options = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
    else:
        model_options = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
    
    selected_model_name = st.selectbox("Choose model:", list(model_options.keys()))
    model = model_options[selected_model_name]
    
    # Train model
    model.fit(X_train, y_train)
    
    # Model performance
    if is_classification:
        accuracy = accuracy_score(y_test, model.predict(X_test))
        st.metric("Test Accuracy", f"{accuracy:.3f}")
    else:
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        r2 = model.score(X_test, y_test)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("RMSE", f"{rmse:.3f}")
        with col2:
            st.metric("R² Score", f"{r2:.3f}")
    
    # Comparison analysis
    st.subheader("🔍 Explainability Method Comparison")
    
    methods = st.multiselect(
        "Select methods to compare:",
        ["Model Feature Importance", "SHAP", "LIME"],
        default=["Model Feature Importance"]
    )
    
    if st.button("Run Comparison Analysis", type="primary"):
        comparison_results = {}
        
        # Model feature importance
        if "Model Feature Importance" in methods:
            if hasattr(model, 'feature_importances_'):
                comparison_results["Model Feature Importance"] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                if len(model.coef_.shape) == 1:
                    comparison_results["Model Feature Importance"] = np.abs(model.coef_)
                else:
                    comparison_results["Model Feature Importance"] = np.abs(model.coef_).mean(axis=0)
        
        # SHAP
        if "SHAP" in methods:
            with st.spinner("Computing SHAP values..."):
                try:
                    if hasattr(model, 'estimators_') or 'forest' in str(type(model)).lower():
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_test[:100])  # Limit for performance
                    else:
                        explainer = shap.KernelExplainer(model.predict, X_train[:50])
                        shap_values = explainer.shap_values(X_test[:100])
                    
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]
                    
                    comparison_results["SHAP"] = np.abs(shap_values).mean(axis=0)
                except Exception as e:
                    st.error(f"SHAP analysis failed: {str(e)}")
        
        # LIME
        if "LIME" in methods:
            with st.spinner("Computing LIME explanations..."):
                try:
                    if is_classification:
                        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                            X_train,
                            feature_names=feature_names,
                            mode='classification'
                        )
                        lime_importances = []
                        
                        for i in range(min(20, len(X_test))):  # Limit for performance
                            explanation = lime_explainer.explain_instance(
                                X_test[i],
                                model.predict_proba,
                                num_features=len(feature_names)
                            )
                            exp_dict = dict(explanation.as_list())
                            lime_importances.append([abs(exp_dict.get(f, 0)) for f in feature_names])
                    else:
                        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                            X_train,
                            feature_names=feature_names,
                            mode='regression'
                        )
                        lime_importances = []
                        
                        for i in range(min(20, len(X_test))):  # Limit for performance
                            explanation = lime_explainer.explain_instance(
                                X_test[i],
                                model.predict,
                                num_features=len(feature_names)
                            )
                            exp_dict = dict(explanation.as_list())
                            lime_importances.append([abs(exp_dict.get(f, 0)) for f in feature_names])
                    
                    comparison_results["LIME"] = np.mean(lime_importances, axis=0)
                except Exception as e:
                    st.error(f"LIME analysis failed: {str(e)}")
        
        # Create comparison visualization
        if comparison_results:
            st.subheader("📊 Method Comparison Results")
            
            # Normalize results for comparison
            normalized_results = {}
            for method, values in comparison_results.items():
                normalized_results[method] = values / np.max(values)
            
            # Create comparison dataframe
            comparison_df = pd.DataFrame(normalized_results, index=feature_names)
            
            # Heatmap
            fig = px.imshow(
                comparison_df.T,
                title="Normalized Feature Importance Comparison",
                labels=dict(x="Features", y="Methods", color="Importance"),
                aspect="auto"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation analysis
            if len(comparison_results) > 1:
                st.subheader("🔗 Method Correlation")
                
                corr_matrix = comparison_df.corr()
                
                fig = px.imshow(
                    corr_matrix,
                    title="Correlation Between Explainability Methods",
                    text_auto=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                st.subheader("📈 Summary Statistics")
                st.dataframe(comparison_df.describe().round(3), use_container_width=True)

# Run selected example
if example_type == "Iris Classification":
    run_iris_example()
elif example_type == "Wine Classification":
    run_wine_example()
elif example_type == "Diabetes Regression":
    run_diabetes_example()
else:
    run_custom_comparison()

# Footer information
st.markdown("---")
with st.expander("ℹ️ About These Examples"):
    st.markdown("""
    ### Dataset Information:
    
    **Iris Dataset:**
    - 150 samples, 4 features, 3 classes
    - Classic classification problem
    - Features: sepal/petal length and width
    
    **Wine Dataset:**
    - 178 samples, 13 features, 3 classes
    - Wine quality classification
    - Features: chemical properties
    
    **Diabetes Dataset:**
    - 442 samples, 10 features, continuous target
    - Diabetes progression prediction
    - Features: age, sex, BMI, blood pressure, etc.
    
    ### Learning Objectives:
    - Compare different explainability methods
    - Understand model behavior through visualizations
    - See how different algorithms explain the same data
    - Learn best practices for model interpretation
    """)
