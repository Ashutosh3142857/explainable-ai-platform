import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import warnings

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

class ExplanationEngine:
    """Main engine for generating model explanations"""
    
    def __init__(self, model, X_train, feature_names=None, target_names=None):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(X_train.shape[1])]
        self.target_names = target_names
        self.is_classifier = hasattr(model, 'predict_proba')
        
        # Initialize explainers lazily
        self._shap_explainer = None
        self._lime_explainer = None
    
    def get_shap_explainer(self):
        """Get or create SHAP explainer"""
        if self._shap_explainer is None:
            try:
                # Try TreeExplainer first (fastest for tree-based models)
                if hasattr(self.model, 'estimators_') or 'forest' in str(type(self.model)).lower():
                    self._shap_explainer = shap.TreeExplainer(self.model)
                else:
                    # Fall back to KernelExplainer
                    background = shap.sample(self.X_train, min(100, len(self.X_train)))
                    self._shap_explainer = shap.KernelExplainer(self.model.predict, background)
            except Exception as e:
                warnings.warn(f"Could not create SHAP explainer: {str(e)}")
                return None
        
        return self._shap_explainer
    
    def get_lime_explainer(self):
        """Get or create LIME explainer"""
        if self._lime_explainer is None:
            try:
                if self.is_classifier:
                    self._lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                        self.X_train.values if hasattr(self.X_train, 'values') else self.X_train,
                        feature_names=self.feature_names,
                        class_names=self.target_names or ['Class 0', 'Class 1'],
                        mode='classification',
                        discretize_continuous=True
                    )
                else:
                    self._lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                        self.X_train.values if hasattr(self.X_train, 'values') else self.X_train,
                        feature_names=self.feature_names,
                        mode='regression',
                        discretize_continuous=True
                    )
            except Exception as e:
                warnings.warn(f"Could not create LIME explainer: {str(e)}")
                return None
        
        return self._lime_explainer
    
    def explain_shap_global(self, X_sample, max_display=10):
        """Generate global SHAP explanations"""
        explainer = self.get_shap_explainer()
        if explainer is None:
            return None
        
        try:
            shap_values = explainer.shap_values(X_sample)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                if len(shap_values) == 2:  # Binary classification
                    shap_values = shap_values[1]  # Use positive class
                else:  # Multi-class - use first class or average
                    shap_values = shap_values[0]
            
            # Calculate feature importance
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            # Get top features
            top_indices = np.argsort(feature_importance)[-max_display:]
            
            return {
                'shap_values': shap_values,
                'feature_importance': feature_importance,
                'top_features': [self.feature_names[i] for i in top_indices],
                'top_importance': feature_importance[top_indices],
                'explainer': explainer
            }
        
        except Exception as e:
            warnings.warn(f"SHAP global explanation failed: {str(e)}")
            return None
    
    def explain_shap_local(self, instance, explainer=None):
        """Generate local SHAP explanation for a single instance"""
        if explainer is None:
            explainer = self.get_shap_explainer()
        
        if explainer is None:
            return None
        
        try:
            # Ensure instance is in correct format
            if hasattr(instance, 'values'):
                instance_array = instance.values.reshape(1, -1)
            else:
                instance_array = instance.reshape(1, -1)
            
            shap_values = explainer.shap_values(instance_array)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                if len(shap_values) == 2:
                    shap_values = shap_values[1][0]  # Binary classification, positive class
                else:
                    shap_values = shap_values[0][0]  # Multi-class, first class
            else:
                shap_values = shap_values[0]
            
            return {
                'shap_values': shap_values,
                'feature_names': self.feature_names,
                'feature_values': instance_array[0],
                'expected_value': explainer.expected_value
            }
        
        except Exception as e:
            warnings.warn(f"SHAP local explanation failed: {str(e)}")
            return None
    
    def explain_lime_local(self, instance, num_features=10, num_samples=1000):
        """Generate local LIME explanation for a single instance"""
        explainer = self.get_lime_explainer()
        if explainer is None:
            return None
        
        try:
            # Ensure instance is in correct format
            if hasattr(instance, 'values'):
                instance_array = instance.values.flatten()
            else:
                instance_array = instance.flatten()
            
            if self.is_classifier:
                explanation = explainer.explain_instance(
                    instance_array,
                    self.model.predict_proba,
                    num_features=num_features,
                    num_samples=num_samples
                )
            else:
                explanation = explainer.explain_instance(
                    instance_array,
                    self.model.predict,
                    num_features=num_features,
                    num_samples=num_samples
                )
            
            # Extract explanation data
            exp_list = explanation.as_list()
            
            return {
                'explanation': explanation,
                'feature_importance': {item[0]: item[1] for item in exp_list},
                'features': [item[0] for item in exp_list],
                'importances': [item[1] for item in exp_list],
                'score': explanation.score,
                'local_pred': explanation.local_pred
            }
        
        except Exception as e:
            warnings.warn(f"LIME local explanation failed: {str(e)}")
            return None
    
    def create_feature_importance_plot(self, importance_dict, title="Feature Importance"):
        """Create a feature importance plot"""
        features = list(importance_dict.keys())
        importances = list(importance_dict.values())
        
        # Sort by absolute importance
        sorted_indices = np.argsort(np.abs(importances))
        features = [features[i] for i in sorted_indices]
        importances = [importances[i] for i in sorted_indices]
        
        # Color coding
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
            title=title,
            xaxis_title="Importance",
            yaxis_title="Features",
            height=max(400, len(features) * 30)
        )
        
        return fig
    
    def create_shap_summary_plot(self, shap_values, X_sample, max_display=10):
        """Create a SHAP summary plot using Plotly"""
        # Calculate feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        # Get top features
        top_indices = np.argsort(feature_importance)[-max_display:]
        top_features = [self.feature_names[i] for i in top_indices]
        top_shap_values = shap_values[:, top_indices]
        
        # Create scatter plot for each feature
        fig = go.Figure()
        
        for i, feature_idx in enumerate(top_indices):
            feature_name = self.feature_names[feature_idx]
            feature_values = X_sample.iloc[:, feature_idx] if hasattr(X_sample, 'iloc') else X_sample[:, feature_idx]
            shap_vals = shap_values[:, feature_idx]
            
            fig.add_trace(go.Scatter(
                x=shap_vals,
                y=[feature_name] * len(shap_vals),
                mode='markers',
                marker=dict(
                    color=feature_values,
                    colorscale='Viridis',
                    size=4,
                    opacity=0.6
                ),
                name=feature_name,
                hovertemplate=f"<b>{feature_name}</b><br>" +
                             "SHAP value: %{x:.3f}<br>" +
                             "Feature value: %{marker.color:.3f}<br>" +
                             "<extra></extra>"
            ))
        
        fig.update_layout(
            title="SHAP Summary Plot",
            xaxis_title="SHAP Value",
            yaxis_title="Features",
            height=max(400, len(top_features) * 40),
            showlegend=False
        )
        
        return fig
    
    def compare_explanations(self, instance, methods=['shap', 'lime']):
        """Compare different explanation methods for the same instance"""
        results = {}
        
        if 'shap' in methods:
            shap_result = self.explain_shap_local(instance)
            if shap_result:
                results['SHAP'] = {
                    'importance': dict(zip(self.feature_names, shap_result['shap_values']))
                }
        
        if 'lime' in methods:
            lime_result = self.explain_lime_local(instance)
            if lime_result:
                results['LIME'] = {
                    'importance': lime_result['feature_importance']
                }
        
        # Model-based importance
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            results['Model Feature Importance'] = {
                'importance': dict(zip(self.feature_names, importance))
            }
        elif hasattr(self.model, 'coef_'):
            coef = self.model.coef_
            if len(coef.shape) > 1:
                importance = np.abs(coef).mean(axis=0)
            else:
                importance = np.abs(coef)
            results['Model Coefficients'] = {
                'importance': dict(zip(self.feature_names, importance))
            }
        
        return results

class ExplanationVisualizer:
    """Utility class for creating explanation visualizations"""
    
    @staticmethod
    def create_waterfall_plot(shap_values, feature_values, feature_names, expected_value):
        """Create a waterfall plot for SHAP values"""
        # Sort by absolute SHAP values
        sorted_indices = np.argsort(np.abs(shap_values))
        
        cumulative = expected_value
        cumulative_values = [cumulative]
        
        for idx in sorted_indices:
            cumulative += shap_values[idx]
            cumulative_values.append(cumulative)
        
        # Create the plot
        fig = go.Figure()
        
        # Base value
        fig.add_trace(go.Bar(
            x=['Base Value'],
            y=[expected_value],
            name='Base Value',
            marker_color='lightgray'
        ))
        
        # Feature contributions
        for i, idx in enumerate(sorted_indices):
            color = 'green' if shap_values[idx] > 0 else 'red'
            fig.add_trace(go.Bar(
                x=[feature_names[idx]],
                y=[shap_values[idx]],
                name=f"{feature_names[idx]}: {feature_values[idx]:.3f}",
                marker_color=color,
                base=cumulative_values[i],
                hovertemplate=f"<b>{feature_names[idx]}</b><br>" +
                             f"Value: {feature_values[idx]:.3f}<br>" +
                             f"SHAP: {shap_values[idx]:.3f}<br>" +
                             "<extra></extra>"
            ))
        
        # Final prediction
        fig.add_trace(go.Bar(
            x=['Prediction'],
            y=[cumulative_values[-1]],
            name='Final Prediction',
            marker_color='blue'
        ))
        
        fig.update_layout(
            title="SHAP Waterfall Plot",
            xaxis_title="Features",
            yaxis_title="Model Output",
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_partial_dependence_plot(model, X, feature_name, feature_idx, n_points=50):
        """Create a partial dependence plot for a specific feature"""
        feature_values = X.iloc[:, feature_idx] if hasattr(X, 'iloc') else X[:, feature_idx]
        
        # Create range of values
        feature_range = np.linspace(
            feature_values.min(),
            feature_values.max(),
            n_points
        )
        
        # Calculate partial dependence
        predictions = []
        for value in feature_range:
            X_temp = X.copy()
            if hasattr(X_temp, 'iloc'):
                X_temp.iloc[:, feature_idx] = value
            else:
                X_temp[:, feature_idx] = value
            
            pred = model.predict(X_temp).mean()
            predictions.append(pred)
        
        # Create plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=feature_range,
            y=predictions,
            mode='lines+markers',
            name='Partial Dependence'
        ))
        
        fig.update_layout(
            title=f"Partial Dependence: {feature_name}",
            xaxis_title=feature_name,
            yaxis_title="Partial Dependence"
        )
        
        return fig
