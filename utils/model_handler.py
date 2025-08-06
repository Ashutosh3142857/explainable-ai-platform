import joblib
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import io
import warnings

class ModelHandler:
    """Handles loading and validating different types of ML models"""
    
    def __init__(self):
        self.supported_formats = ['.pkl', '.joblib', '.json']
        
    def load_model(self, file_obj):
        """
        Load a model from a file object
        
        Args:
            file_obj: Streamlit file upload object
            
        Returns:
            Loaded model object
        """
        file_name = file_obj.name.lower()
        file_bytes = file_obj.read()
        
        try:
            if file_name.endswith('.pkl'):
                return self._load_pickle(file_bytes)
            elif file_name.endswith('.joblib'):
                return self._load_joblib(file_bytes)
            elif file_name.endswith('.json'):
                return self._load_json(file_bytes)
            else:
                raise ValueError(f"Unsupported file format. Supported formats: {self.supported_formats}")
                
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")
    
    def _load_pickle(self, file_bytes):
        """Load pickle file"""
        try:
            model = pickle.loads(file_bytes)
            return self._validate_model(model)
        except Exception as e:
            raise ValueError(f"Error loading pickle file: {str(e)}")
    
    def _load_joblib(self, file_bytes):
        """Load joblib file"""
        try:
            # Save bytes to temporary file-like object
            file_like = io.BytesIO(file_bytes)
            model = joblib.load(file_like)
            return self._validate_model(model)
        except Exception as e:
            raise ValueError(f"Error loading joblib file: {str(e)}")
    
    def _load_json(self, file_bytes):
        """Load JSON file (for simple model configurations)"""
        try:
            model_config = json.loads(file_bytes.decode('utf-8'))
            # This would need custom implementation based on your JSON structure
            raise NotImplementedError("JSON model loading not implemented yet")
        except Exception as e:
            raise ValueError(f"Error loading JSON file: {str(e)}")
    
    def _validate_model(self, model):
        """
        Validate that the model has required methods for explainability
        
        Args:
            model: The loaded model object
            
        Returns:
            The validated model
        """
        required_methods = ['predict']
        
        for method in required_methods:
            if not hasattr(model, method):
                raise ValueError(f"Model must have '{method}' method for explainability analysis")
        
        # Additional validation for common model types
        if hasattr(model, 'predict_proba'):
            # Classification model
            pass
        elif hasattr(model, 'decision_function'):
            # SVM or similar
            pass
        
        return model
    
    def get_model_info(self, model):
        """
        Extract information about the model
        
        Args:
            model: The model object
            
        Returns:
            Dictionary with model information
        """
        info = {
            'type': type(model).__name__,
            'module': type(model).__module__,
            'methods': [method for method in dir(model) if not method.startswith('_')],
            'is_classifier': hasattr(model, 'predict_proba'),
            'is_sklearn': hasattr(model, 'get_params'),
        }
        
        # Sklearn specific info
        if info['is_sklearn']:
            try:
                info['parameters'] = model.get_params()
            except:
                info['parameters'] = {}
        
        # Feature information
        if hasattr(model, 'n_features_in_'):
            info['n_features'] = model.n_features_in_
        
        if hasattr(model, 'feature_names_in_'):
            info['feature_names'] = list(model.feature_names_in_)
        
        # Class information for classifiers
        if hasattr(model, 'classes_'):
            info['classes'] = list(model.classes_)
            info['n_classes'] = len(model.classes_)
        
        return info
    
    def validate_data_compatibility(self, model, X, y=None):
        """
        Check if data is compatible with the model
        
        Args:
            model: The model object
            X: Feature matrix
            y: Target vector (optional)
            
        Returns:
            Dictionary with compatibility information
        """
        compatibility = {
            'compatible': True,
            'issues': [],
            'warnings': []
        }
        
        try:
            # Check if model can make predictions
            if len(X) > 0:
                # Try with a single sample first
                sample = X.iloc[0:1] if hasattr(X, 'iloc') else X[0:1]
                
                try:
                    prediction = model.predict(sample)
                    compatibility['can_predict'] = True
                except Exception as e:
                    compatibility['compatible'] = False
                    compatibility['issues'].append(f"Model cannot make predictions: {str(e)}")
                    compatibility['can_predict'] = False
            
            # Check feature count compatibility
            if hasattr(model, 'n_features_in_'):
                expected_features = model.n_features_in_
                actual_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])
                
                if expected_features != actual_features:
                    compatibility['compatible'] = False
                    compatibility['issues'].append(
                        f"Feature count mismatch: model expects {expected_features}, got {actual_features}"
                    )
            
            # Check for missing values
            if hasattr(X, 'isnull'):
                if X.isnull().any().any():
                    compatibility['warnings'].append("Data contains missing values")
            
            # Check data types
            if hasattr(X, 'dtypes'):
                non_numeric = X.select_dtypes(exclude=[np.number]).columns
                if len(non_numeric) > 0:
                    compatibility['warnings'].append(
                        f"Non-numeric columns found: {list(non_numeric)}"
                    )
        
        except Exception as e:
            compatibility['compatible'] = False
            compatibility['issues'].append(f"Compatibility check failed: {str(e)}")
        
        return compatibility

class ModelWrapper:
    """Wrapper class to standardize different model interfaces"""
    
    def __init__(self, model):
        self.model = model
        self.model_info = ModelHandler().get_model_info(model)
    
    def predict(self, X):
        """Standardized predict method"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Standardized predict_proba method"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'decision_function'):
            # Convert decision function to probabilities for binary classification
            scores = self.model.decision_function(X)
            if len(scores.shape) == 1:  # Binary classification
                scores = scores.reshape(-1, 1)
                scores = np.hstack([-scores, scores])
            
            # Apply softmax
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        else:
            raise ValueError("Model does not support probability predictions")
    
    def get_feature_importance(self):
        """Get feature importance if available"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            coef = self.model.coef_
            if len(coef.shape) > 1:
                return np.abs(coef).mean(axis=0)
            else:
                return np.abs(coef)
        else:
            return None
    
    def is_classifier(self):
        """Check if model is a classifier"""
        return self.model_info['is_classifier']
    
    def get_classes(self):
        """Get class labels for classifiers"""
        if hasattr(self.model, 'classes_'):
            return self.model.classes_
        else:
            return None
