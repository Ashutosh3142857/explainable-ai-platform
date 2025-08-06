import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
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

class SampleDatasets:
    """
    Utility class for loading and preparing sample datasets with pre-trained models
    for demonstration purposes in the ExplainableAI platform.
    """
    
    def __init__(self):
        self.random_state = 42
        
    def get_iris_example(self):
        """
        Load Iris dataset with a pre-trained Random Forest classifier
        
        Returns:
            tuple: (model, data, feature_names, target_names)
        """
        # Load iris dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        feature_names = iris.feature_names
        target_names = iris.target_names
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['species'] = y
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        # Train Random Forest model
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=self.random_state,
            max_depth=3  # Prevent overfitting for demo
        )
        model.fit(X_train, y_train)
        
        # Add model performance info
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        test_accuracy = accuracy_score(y_test, model.predict(X_test))
        
        print(f"Iris model performance - Train: {train_accuracy:.3f}, Test: {test_accuracy:.3f}")
        
        return model, df, feature_names, target_names
    
    def get_wine_example(self):
        """
        Load Wine dataset with a pre-trained Random Forest classifier
        
        Returns:
            tuple: (model, data, feature_names, target_names)
        """
        # Load wine dataset
        wine = load_wine()
        X, y = wine.data, wine.target
        feature_names = wine.feature_names
        target_names = wine.target_names
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['wine_class'] = y
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        # Train Random Forest model
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=self.random_state,
            max_depth=5  # Reasonable depth for wine dataset
        )
        model.fit(X_train, y_train)
        
        # Add model performance info
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        test_accuracy = accuracy_score(y_test, model.predict(X_test))
        
        print(f"Wine model performance - Train: {train_accuracy:.3f}, Test: {test_accuracy:.3f}")
        
        return model, df, feature_names, target_names
    
    def get_boston_example(self):
        """
        Load Diabetes dataset (replacement for Boston housing) with a pre-trained Random Forest regressor
        
        Returns:
            tuple: (model, data, feature_names, target_names)
        """
        # Load diabetes dataset (Boston housing is deprecated)
        diabetes = load_diabetes()
        X, y = diabetes.data, diabetes.target
        feature_names = diabetes.feature_names
        target_names = ['diabetes_progression']
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['diabetes_progression'] = y
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        # Train Random Forest model
        model = RandomForestRegressor(
            n_estimators=100, 
            random_state=self.random_state,
            max_depth=5  # Prevent overfitting
        )
        model.fit(X_train, y_train)
        
        # Add model performance info
        train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
        test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        test_r2 = model.score(X_test, y_test)
        
        print(f"Diabetes model performance - Train RMSE: {train_rmse:.3f}, Test RMSE: {test_rmse:.3f}, R²: {test_r2:.3f}")
        
        return model, df, feature_names, target_names
    
    def get_classification_comparison_models(self, dataset_name='iris'):
        """
        Get multiple classification models for comparison
        
        Args:
            dataset_name: 'iris' or 'wine'
            
        Returns:
            dict: Dictionary with model names as keys and (model, accuracy) as values
        """
        if dataset_name == 'iris':
            data = load_iris()
        elif dataset_name == 'wine':
            data = load_wine()
        else:
            raise ValueError("Supported datasets: 'iris', 'wine'")
        
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state,
                max_depth=5
            ),
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                solver='lbfgs'
            )
        }
        
        # Train and evaluate models
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, model.predict(X_test))
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
            
        return results, data.feature_names, data.target_names
    
    def get_regression_comparison_models(self):
        """
        Get multiple regression models for comparison
        
        Returns:
            dict: Dictionary with model names as keys and model info as values
        """
        diabetes = load_diabetes()
        X, y = diabetes.data, diabetes.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        # Define models
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                random_state=self.random_state,
                max_depth=5
            ),
            'Linear Regression': LinearRegression()
        }
        
        # Train and evaluate models
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = model.score(X_test, y_test)
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'r2': r2,
                'predictions': predictions,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
            
        return results, diabetes.feature_names
    
    def create_synthetic_dataset(self, n_samples=1000, n_features=10, n_classes=3, 
                                noise=0.1, random_state=None):
        """
        Create a synthetic dataset for testing explainability methods
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            n_classes: Number of classes (for classification)
            noise: Noise level
            random_state: Random state for reproducibility
            
        Returns:
            tuple: (X, y, feature_names)
        """
        if random_state is None:
            random_state = self.random_state
            
        np.random.seed(random_state)
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Generate target with some meaningful relationships
        # First few features are important, rest are noise
        important_features = min(3, n_features)
        
        if n_classes > 1:  # Classification
            # Create target based on linear combination of important features
            target_continuous = (
                X[:, 0] * 2.0 + 
                X[:, 1] * 1.5 + 
                (X[:, 2] * 1.0 if n_features > 2 else 0) +
                np.random.normal(0, noise, n_samples)
            )
            
            # Convert to classes
            y = np.digitize(target_continuous, 
                          bins=np.percentile(target_continuous, 
                                           np.linspace(0, 100, n_classes + 1)[1:-1]))
            y = np.clip(y, 0, n_classes - 1)
            
        else:  # Regression
            y = (
                X[:, 0] * 2.0 + 
                X[:, 1] * 1.5 + 
                (X[:, 2] * 1.0 if n_features > 2 else 0) +
                np.random.normal(0, noise, n_samples)
            )
        
        # Create feature names
        feature_names = [f"feature_{i}" for i in range(n_features)]
        
        return X, y, feature_names
    
    def get_dataset_info(self, dataset_name):
        """
        Get information about available datasets
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            dict: Dataset information
        """
        info = {
            'iris': {
                'description': 'Classic iris flower classification dataset',
                'type': 'classification',
                'samples': 150,
                'features': 4,
                'classes': 3,
                'feature_names': ['sepal length', 'sepal width', 'petal length', 'petal width'],
                'target_names': ['setosa', 'versicolor', 'virginica']
            },
            'wine': {
                'description': 'Wine recognition dataset',
                'type': 'classification', 
                'samples': 178,
                'features': 13,
                'classes': 3,
                'feature_names': 'chemical properties of wine',
                'target_names': ['class_0', 'class_1', 'class_2']
            },
            'diabetes': {
                'description': 'Diabetes progression prediction dataset',
                'type': 'regression',
                'samples': 442,
                'features': 10,
                'target': 'disease progression after one year',
                'feature_names': ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
            }
        }
        
        return info.get(dataset_name, {})
    
    def validate_model_compatibility(self, model, X, y):
        """
        Validate that a model is compatible with explainability tools
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            
        Returns:
            dict: Compatibility report
        """
        compatibility = {
            'shap_compatible': False,
            'lime_compatible': False,
            'feature_importance_available': False,
            'issues': []
        }
        
        try:
            # Test basic prediction
            sample = X[:1] if len(X.shape) == 2 else X.iloc[:1]
            prediction = model.predict(sample)
            compatibility['can_predict'] = True
        except Exception as e:
            compatibility['issues'].append(f"Prediction failed: {str(e)}")
            return compatibility
        
        # Test SHAP compatibility
        try:
            import shap
            if hasattr(model, 'estimators_') or 'forest' in str(type(model)).lower():
                explainer = shap.TreeExplainer(model)
                compatibility['shap_compatible'] = True
            else:
                # Test KernelExplainer with small sample
                background = X[:min(10, len(X))]
                explainer = shap.KernelExplainer(model.predict, background)
                compatibility['shap_compatible'] = True
        except Exception as e:
            compatibility['issues'].append(f"SHAP compatibility issue: {str(e)}")
        
        # Test LIME compatibility
        try:
            import lime.lime_tabular
            if hasattr(model, 'predict_proba'):
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    X.values if hasattr(X, 'values') else X,
                    mode='classification'
                )
                compatibility['lime_compatible'] = True
            else:
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    X.values if hasattr(X, 'values') else X,
                    mode='regression'
                )
                compatibility['lime_compatible'] = True
        except Exception as e:
            compatibility['issues'].append(f"LIME compatibility issue: {str(e)}")
        
        # Check feature importance availability
        if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
            compatibility['feature_importance_available'] = True
        
        return compatibility

