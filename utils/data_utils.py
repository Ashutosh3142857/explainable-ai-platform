import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings

class DataProcessor:
    """Utility class for data processing and validation"""
    
    def __init__(self):
        self.scaler = None
        self.label_encoders = {}
        self.imputer = None
        
    def load_data(self, file_obj):
        """
        Load data from uploaded file
        
        Args:
            file_obj: Streamlit file upload object
            
        Returns:
            pandas DataFrame
        """
        file_name = file_obj.name.lower()
        
        try:
            if file_name.endswith('.csv'):
                df = pd.read_csv(file_obj)
            elif file_name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_obj)
            else:
                raise ValueError("Unsupported file format. Please upload CSV or Excel files.")
            
            return self.validate_dataframe(df)
            
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")
    
    def validate_dataframe(self, df):
        """
        Validate and clean the dataframe
        
        Args:
            df: pandas DataFrame
            
        Returns:
            Validated DataFrame
        """
        if df.empty:
            raise ValueError("Dataset is empty")
        
        if df.shape[0] < 10:
            warnings.warn("Dataset has fewer than 10 rows. Results may not be reliable.")
        
        if df.shape[1] < 2:
            raise ValueError("Dataset must have at least 2 columns (features and target)")
        
        # Remove completely empty columns
        df = df.dropna(axis=1, how='all')
        
        # Remove completely empty rows
        df = df.dropna(axis=0, how='all')
        
        return df
    
    def get_data_summary(self, df):
        """
        Generate summary statistics for the dataset
        
        Args:
            df: pandas DataFrame
            
        Returns:
            Dictionary with summary information
        """
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        # Statistics for numeric columns
        if summary['numeric_columns']:
            summary['numeric_stats'] = df[summary['numeric_columns']].describe().to_dict()
        
        # Information about categorical columns
        if summary['categorical_columns']:
            summary['categorical_info'] = {}
            for col in summary['categorical_columns']:
                summary['categorical_info'][col] = {
                    'unique_values': df[col].nunique(),
                    'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                    'frequency': df[col].value_counts().head().to_dict()
                }
        
        return summary
    
    def preprocess_features(self, X, target_col=None, handle_missing='impute', scale_features=False):
        """
        Preprocess features for ML model compatibility
        
        Args:
            X: Feature DataFrame
            target_col: Target column name (to exclude from preprocessing)
            handle_missing: How to handle missing values ('impute', 'drop', 'none')
            scale_features: Whether to scale numeric features
            
        Returns:
            Preprocessed DataFrame
        """
        X_processed = X.copy()
        
        # Remove target column if present
        if target_col and target_col in X_processed.columns:
            X_processed = X_processed.drop(columns=[target_col])
        
        # Handle missing values
        if handle_missing == 'impute':
            X_processed = self._impute_missing_values(X_processed)
        elif handle_missing == 'drop':
            X_processed = X_processed.dropna()
        
        # Encode categorical variables
        X_processed = self._encode_categorical_features(X_processed)
        
        # Scale numeric features
        if scale_features:
            X_processed = self._scale_numeric_features(X_processed)
        
        return X_processed
    
    def _impute_missing_values(self, df):
        """Impute missing values using appropriate strategies"""
        df_imputed = df.copy()
        
        # Numeric columns: use median
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            if self.imputer is None:
                self.imputer = SimpleImputer(strategy='median')
                df_imputed[numeric_cols] = self.imputer.fit_transform(df_imputed[numeric_cols])
            else:
                df_imputed[numeric_cols] = self.imputer.transform(df_imputed[numeric_cols])
        
        # Categorical columns: use most frequent
        categorical_cols = df_imputed.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df_imputed[col].isnull().any():
                most_frequent = df_imputed[col].mode().iloc[0] if len(df_imputed[col].mode()) > 0 else 'Unknown'
                df_imputed[col] = df_imputed[col].fillna(most_frequent)
        
        return df_imputed
    
    def _encode_categorical_features(self, df):
        """Encode categorical features as numeric"""
        df_encoded = df.copy()
        
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            # Use label encoding for simplicity
            # In production, consider one-hot encoding for nominal variables
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
            else:
                # Handle unseen categories
                unique_values = df_encoded[col].unique()
                for value in unique_values:
                    if value not in self.label_encoders[col].classes_:
                        # Add unknown category
                        self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, value)
                
                df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def _scale_numeric_features(self, df):
        """Scale numeric features to standardized values"""
        df_scaled = df.copy()
        
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            if self.scaler is None:
                self.scaler = StandardScaler()
                df_scaled[numeric_cols] = self.scaler.fit_transform(df_scaled[numeric_cols])
            else:
                df_scaled[numeric_cols] = self.scaler.transform(df_scaled[numeric_cols])
        
        return df_scaled
    
    def detect_target_column(self, df, user_hint=None):
        """
        Suggest potential target columns based on heuristics
        
        Args:
            df: pandas DataFrame
            user_hint: User provided hint for target column
            
        Returns:
            List of suggested target columns
        """
        suggestions = []
        
        # If user provided hint, prioritize it
        if user_hint and user_hint in df.columns:
            suggestions.append(user_hint)
        
        # Common target column names
        common_targets = [
            'target', 'label', 'class', 'y', 'output', 'prediction',
            'outcome', 'result', 'response', 'dependent'
        ]
        
        for col in df.columns:
            col_lower = col.lower()
            if any(target in col_lower for target in common_targets):
                if col not in suggestions:
                    suggestions.append(col)
        
        # Columns with fewer unique values (potential categorical targets)
        for col in df.columns:
            if df[col].nunique() <= 20 and df[col].nunique() > 1:
                if col not in suggestions:
                    suggestions.append(col)
        
        # If no suggestions, use the last column
        if not suggestions and len(df.columns) > 1:
            suggestions.append(df.columns[-1])
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def split_features_target(self, df, target_col):
        """
        Split dataframe into features and target
        
        Args:
            df: pandas DataFrame
            target_col: Target column name
            
        Returns:
            Tuple of (X, y)
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        return X, y
    
    def get_feature_types(self, df):
        """
        Classify features by type for better visualization
        
        Args:
            df: pandas DataFrame
            
        Returns:
            Dictionary with feature classifications
        """
        feature_types = {
            'numeric_continuous': [],
            'numeric_discrete': [],
            'categorical_nominal': [],
            'categorical_ordinal': [],
            'binary': [],
            'datetime': []
        }
        
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                feature_types['datetime'].append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                unique_count = df[col].nunique()
                if unique_count == 2:
                    feature_types['binary'].append(col)
                elif unique_count <= 10:
                    feature_types['numeric_discrete'].append(col)
                else:
                    feature_types['numeric_continuous'].append(col)
            else:
                unique_count = df[col].nunique()
                if unique_count == 2:
                    feature_types['binary'].append(col)
                elif unique_count <= 10:
                    feature_types['categorical_nominal'].append(col)
                else:
                    feature_types['categorical_nominal'].append(col)
        
        return feature_types

class DataValidator:
    """Utility class for validating data quality and ML readiness"""
    
    @staticmethod
    def check_data_quality(df):
        """
        Comprehensive data quality check
        
        Args:
            df: pandas DataFrame
            
        Returns:
            Dictionary with quality assessment
        """
        quality_report = {
            'overall_score': 0,
            'issues': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Check for missing values
        missing_percentage = (df.isnull().sum() / len(df) * 100)
        high_missing_cols = missing_percentage[missing_percentage > 50].index.tolist()
        
        if high_missing_cols:
            quality_report['issues'].append(f"Columns with >50% missing values: {high_missing_cols}")
        
        moderate_missing_cols = missing_percentage[(missing_percentage > 20) & (missing_percentage <= 50)].index.tolist()
        if moderate_missing_cols:
            quality_report['warnings'].append(f"Columns with 20-50% missing values: {moderate_missing_cols}")
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            duplicate_percentage = (duplicate_count / len(df)) * 100
            if duplicate_percentage > 10:
                quality_report['issues'].append(f"High number of duplicate rows: {duplicate_count} ({duplicate_percentage:.1f}%)")
            else:
                quality_report['warnings'].append(f"Duplicate rows found: {duplicate_count} ({duplicate_percentage:.1f}%)")
        
        # Check for constant columns
        constant_cols = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            quality_report['issues'].append(f"Constant columns (no variance): {constant_cols}")
        
        # Check for high cardinality categorical columns
        high_cardinality_cols = []
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if df[col].nunique() > len(df) * 0.8:  # More than 80% unique values
                high_cardinality_cols.append(col)
        
        if high_cardinality_cols:
            quality_report['warnings'].append(f"High cardinality categorical columns: {high_cardinality_cols}")
        
        # Check data size
        if len(df) < 100:
            quality_report['warnings'].append("Small dataset size (< 100 rows). Results may not be reliable.")
        
        if df.shape[1] > df.shape[0]:
            quality_report['warnings'].append("More features than samples. Consider dimensionality reduction.")
        
        # Calculate overall score
        score = 100
        score -= len(quality_report['issues']) * 20  # Each issue reduces score by 20
        score -= len(quality_report['warnings']) * 10  # Each warning reduces score by 10
        quality_report['overall_score'] = max(0, score)
        
        # Generate suggestions
        if quality_report['overall_score'] < 60:
            quality_report['suggestions'].append("Consider data cleaning before model training")
        
        if missing_percentage.max() > 20:
            quality_report['suggestions'].append("Consider imputation or removal of columns with high missing values")
        
        if duplicate_count > 0:
            quality_report['suggestions'].append("Consider removing duplicate rows")
        
        return quality_report
    
    @staticmethod
    def suggest_preprocessing_steps(df, target_col=None):
        """
        Suggest preprocessing steps based on data characteristics
        
        Args:
            df: pandas DataFrame
            target_col: Target column name
            
        Returns:
            List of suggested preprocessing steps
        """
        suggestions = []
        
        # Missing value handling
        missing_percentage = (df.isnull().sum() / len(df) * 100)
        if missing_percentage.max() > 0:
            suggestions.append({
                'step': 'Handle Missing Values',
                'reason': f"Dataset has missing values (max: {missing_percentage.max():.1f}%)",
                'action': 'Use imputation for numeric columns, mode for categorical columns'
            })
        
        # Categorical encoding
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            suggestions.append({
                'step': 'Encode Categorical Variables',
                'reason': f"Found {len(categorical_cols)} categorical columns",
                'action': 'Use label encoding or one-hot encoding'
            })
        
        # Feature scaling
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if target_col:
            numeric_cols = [col for col in numeric_cols if col != target_col]
        
        if len(numeric_cols) > 1:
            # Check if scaling is needed
            ranges = []
            for col in numeric_cols:
                col_range = df[col].max() - df[col].min()
                ranges.append(col_range)
            
            if max(ranges) / min(ranges) > 100:  # Large difference in scales
                suggestions.append({
                    'step': 'Scale Features',
                    'reason': 'Features have very different scales',
                    'action': 'Use StandardScaler or MinMaxScaler'
                })
        
        # Outlier detection
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            
            if outliers > len(df) * 0.05:  # More than 5% outliers
                suggestions.append({
                    'step': f'Handle Outliers in {col}',
                    'reason': f'Found {outliers} potential outliers ({outliers/len(df)*100:.1f}%)',
                    'action': 'Consider outlier removal or transformation'
                })
                break  # Only suggest once for outliers
        
        return suggestions
