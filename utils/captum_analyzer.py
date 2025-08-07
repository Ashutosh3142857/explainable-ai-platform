"""
Captum Integration for PyTorch Deep Neural Interpretability
Provides gradient-based attribution methods for deep learning models
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Conditional imports for Captum
try:
    from captum.attr import (
        IntegratedGradients, 
        LayerIntegratedGradients,
        GradientShap,
        DeepLift,
        Saliency,
        InputXGradient,
        GuidedBackprop,
        TokenReferenceBase,
        visualization as viz
    )
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False

class CaptumAnalyzer:
    """PyTorch deep neural interpretability using Captum attribution methods"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.attribution_methods = self._initialize_methods()
        
    def _initialize_methods(self) -> Dict[str, Any]:
        """Initialize available attribution methods"""
        methods = {}
        
        if CAPTUM_AVAILABLE:
            methods.update({
                'integrated_gradients': IntegratedGradients,
                'gradient_shap': GradientShap,
                'deep_lift': DeepLift,
                'saliency': Saliency,
                'input_x_gradient': InputXGradient,
                'guided_backprop': GuidedBackprop
            })
        
        return methods
    
    def analyze_model(self, model: nn.Module, inputs: torch.Tensor, 
                     target_class: Optional[int] = None,
                     method: str = 'integrated_gradients') -> Dict[str, Any]:
        """
        Perform attribution analysis on a PyTorch model
        
        Args:
            model: PyTorch neural network model
            inputs: Input tensor to analyze
            target_class: Target class for attribution (optional)
            method: Attribution method to use
            
        Returns:
            Dictionary containing attribution results and visualizations
        """
        
        if not CAPTUM_AVAILABLE:
            return self._mock_captum_analysis(inputs, method)
        
        model.eval()
        inputs.requires_grad_(True)
        
        # Initialize attribution method
        if method not in self.attribution_methods:
            method = 'integrated_gradients'
            
        attributor = self.attribution_methods[method](model)
        
        # Compute attributions
        if target_class is None:
            # Get predicted class
            with torch.no_grad():
                predictions = model(inputs)
                target_class = predictions.argmax(dim=-1).item()
        
        attributions = attributor.attribute(inputs, target=target_class)
        
        # Process results
        results = {
            'method': method,
            'target_class': target_class,
            'attributions': attributions.detach().cpu().numpy(),
            'input_shape': list(inputs.shape),
            'attribution_shape': list(attributions.shape),
            'attribution_summary': self._summarize_attributions(attributions),
            'visualization_data': self._prepare_visualization_data(
                inputs, attributions, method
            )
        }
        
        return results
    
    def layer_attribution_analysis(self, model: nn.Module, inputs: torch.Tensor,
                                 layer_name: str, target_class: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform layer-wise attribution analysis
        
        Args:
            model: PyTorch neural network model
            inputs: Input tensor to analyze
            layer_name: Name of layer to analyze
            target_class: Target class for attribution
            
        Returns:
            Layer attribution results
        """
        
        if not CAPTUM_AVAILABLE:
            return self._mock_layer_analysis(inputs, layer_name)
        
        model.eval()
        
        # Get layer by name
        layer = dict(model.named_modules())[layer_name]
        
        # Initialize layer attribution
        layer_attributor = LayerIntegratedGradients(model, layer)
        
        if target_class is None:
            with torch.no_grad():
                predictions = model(inputs)
                target_class = predictions.argmax(dim=-1).item()
        
        layer_attributions = layer_attributor.attribute(inputs, target=target_class)
        
        results = {
            'layer_name': layer_name,
            'target_class': target_class,
            'layer_attributions': layer_attributions.detach().cpu().numpy(),
            'attribution_shape': list(layer_attributions.shape),
            'layer_summary': self._summarize_layer_attributions(layer_attributions)
        }
        
        return results
    
    def multi_method_comparison(self, model: nn.Module, inputs: torch.Tensor,
                               target_class: Optional[int] = None) -> Dict[str, Any]:
        """
        Compare multiple attribution methods on the same input
        
        Args:
            model: PyTorch neural network model
            inputs: Input tensor to analyze
            target_class: Target class for attribution
            
        Returns:
            Comparison results across methods
        """
        
        if not CAPTUM_AVAILABLE:
            return self._mock_comparison_analysis(inputs)
        
        methods_to_compare = ['integrated_gradients', 'gradient_shap', 'saliency', 'deep_lift']
        results = {}
        
        for method in methods_to_compare:
            if method in self.attribution_methods:
                try:
                    method_results = self.analyze_model(model, inputs, target_class, method)
                    results[method] = {
                        'attributions': method_results['attributions'],
                        'summary': method_results['attribution_summary']
                    }
                except Exception as e:
                    results[method] = {'error': str(e)}
        
        # Calculate correlation between methods
        correlations = self._calculate_method_correlations(results)
        
        return {
            'method_results': results,
            'correlations': correlations,
            'comparison_summary': self._generate_comparison_summary(results)
        }
    
    def _summarize_attributions(self, attributions: torch.Tensor) -> Dict[str, float]:
        """Summarize attribution statistics"""
        attr_np = attributions.detach().cpu().numpy()
        
        return {
            'mean': float(np.mean(attr_np)),
            'std': float(np.std(attr_np)),
            'min': float(np.min(attr_np)),
            'max': float(np.max(attr_np)),
            'abs_mean': float(np.mean(np.abs(attr_np))),
            'positive_ratio': float(np.mean(attr_np > 0)),
            'top_10_percent_mean': float(np.mean(np.sort(np.abs(attr_np.flatten()))[-int(0.1 * attr_np.size):]))
        }
    
    def _summarize_layer_attributions(self, layer_attributions: torch.Tensor) -> Dict[str, Any]:
        """Summarize layer attribution statistics"""
        layer_np = layer_attributions.detach().cpu().numpy()
        
        return {
            'shape': list(layer_attributions.shape),
            'mean_activation': float(np.mean(layer_np)),
            'std_activation': float(np.std(layer_np)),
            'active_neurons': int(np.sum(np.abs(layer_np) > 0.01)),
            'top_neurons': np.argsort(np.abs(layer_np.flatten()))[-10:].tolist()
        }
    
    def _prepare_visualization_data(self, inputs: torch.Tensor, 
                                  attributions: torch.Tensor, method: str) -> Dict[str, Any]:
        """Prepare data for visualization"""
        
        inputs_np = inputs.detach().cpu().numpy()
        attr_np = attributions.detach().cpu().numpy()
        
        # Normalize attributions for visualization
        attr_normalized = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)
        
        return {
            'input_data': inputs_np.tolist(),
            'attribution_data': attr_np.tolist(),
            'attribution_normalized': attr_normalized.tolist(),
            'heatmap_data': self._create_heatmap_data(attr_np),
            'method': method
        }
    
    def _create_heatmap_data(self, attributions: np.ndarray) -> List[List[float]]:
        """Create heatmap data from attributions"""
        
        # For 1D data, create a simple heatmap
        if len(attributions.shape) == 1:
            return [attributions.tolist()]
        
        # For 2D data, use as-is
        elif len(attributions.shape) == 2:
            return attributions.tolist()
        
        # For higher dimensional data, flatten and reshape
        else:
            flattened = attributions.flatten()
            size = int(np.sqrt(len(flattened)))
            if size * size == len(flattened):
                return flattened.reshape(size, size).tolist()
            else:
                # Create a reasonable 2D representation
                rows = min(20, len(flattened))
                cols = len(flattened) // rows
                return flattened[:rows*cols].reshape(rows, cols).tolist()
    
    def _calculate_method_correlations(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate correlations between attribution methods"""
        
        correlations = {}
        methods = list(results.keys())
        
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                if 'attributions' in results[method1] and 'attributions' in results[method2]:
                    attr1 = np.array(results[method1]['attributions']).flatten()
                    attr2 = np.array(results[method2]['attributions']).flatten()
                    
                    # Calculate Pearson correlation
                    correlation = np.corrcoef(attr1, attr2)[0, 1]
                    correlations[f"{method1}_vs_{method2}"] = float(correlation)
        
        return correlations
    
    def _generate_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of method comparison"""
        
        valid_methods = [m for m in results.keys() if 'attributions' in results[m]]
        
        if not valid_methods:
            return {'error': 'No valid attribution results'}
        
        # Calculate agreement score (average correlation)
        all_correlations = []
        for i, method1 in enumerate(valid_methods):
            for method2 in valid_methods[i+1:]:
                attr1 = np.array(results[method1]['attributions']).flatten()
                attr2 = np.array(results[method2]['attributions']).flatten()
                corr = np.corrcoef(attr1, attr2)[0, 1]
                if not np.isnan(corr):
                    all_correlations.append(corr)
        
        agreement_score = np.mean(all_correlations) if all_correlations else 0.0
        
        return {
            'valid_methods': valid_methods,
            'agreement_score': float(agreement_score),
            'recommendation': self._get_method_recommendation(results, agreement_score)
        }
    
    def _get_method_recommendation(self, results: Dict[str, Any], agreement_score: float) -> str:
        """Get recommendation based on analysis results"""
        
        if agreement_score > 0.8:
            return "High agreement between methods. Results are reliable."
        elif agreement_score > 0.5:
            return "Moderate agreement. Consider using Integrated Gradients for most reliable results."
        else:
            return "Low agreement between methods. Results may be sensitive to method choice."
    
    # Mock implementations for when Captum is not available
    def _mock_captum_analysis(self, inputs: torch.Tensor, method: str) -> Dict[str, Any]:
        """Mock analysis when Captum is not available"""
        
        # Generate realistic mock attributions
        mock_attributions = torch.randn_like(inputs) * 0.1
        
        return {
            'method': method,
            'target_class': 1,
            'attributions': mock_attributions.numpy(),
            'input_shape': list(inputs.shape),
            'attribution_shape': list(mock_attributions.shape),
            'attribution_summary': {
                'mean': 0.05,
                'std': 0.1,
                'min': -0.3,
                'max': 0.3,
                'abs_mean': 0.08,
                'positive_ratio': 0.52
            },
            'visualization_data': {
                'method': method,
                'heatmap_data': [[0.1, 0.2], [0.3, 0.4]],
                'attribution_normalized': mock_attributions.numpy().tolist()
            },
            'note': 'Mock data - Captum not available'
        }
    
    def _mock_layer_analysis(self, inputs: torch.Tensor, layer_name: str) -> Dict[str, Any]:
        """Mock layer analysis"""
        
        return {
            'layer_name': layer_name,
            'target_class': 1,
            'layer_attributions': np.random.randn(10, 20).tolist(),
            'attribution_shape': [10, 20],
            'layer_summary': {
                'shape': [10, 20],
                'mean_activation': 0.12,
                'std_activation': 0.34,
                'active_neurons': 150,
                'top_neurons': [45, 67, 89, 123, 156, 178, 190, 195, 197, 199]
            },
            'note': 'Mock data - Captum not available'
        }
    
    def _mock_comparison_analysis(self, inputs: torch.Tensor) -> Dict[str, Any]:
        """Mock comparison analysis"""
        
        return {
            'method_results': {
                'integrated_gradients': {
                    'attributions': np.random.randn(*inputs.shape).tolist(),
                    'summary': {'mean': 0.05, 'std': 0.1}
                },
                'gradient_shap': {
                    'attributions': np.random.randn(*inputs.shape).tolist(),
                    'summary': {'mean': 0.04, 'std': 0.12}
                }
            },
            'correlations': {'integrated_gradients_vs_gradient_shap': 0.75},
            'comparison_summary': {
                'valid_methods': ['integrated_gradients', 'gradient_shap'],
                'agreement_score': 0.75,
                'recommendation': 'Moderate agreement. Consider using Integrated Gradients for most reliable results.'
            },
            'note': 'Mock data - Captum not available'
        }

def create_simple_model() -> nn.Module:
    """Create a simple model for demonstration purposes"""
    
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear1 = nn.Linear(784, 128)
            self.linear2 = nn.Linear(128, 64)
            self.linear3 = nn.Linear(64, 10)
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x):
            x = self.flatten(x)
            x = F.relu(self.linear1(x))
            x = self.dropout(x)
            x = F.relu(self.linear2(x))
            x = self.linear3(x)
            return F.log_softmax(x, dim=1)
    
    return SimpleNet()

def get_available_attribution_methods() -> List[str]:
    """Get list of available attribution methods"""
    
    if CAPTUM_AVAILABLE:
        return [
            'integrated_gradients',
            'gradient_shap', 
            'deep_lift',
            'saliency',
            'input_x_gradient',
            'guided_backprop'
        ]
    else:
        return ['mock_integrated_gradients', 'mock_gradient_shap']