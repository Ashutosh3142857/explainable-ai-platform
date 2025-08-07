"""
Anthropic Circuit Tracing for LLMs
Implements circuit analysis and pathway tracing for understanding 
internal computations in large language models
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Set
import json
import re
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class CircuitNode:
    """Represents a node in the neural circuit"""
    node_id: str
    node_type: str  # 'attention_head', 'mlp_neuron', 'layer_norm', 'residual'
    layer: int
    position: Optional[int] = None  # For attention heads
    activation_strength: float = 0.0
    connections: List[str] = None
    
    def __post_init__(self):
        if self.connections is None:
            self.connections = []

@dataclass
class CircuitEdge:
    """Represents a connection between circuit nodes"""
    source: str
    target: str
    weight: float
    edge_type: str  # 'attention', 'mlp', 'residual', 'skip'
    pathway_name: Optional[str] = None

class AnthropicCircuitTracer:
    """
    Circuit tracing for LLMs inspired by Anthropic's mechanistic interpretability work
    
    This implementation traces computational pathways through transformer architectures,
    identifying critical circuits for specific behaviors and tasks.
    """
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize circuit tracer
        
        Args:
            model_config: Configuration for the model being analyzed
        """
        self.model_config = model_config or self._default_model_config()
        self.circuit_graph = nx.DiGraph()
        self.discovered_circuits = {}
        self.pathway_analysis = {}
        
    def _default_model_config(self) -> Dict[str, Any]:
        """Default transformer model configuration"""
        return {
            'num_layers': 12,
            'num_attention_heads': 8,
            'hidden_size': 768,
            'intermediate_size': 3072,
            'vocab_size': 50257
        }
    
    def trace_circuit(self, prompt: str, target_behavior: str, 
                     method: str = 'activation_patching') -> Dict[str, Any]:
        """
        Trace circuits responsible for specific behavior
        
        Args:
            prompt: Input prompt to analyze
            target_behavior: Specific behavior to trace (e.g., 'factual_recall', 'reasoning')
            method: Circuit tracing method to use
            
        Returns:
            Circuit analysis results
        """
        
        # Tokenize and prepare input
        tokens = self._tokenize_prompt(prompt)
        
        # Initialize circuit graph
        self._build_circuit_graph(tokens)
        
        # Perform circuit tracing
        if method == 'activation_patching':
            circuit_results = self._activation_patching_analysis(tokens, target_behavior)
        elif method == 'causal_tracing':
            circuit_results = self._causal_tracing_analysis(tokens, target_behavior)
        elif method == 'attention_pattern_analysis':
            circuit_results = self._attention_pattern_analysis(tokens, target_behavior)
        else:
            circuit_results = self._comprehensive_circuit_analysis(tokens, target_behavior)
        
        # Identify critical pathways
        critical_pathways = self._identify_critical_pathways(circuit_results)
        
        # Generate circuit visualization data
        visualization_data = self._prepare_circuit_visualization(critical_pathways)
        
        return {
            'prompt': prompt,
            'target_behavior': target_behavior,
            'method': method,
            'tokens': tokens,
            'circuit_results': circuit_results,
            'critical_pathways': critical_pathways,
            'visualization_data': visualization_data,
            'circuit_summary': self._generate_circuit_summary(critical_pathways),
            'interpretability_insights': self._generate_interpretability_insights(
                circuit_results, critical_pathways
            )
        }
    
    def _tokenize_prompt(self, prompt: str) -> List[str]:
        """Simple tokenization (would use actual tokenizer in practice)"""
        # Basic word tokenization for demonstration
        tokens = prompt.lower().split()
        # Add special tokens
        return ['<bos>'] + tokens + ['<eos>']
    
    def _build_circuit_graph(self, tokens: List[str]) -> None:
        """Build the circuit graph structure"""
        
        self.circuit_graph.clear()
        num_layers = self.model_config['num_layers']
        num_heads = self.model_config['num_attention_heads']
        
        # Add nodes for each layer component
        for layer in range(num_layers):
            # Attention heads
            for head in range(num_heads):
                node_id = f"attn_L{layer}_H{head}"
                node = CircuitNode(
                    node_id=node_id,
                    node_type='attention_head',
                    layer=layer,
                    position=head,
                    activation_strength=np.random.random()  # Would be actual activations
                )
                self.circuit_graph.add_node(node_id, **node.__dict__)
            
            # MLP neurons (representative subset)
            for neuron in range(0, 100, 10):  # Sample every 10th neuron
                node_id = f"mlp_L{layer}_N{neuron}"
                node = CircuitNode(
                    node_id=node_id,
                    node_type='mlp_neuron',
                    layer=layer,
                    position=neuron,
                    activation_strength=np.random.random()
                )
                self.circuit_graph.add_node(node_id, **node.__dict__)
            
            # Layer norm
            node_id = f"layernorm_L{layer}"
            node = CircuitNode(
                node_id=node_id,
                node_type='layer_norm',
                layer=layer,
                activation_strength=np.random.random()
            )
            self.circuit_graph.add_node(node_id, **node.__dict__)
        
        # Add edges (connections)
        self._add_circuit_connections()
    
    def _add_circuit_connections(self) -> None:
        """Add connections between circuit components"""
        
        num_layers = self.model_config['num_layers']
        num_heads = self.model_config['num_attention_heads']
        
        for layer in range(num_layers - 1):
            # Attention to MLP connections
            for head in range(num_heads):
                attn_node = f"attn_L{layer}_H{head}"
                for neuron in range(0, 100, 10):
                    mlp_node = f"mlp_L{layer}_N{neuron}"
                    
                    # Add connection with random weight (would be learned/measured)
                    weight = np.random.random() * 0.5
                    if weight > 0.2:  # Only add significant connections
                        edge = CircuitEdge(
                            source=attn_node,
                            target=mlp_node,
                            weight=weight,
                            edge_type='attention_to_mlp'
                        )
                        self.circuit_graph.add_edge(
                            attn_node, mlp_node, **edge.__dict__
                        )
            
            # Layer to layer connections
            for head in range(num_heads):
                current_attn = f"attn_L{layer}_H{head}"
                next_attn = f"attn_L{layer+1}_H{head}"
                
                weight = np.random.random() * 0.3
                if weight > 0.1:
                    edge = CircuitEdge(
                        source=current_attn,
                        target=next_attn,
                        weight=weight,
                        edge_type='layer_to_layer'
                    )
                    self.circuit_graph.add_edge(
                        current_attn, next_attn, **edge.__dict__
                    )
    
    def _activation_patching_analysis(self, tokens: List[str], 
                                    target_behavior: str) -> Dict[str, Any]:
        """
        Perform activation patching to identify circuit components
        
        This simulates the process of systematically patching activations
        to identify which components are critical for the target behavior.
        """
        
        # Simulate activation patching results
        critical_components = []
        patching_results = {}
        
        for node in self.circuit_graph.nodes():
            # Simulate patching this component
            impact_score = self._simulate_patching_impact(node, target_behavior)
            
            patching_results[node] = {
                'impact_score': impact_score,
                'component_type': self.circuit_graph.nodes[node]['node_type'],
                'layer': self.circuit_graph.nodes[node]['layer']
            }
            
            if impact_score > 0.7:  # High impact threshold
                critical_components.append(node)
        
        return {
            'method': 'activation_patching',
            'critical_components': critical_components,
            'patching_results': patching_results,
            'impact_distribution': self._analyze_impact_distribution(patching_results)
        }
    
    def _simulate_patching_impact(self, component: str, target_behavior: str) -> float:
        """Simulate the impact of patching a specific component"""
        
        # Get component info
        node_data = self.circuit_graph.nodes[component]
        node_type = node_data['node_type']
        layer = node_data['layer']
        
        # Simulate behavior-specific impact patterns
        if target_behavior == 'factual_recall':
            # Later layers and specific attention heads more important
            if node_type == 'attention_head' and layer > 8:
                return min(1.0, np.random.random() + 0.5)
            elif node_type == 'mlp_neuron' and layer > 6:
                return min(1.0, np.random.random() + 0.3)
        
        elif target_behavior == 'reasoning':
            # Mid-layer MLPs and specific attention patterns
            if node_type == 'mlp_neuron' and 4 <= layer <= 8:
                return min(1.0, np.random.random() + 0.4)
            elif node_type == 'attention_head' and layer in [3, 6, 9]:
                return min(1.0, np.random.random() + 0.6)
        
        elif target_behavior == 'syntax_processing':
            # Earlier layers more important
            if layer <= 4:
                return min(1.0, np.random.random() + 0.5)
        
        # Base random impact
        return np.random.random() * 0.4
    
    def _causal_tracing_analysis(self, tokens: List[str], 
                               target_behavior: str) -> Dict[str, Any]:
        """Perform causal tracing to identify information flow"""
        
        # Trace information flow through the model
        information_flow = self._trace_information_flow(tokens)
        
        # Identify causal pathways
        causal_pathways = self._identify_causal_pathways(information_flow, target_behavior)
        
        return {
            'method': 'causal_tracing',
            'information_flow': information_flow,
            'causal_pathways': causal_pathways,
            'pathway_strengths': self._calculate_pathway_strengths(causal_pathways)
        }
    
    def _attention_pattern_analysis(self, tokens: List[str], 
                                  target_behavior: str) -> Dict[str, Any]:
        """Analyze attention patterns to understand circuit behavior"""
        
        attention_patterns = {}
        
        # Analyze patterns for each layer and head
        for layer in range(self.model_config['num_layers']):
            layer_patterns = {}
            
            for head in range(self.model_config['num_attention_heads']):
                # Simulate attention pattern analysis
                pattern = self._simulate_attention_pattern(tokens, layer, head, target_behavior)
                layer_patterns[f"head_{head}"] = pattern
            
            attention_patterns[f"layer_{layer}"] = layer_patterns
        
        # Identify specialized attention heads
        specialized_heads = self._identify_specialized_heads(attention_patterns, target_behavior)
        
        return {
            'method': 'attention_pattern_analysis',
            'attention_patterns': attention_patterns,
            'specialized_heads': specialized_heads,
            'pattern_summary': self._summarize_attention_patterns(attention_patterns)
        }
    
    def _simulate_attention_pattern(self, tokens: List[str], layer: int, 
                                  head: int, target_behavior: str) -> Dict[str, Any]:
        """Simulate attention pattern for a specific head"""
        
        # Create attention matrix
        seq_len = len(tokens)
        attention_matrix = np.random.random((seq_len, seq_len))
        
        # Apply behavior-specific patterns
        if target_behavior == 'factual_recall':
            # Strong attention to subject tokens
            for i, token in enumerate(tokens):
                if token in ['who', 'what', 'where', 'when']:
                    attention_matrix[i, :] *= 2.0
        
        elif target_behavior == 'reasoning':
            # Attention to logical connectors
            for i, token in enumerate(tokens):
                if token in ['because', 'therefore', 'since', 'if']:
                    attention_matrix[:, i] *= 1.5
        
        # Normalize
        attention_matrix = attention_matrix / attention_matrix.sum(axis=-1, keepdims=True)
        
        return {
            'layer': layer,
            'head': head,
            'attention_matrix': attention_matrix.tolist(),
            'entropy': float(-np.sum(attention_matrix * np.log(attention_matrix + 1e-8))),
            'max_attention': float(np.max(attention_matrix)),
            'specialization_score': self._calculate_specialization_score(attention_matrix, target_behavior)
        }
    
    def _comprehensive_circuit_analysis(self, tokens: List[str], 
                                      target_behavior: str) -> Dict[str, Any]:
        """Comprehensive analysis combining multiple methods"""
        
        # Run all analysis methods
        activation_results = self._activation_patching_analysis(tokens, target_behavior)
        causal_results = self._causal_tracing_analysis(tokens, target_behavior)
        attention_results = self._attention_pattern_analysis(tokens, target_behavior)
        
        # Combine results
        combined_analysis = {
            'activation_patching': activation_results,
            'causal_tracing': causal_results,
            'attention_analysis': attention_results,
            'consensus_components': self._find_consensus_components(
                activation_results, causal_results, attention_results
            )
        }
        
        return combined_analysis
    
    def _identify_critical_pathways(self, circuit_results: Dict[str, Any]) -> Dict[str, Any]:
        """Identify the most critical computational pathways"""
        
        pathways = {}
        
        if 'critical_components' in circuit_results:
            # Build pathways from critical components
            critical_components = circuit_results['critical_components']
            
            # Group by pathway type
            pathways['attention_pathway'] = [
                comp for comp in critical_components 
                if 'attn' in comp
            ]
            pathways['mlp_pathway'] = [
                comp for comp in critical_components 
                if 'mlp' in comp
            ]
            
        else:
            # For comprehensive analysis, extract from multiple sources
            pathways = self._extract_pathways_from_comprehensive_analysis(circuit_results)
        
        # Calculate pathway statistics
        pathway_stats = {}
        for pathway_name, components in pathways.items():
            pathway_stats[pathway_name] = {
                'num_components': len(components),
                'layer_distribution': self._analyze_layer_distribution(components),
                'pathway_strength': self._calculate_pathway_strength(components)
            }
        
        return {
            'pathways': pathways,
            'pathway_statistics': pathway_stats,
            'pathway_interactions': self._analyze_pathway_interactions(pathways)
        }
    
    def _prepare_circuit_visualization(self, critical_pathways: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for circuit visualization"""
        
        # Create visualization nodes and edges
        vis_nodes = []
        vis_edges = []
        
        # Process pathways for visualization
        all_components = set()
        for pathway_name, components in critical_pathways['pathways'].items():
            all_components.update(components)
        
        # Create nodes
        for component in all_components:
            if component in self.circuit_graph.nodes:
                node_data = self.circuit_graph.nodes[component]
                vis_nodes.append({
                    'id': component,
                    'label': self._format_component_label(component),
                    'type': node_data['node_type'],
                    'layer': node_data['layer'],
                    'activation': node_data['activation_strength'],
                    'size': min(50, max(10, node_data['activation_strength'] * 40))
                })
        
        # Create edges
        for component in all_components:
            if component in self.circuit_graph:
                for neighbor in self.circuit_graph.neighbors(component):
                    if neighbor in all_components:
                        edge_data = self.circuit_graph.edges[component, neighbor]
                        vis_edges.append({
                            'source': component,
                            'target': neighbor,
                            'weight': edge_data['weight'],
                            'type': edge_data['edge_type'],
                            'width': max(1, edge_data['weight'] * 5)
                        })
        
        # Create layout positions
        layout_positions = self._calculate_layout_positions(vis_nodes)
        
        return {
            'nodes': vis_nodes,
            'edges': vis_edges,
            'layout_positions': layout_positions,
            'circuit_metrics': self._calculate_circuit_metrics(vis_nodes, vis_edges)
        }
    
    def _format_component_label(self, component: str) -> str:
        """Format component name for display"""
        if 'attn' in component:
            parts = component.split('_')
            return f"Attn {parts[1]} {parts[2]}"
        elif 'mlp' in component:
            parts = component.split('_')
            return f"MLP {parts[1]} {parts[2]}"
        else:
            return component.replace('_', ' ').title()
    
    def _calculate_layout_positions(self, nodes: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
        """Calculate positions for circuit visualization"""
        
        positions = {}
        
        # Group nodes by layer
        layer_groups = {}
        for node in nodes:
            layer = node['layer']
            if layer not in layer_groups:
                layer_groups[layer] = []
            layer_groups[layer].append(node)
        
        # Position nodes layer by layer
        for layer, layer_nodes in layer_groups.items():
            y_pos = layer * 100  # Vertical spacing
            
            # Horizontal spacing within layer
            num_nodes = len(layer_nodes)
            for i, node in enumerate(layer_nodes):
                x_pos = (i - num_nodes/2) * 80
                positions[node['id']] = (x_pos, y_pos)
        
        return positions
    
    def _generate_circuit_summary(self, critical_pathways: Dict[str, Any]) -> Dict[str, Any]:
        """Generate human-readable summary of circuit analysis"""
        
        pathways = critical_pathways['pathways']
        stats = critical_pathways['pathway_statistics']
        
        summary = {
            'num_pathways': len(pathways),
            'total_components': sum(stats[p]['num_components'] for p in stats),
            'dominant_pathway': max(stats.keys(), key=lambda x: stats[x]['pathway_strength']),
            'layer_span': self._calculate_layer_span(pathways),
            'interpretability_score': self._calculate_interpretability_score(critical_pathways)
        }
        
        return summary
    
    def _generate_interpretability_insights(self, circuit_results: Dict[str, Any], 
                                          critical_pathways: Dict[str, Any]) -> List[str]:
        """Generate insights about model interpretability"""
        
        insights = []
        
        # Pathway-based insights
        pathways = critical_pathways['pathways']
        
        if 'attention_pathway' in pathways and len(pathways['attention_pathway']) > 3:
            insights.append(
                "Strong attention-based processing detected. "
                "Model relies heavily on attention mechanisms for this behavior."
            )
        
        if 'mlp_pathway' in pathways and len(pathways['mlp_pathway']) > 5:
            insights.append(
                "Significant MLP involvement suggests complex feature composition "
                "and non-linear processing."
            )
        
        # Layer distribution insights
        stats = critical_pathways['pathway_statistics']
        for pathway_name, pathway_stats in stats.items():
            layer_dist = pathway_stats['layer_distribution']
            if max(layer_dist.values()) > 0.6:  # Concentrated in specific layers
                concentrated_layer = max(layer_dist.keys(), key=lambda x: layer_dist[x])
                insights.append(
                    f"Critical processing for {pathway_name} concentrated in layer {concentrated_layer}. "
                    "This suggests specialized function localization."
                )
        
        # Circuit complexity insights
        summary = self._generate_circuit_summary(critical_pathways)
        if summary['interpretability_score'] > 0.8:
            insights.append(
                "High interpretability score indicates clear, localized circuits. "
                "Model behavior is well-understood and predictable."
            )
        elif summary['interpretability_score'] < 0.4:
            insights.append(
                "Low interpretability score suggests distributed, complex processing. "
                "Model behavior may be difficult to predict or modify."
            )
        
        return insights
    
    # Helper methods for circuit analysis
    def _trace_information_flow(self, tokens: List[str]) -> Dict[str, Any]:
        """Trace how information flows through the circuit"""
        # Implementation for information flow tracing
        return {'flow_patterns': {}, 'bottlenecks': [], 'information_paths': []}
    
    def _identify_causal_pathways(self, flow_data: Dict[str, Any], 
                                target_behavior: str) -> List[str]:
        """Identify causal pathways from flow data"""
        return ['pathway1', 'pathway2']  # Simplified
    
    def _calculate_pathway_strengths(self, pathways: List[str]) -> Dict[str, float]:
        """Calculate strength of each pathway"""
        return {p: np.random.random() for p in pathways}
    
    def _identify_specialized_heads(self, patterns: Dict[str, Any], 
                                  target_behavior: str) -> List[str]:
        """Identify attention heads specialized for specific behaviors"""
        specialized = []
        for layer_name, layer_data in patterns.items():
            for head_name, head_data in layer_data.items():
                if head_data['specialization_score'] > 0.7:
                    specialized.append(f"{layer_name}_{head_name}")
        return specialized
    
    def _calculate_specialization_score(self, attention_matrix: np.ndarray, 
                                      target_behavior: str) -> float:
        """Calculate how specialized an attention head is for a behavior"""
        # Simplified specialization calculation
        entropy = -np.sum(attention_matrix * np.log(attention_matrix + 1e-8))
        return max(0, min(1, (5 - entropy) / 3))  # Normalize entropy to [0,1]
    
    def _analyze_impact_distribution(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze distribution of impact scores"""
        scores = [r['impact_score'] for r in results.values()]
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'high_impact_components': len([s for s in scores if s > 0.7])
        }
    
    def _summarize_attention_patterns(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize attention patterns across layers"""
        return {
            'num_layers': len(patterns),
            'avg_entropy': np.mean([
                h['entropy'] for layer in patterns.values() 
                for h in layer.values()
            ]),
            'specialization_distribution': [
                h['specialization_score'] for layer in patterns.values() 
                for h in layer.values()
            ]
        }
    
    def _find_consensus_components(self, *analyses) -> List[str]:
        """Find components identified as important across multiple analyses"""
        # Simplified consensus finding
        all_components = set()
        for analysis in analyses:
            if 'critical_components' in analysis:
                all_components.update(analysis['critical_components'])
        return list(all_components)
    
    def _extract_pathways_from_comprehensive_analysis(self, results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract pathways from comprehensive analysis results"""
        return {'combined_pathway': []}  # Simplified
    
    def _analyze_layer_distribution(self, components: List[str]) -> Dict[int, float]:
        """Analyze distribution of components across layers"""
        layer_counts = {}
        for comp in components:
            if comp in self.circuit_graph.nodes:
                layer = self.circuit_graph.nodes[comp]['layer']
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
        
        total = len(components)
        return {layer: count/total for layer, count in layer_counts.items()}
    
    def _calculate_pathway_strength(self, components: List[str]) -> float:
        """Calculate overall strength of a pathway"""
        if not components:
            return 0.0
        
        strengths = []
        for comp in components:
            if comp in self.circuit_graph.nodes:
                strengths.append(self.circuit_graph.nodes[comp]['activation_strength'])
        
        return np.mean(strengths) if strengths else 0.0
    
    def _analyze_pathway_interactions(self, pathways: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze interactions between different pathways"""
        interactions = {}
        pathway_names = list(pathways.keys())
        
        for i, pathway1 in enumerate(pathway_names):
            for pathway2 in pathway_names[i+1:]:
                # Calculate overlap
                set1 = set(pathways[pathway1])
                set2 = set(pathways[pathway2])
                overlap = len(set1.intersection(set2))
                
                interactions[f"{pathway1}_x_{pathway2}"] = {
                    'overlap': overlap,
                    'jaccard_similarity': overlap / len(set1.union(set2)) if set1.union(set2) else 0
                }
        
        return interactions
    
    def _calculate_circuit_metrics(self, nodes: List[Dict[str, Any]], 
                                 edges: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics for the circuit"""
        return {
            'num_nodes': len(nodes),
            'num_edges': len(edges),
            'avg_connectivity': len(edges) / len(nodes) if nodes else 0,
            'total_activation': sum(node['activation'] for node in nodes)
        }
    
    def _calculate_layer_span(self, pathways: Dict[str, List[str]]) -> Tuple[int, int]:
        """Calculate the span of layers involved in critical pathways"""
        all_layers = []
        for components in pathways.values():
            for comp in components:
                if comp in self.circuit_graph.nodes:
                    all_layers.append(self.circuit_graph.nodes[comp]['layer'])
        
        return (min(all_layers), max(all_layers)) if all_layers else (0, 0)
    
    def _calculate_interpretability_score(self, critical_pathways: Dict[str, Any]) -> float:
        """Calculate overall interpretability score"""
        # Based on pathway clarity, concentration, and consistency
        pathways = critical_pathways['pathways']
        stats = critical_pathways['pathway_statistics']
        
        if not pathways:
            return 0.0
        
        # Factors contributing to interpretability
        num_pathways = len(pathways)
        avg_pathway_strength = np.mean([
            stats[p]['pathway_strength'] for p in stats
        ])
        
        # Simpler circuits are more interpretable
        complexity_penalty = min(1.0, num_pathways / 5)
        
        score = avg_pathway_strength * (1 - complexity_penalty * 0.3)
        return max(0.0, min(1.0, score))

def get_available_circuit_methods() -> List[str]:
    """Get available circuit tracing methods"""
    return [
        'activation_patching',
        'causal_tracing', 
        'attention_pattern_analysis',
        'comprehensive_analysis'
    ]

def get_supported_behaviors() -> List[str]:
    """Get list of supported target behaviors for analysis"""
    return [
        'factual_recall',
        'reasoning',
        'syntax_processing',
        'semantic_understanding',
        'creative_generation',
        'logical_inference'
    ]