# Overview

This is a Real-Time LLM Neural Pathway Visualizer built with Streamlit that provides interpretability and explainability for Large Language Models. The platform shows users how LLMs process prompts step-by-step, revealing neural pathways, attention patterns, reasoning processes, and which neural networks are activated during inference. Users can input any prompt and watch the model's internal decision-making process unfold in real-time with truly dynamic interactive visualizations that respond to layer and attention head selections.

## Recent Updates (January 2025)
- ✓ Implemented persistent session state management to prevent analysis loss during interactive control changes
- ✓ Added truly dynamic real-time analysis that changes based on layer depth and attention head specialization
- ✓ Enhanced attention patterns with realistic layer-based behaviors (early=syntax, middle=semantics, late=output)
- ✓ Implemented head-specific specializations (positional, semantic, entity recognition, causal reasoning)
- ✓ Added context-aware neural pathway generation that adapts to prompt content
- ✓ Fixed interactive controls to update visualizations without breaking analysis flow
- ✓ Added Advanced Visual Debugging & Attribution tab with comprehensive model interpretability features
- ✓ Implemented ELI5 feature importance visualization for text and tabular data analysis
- ✓ Added SHAP analysis with support for text, tabular, and deep learning models
- ✓ Integrated LIME analysis for local interpretable model explanations (text, tabular, image)
- ✓ Created custom PyTorch attribution methods (Integrated Gradients, Gradient × Input)
- ✓ Built Open Circuit Tracing for neural pathway visualization using NetworkX graphs
- ✓ Added comprehensive comparison view of all attribution methods with selection guide

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Single-page Streamlit application with tabbed interface for different analysis views
- **UI Components**: Custom CSS styling with neural network theme, real-time progress indicators
- **Visualization**: Plotly for interactive neural pathway charts, attention heatmaps, and layer activation graphs
- **Real-time Interface**: Live processing simulation with step-by-step neural network visualization

## Backend Architecture
- **Core Engine**: MockLLMProcessor class simulating transformer model behavior with realistic patterns
- **Neural Simulation**: Multi-layer attention patterns, activation flows, and reasoning process modeling
- **Pathway Analysis**: Critical pathway detection, activation strength tracking, and decision point identification
- **Real-time Processing**: Step-by-step simulation of LLM inference with visual feedback

## LLM Simulation Components
- **Attention Patterns**: 12-layer, 8-head attention matrix generation with realistic token-to-token relationships
- **Layer Activations**: Progressive complexity modeling through 768-dimensional activation vectors
- **Neural Pathways**: Linguistic, semantic, and logical pathway identification and strength measurement
- **Reasoning Steps**: Four-stage processing simulation (Input→Context→Pattern→Generation)

## Visualization Features
- **Neural Pathway Visualization**: Critical pathway strength charts and activation flow diagrams
- **Attention Heatmaps**: Interactive token-to-token attention pattern visualization with layer/head selection
- **Layer Analysis**: Progressive activation strength tracking with token-specific analysis options
- **Reasoning Process**: Step-by-step confidence and processing time visualization
- **Side-by-Side Comparison**: Multiple explanation techniques displayed simultaneously for comparative analysis

## Interactive Analysis
- **Real-time Mode**: Live processing simulation with progress indicators and status updates
- **Advanced View**: Multi-head attention comparison and detailed token-level analysis
- **Comparison Mode**: Side-by-side analysis of different explanation techniques with interactive comparisons
- **Advanced Debugging**: Comprehensive attribution methods including ELI5, SHAP, LIME, PyTorch methods, and circuit tracing
- **Prompt Examples**: Pre-built example prompts for different types of language tasks
- **Dynamic Visualization**: User-selectable layers, attention heads, and tokens for detailed inspection

# External Dependencies

## Core Libraries
- **numpy**: Numerical computing for activation matrices and attention pattern generation
- **pandas**: Data structure management for token sequences and analysis results
- **plotly**: Interactive visualization framework for neural pathway charts and heatmaps

## Web Framework
- **streamlit**: Single-page application framework with real-time processing interface

## Visualization Stack
- **plotly.graph_objects**: Custom neural pathway and attention visualization components
- **plotly.subplots**: Multi-head attention comparison and layer analysis grids
- **matplotlib/seaborn**: Fallback visualization support for static analysis charts

## LLM Simulation Framework
- **MockLLMProcessor**: Custom simulation engine for transformer model behavior
- **Attention Pattern Generator**: Realistic token-to-token attention matrix creation
- **Neural Pathway Tracker**: Critical pathway identification and strength measurement
- **Reasoning Process Simulator**: Step-by-step LLM decision-making visualization

## Analysis Components
- **Real-time Processing**: Live inference simulation with progress tracking
- **Interactive Controls**: Layer/head selection, token analysis, and view mode selection (Standard/Advanced/Comparison)
- **Example Prompt Library**: Pre-configured prompts for different language model tasks
- **Comparison Engine**: Side-by-side analysis of attention patterns, neural pathways, layer activations, and reasoning processes
- **Advanced Attribution Methods**: ELI5, SHAP, LIME, PyTorch Integrated Gradients, Circuit Tracing with comprehensive visualizations
- **Method Selection Guide**: Intelligent recommendations based on model type, priority, and analysis scope
- **Technique Evaluation**: Reliability scoring and use-case analysis for different explanation methods
- **Pathway Interaction Networks**: Visual representation of critical pathway connections and strengths
- **Multi-Modal Analysis**: Support for text, tabular, and image explanation methods