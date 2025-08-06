# Explainable AI Platform Project Overview

This is a Real-Time LLM Neural Pathway Visualizer built with Streamlit that provides interpretability and explainability for Large Language Models. The platform shows users how LLMs process prompts step-by-step, revealing neural pathways, attention patterns, reasoning processes, and which neural networks are activated during inference.

## Recent Updates (August 2025)
- ✓ Implemented comprehensive dual-mode analysis system with utils/dual_mode_analyzer.py
- ✓A Added new "Accurate Metrics Analysis" tab with transparency notices and confidence scoring
- ✓ Resolved transparency issue by clearly labeling simulated vs real analysis results
- ⌓ Created GitHub repository (Ashutosh3142857/explainable-ai-platform)
- ✓I Implemented automatic detection system that chooses appropriate analysis method based on model type and API availability

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

## LLM Analysis Components
- **Dual-Mode System**: Supports both private model prediction and open-source real analysis
- **Private Model Prediction**: Advanced algorithms for OpenAI/Anthropic models using linguistic analysis, semantic complexity modeling, and task-type classification
- **Open-Source Real Analysis**: Direct extraction of attention weights, layer activations, and neural pathways from accessible models
- **Attention Patterns**: 12-layer, 8-head attention matrix generation with realistic token-to-token relationships (prediction mode) or real extraction (open-source mode)
- **Layer Activations**: Progressive complexity modeling through 768-dimensional activation vectors or genuine hidden state extraction
- **Neural Pathways**: Linguistic, semantic, and logical pathway identification and strength measurement with confidence scoring
- **Reasoning Steps**: Four-stage processing simulation (Input↑Context↑Pattern↑Generation)
- **Attribution Analysis**: Real gradient-based attributions for open-source models using Captum integration

## Visualization Features
- **Neural Pathway Visualization**: Critical pathway strength charts and activation flow diagrams
- **Attention Heatmaps**: Interactive token-to-token attention pattern visualization with layer/head selection
- **Layer Analysis**: Progressive activation strength tracking with token-specific analysis options
- **Reasoning Process**: Step-by-step confidence and processing time visualization
- **Side-by-Side Comparison**: Multiple explanation techniques displayed simultaneously for comparative analysis
- **Advanced Visual Debugging**: 6-method debugging platform including ELI5, PyTorch attribution, Circuit Tracing, InterpretML unified platform, Context-Specific Techniques, and comparison views

# External Dependencies

## Core Libraries
- **numpy**: Numerical computing for activation matrices and attention pattern generation
- **pandas**: Data structure management for token sequences and analysis results
- **plotly**: Interactive visualization framework for neural pathway charts and heatmaps
- **scikit-learn**: Machine learning models and utilities for debugging demonstrations
- **lime**: Local interpretable model-agnostic explanations (LIME) for black box model explanations
- **networkx**: Graph analysis and visualization for neural circuit tracing
- **torch/torchvision**: PyTorch deep learning framework for attribution analysis (when available)
