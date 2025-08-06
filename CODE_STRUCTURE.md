# Source Code Structure

## Main Application (app.py)

```python
# Streamlit application with four main tabs:
# 1. Standard Neural Pathway Analysis
# 2. Advanced Visual Debugging & Attribution Methods
# 3. Comparison Mode
# 4. Accurate Metrics Analysis
```

## Utility Modules

### dual_mode_analyzer.py
Comprehensive analyzer supporting both private and open-source models.

- **PrivateLLMPredictor**: Advanced metric prediction for private models
- **OpenSourceAnalyzer**: Genuine real-time analysis for open-source models
- **AttentionPredictor**: Linguistic analysis-based attention pattern prediction
- **ComplexityAnalyzer**: Semantic complexity modeling for layer activations

### simple_openai.py
Simplified OpenAI API integration for LLM responses.

### explanation_utils.py
Helper functions for explanation generation and visualization.

### model_handler.py
Model loading and management for both open-source and private models.

## Feature Highlights

### Dual-Mode Analysis
- Automatic detection of model type (private vs open-source)
- Advanced linguistic analysis for private model metric prediction
- Real attention weight extraction for open-source models
- Transparency notices and confidence scoring

### Advanced Debugging
1. **ELI5 Feature Importance**: Text and tabular data analysis
2. **PyTorch Attribution**: Gradient-based attribution methods
3. **Open Circuit Tracing**: Neural pathway visualization with NetworkX
4. **InterpretML Unified Platform**: Glass box models and black box explainers
5. **Context-Specific Techniques**: Computer Vision and NLP frameworks
6. **Method Comparison**: Reliability scoring and evaluation

### Interactive Visualization
- Real-time attention heatmaps with layer/head selection
- Dynamic neural pathway charts
- Progressive layer activation visualization
- Step-by-step reasoning process display
- Comparative method analysis

## Deployment

- **Local Development**: Streamlit with port 5000
- **Docker**: Containerized deployment
- **Cloud**: Heroku and Streamlit Cloud support

## API Integration

- **OpenAI@**: GPT-4, GPT-3.5 Turbo for LLM responses
- **Anthropic**: Claude for alternative LLM analysis
- **Hugging Face**: Open-source model access
- **Captum**: PyTorch interpretability tools
