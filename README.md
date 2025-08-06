# ExplainableAI Platform 🧠

A comprehensive Streamlit-based platform that makes black box machine learning models transparent and interpretable through interactive visualizations and explanations.

## 🌟 Features

- **Interactive Dashboard**: Clean, modern interface with gradient styling
- **Model Upload**: Support for pickle and joblib model formats
- **Sample Datasets**: Built-in Iris, Wine, and Diabetes datasets for quick testing
- **LIME Analysis**: Local interpretable model-agnostic explanations
- **Model Performance**: Comprehensive metrics and visualizations
- **Examples**: Pre-built demonstrations and tutorials
- **Robust Error Handling**: Graceful fallbacks for missing dependencies

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Streamlit
- scikit-learn
- LIME (for local explanations)
- plotly (for interactive visualizations)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/Ashutosh3142857/explainable-ai-platform.git
cd explainable-ai-platform
```

2. Install dependencies:
```bash
pip install streamlit>=1.28.0 pandas>=1.5.0 numpy>=1.24.0 scikit-learn>=1.3.0 plotly>=5.15.0 lime>=0.2.0 matplotlib>=3.7.0 seaborn>=0.12.0
```

3. Run the application:
```bash
streamlit run app.py --server.port 5000
```

4. Open your browser and navigate to `http://localhost:5000`

### Quick Docker Setup

```bash
# Using Docker Compose (recommended)
docker-compose up -d

# Or build and run manually
docker build -t explainable-ai-platform .
docker run -p 5000:5000 explainable-ai-platform
```

## 📁 Project Structure

```
├── app.py                 # Main Streamlit application
├── pages/                 # Multi-page application structure
│   ├── 1_Model_Upload.py  # Model and data upload functionality
│   ├── 2_SHAP_Analysis.py # SHAP-based explanations (with fallbacks)
│   ├── 3_LIME_Analysis.py # LIME-based local explanations
│   └── 4_Examples.py      # Pre-built examples and demos
├── utils/                 # Utility modules
│   ├── model_handler.py   # Model loading and validation
│   ├── explanation_utils.py # Explanation generation utilities
│   └── data_utils.py      # Data processing utilities
├── data/                  # Sample datasets and utilities
│   └── sample_datasets.py # Built-in sample data generators
└── requirements.txt       # Python dependencies
```

## 🎯 How to Use

### 1. Upload Your Model
- Navigate to the "Model Upload" page
- Upload a trained model (pickle/joblib format)
- Or select from built-in sample datasets
- The platform will automatically analyze your model and data

### 2. Explore Explanations
- **LIME Analysis**: Get local explanations for individual predictions
- **Model Performance**: View comprehensive metrics and visualizations
- **Feature Importance**: Understand which features matter most

### 3. Try Examples
- Visit the "Examples" page for pre-built demonstrations
- Learn from Iris classification, Wine classification, and Diabetes regression examples
- See how different explanation methods work in practice

## 🔧 Supported Models

- **Classification Models**: RandomForest, LogisticRegression, SVM, etc.
- **Regression Models**: RandomForest, LinearRegression, etc.
- **Model Formats**: Pickle (.pkl), Joblib (.joblib)

## 🎨 Key Technologies

- **Frontend**: Streamlit with custom CSS styling
- **Visualizations**: Plotly for interactive charts
- **ML Interpretability**: LIME for local explanations
- **Data Processing**: Pandas, NumPy, scikit-learn
- **Model Support**: Any scikit-learn compatible model

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🚀 Deployment Options

This platform can be deployed in multiple ways:

- **Local Development**: Quick setup for testing and development
- **Docker**: Containerized deployment for any environment
- **Cloud Platforms**: AWS, Google Cloud, Azure, Heroku
- **Replit**: One-click deployment (already configured)
- **Dedicated Servers**: Ubuntu/CentOS with nginx and SSL

📖 **[View Complete Deployment Guide](DEPLOYMENT.md)** for detailed instructions on all deployment methods.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Explanations powered by [LIME](https://github.com/marcotcr/lime)
- Visualizations created with [Plotly](https://plotly.com/)
- Machine learning utilities from [scikit-learn](https://scikit-learn.org/)

## 📞 Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

**Made with ❤️ for transparent and explainable AI**