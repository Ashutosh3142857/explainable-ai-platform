#!/bin/bash

# ExplainableAI Platform - Quick Start Script
# This script sets up and runs the ExplainableAI Platform locally

set -e

echo "🧠 ExplainableAI Platform - Quick Start"
echo "========================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✓ Found Python $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install streamlit>=1.28.0 \
            pandas>=1.5.0 \
            numpy>=1.24.0 \
            scikit-learn>=1.3.0 \
            plotly>=5.15.0 \
            lime>=0.2.0 \
            matplotlib>=3.7.0 \
            seaborn>=0.12.0

# Create .streamlit directory if it doesn't exist
if [ ! -d ".streamlit" ]; then
    mkdir -p .streamlit
fi

# Create config.toml if it doesn't exist
if [ ! -f ".streamlit/config.toml" ]; then
    echo "⚙️  Creating Streamlit configuration..."
    cat > .streamlit/config.toml << EOF
[server]
headless = true
address = "0.0.0.0"
port = 5000
enableCORS = false
enableXsrfProtection = false

[theme]
base = "light"
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[browser]
gatherUsageStats = false
EOF
fi

echo ""
echo "🚀 Starting ExplainableAI Platform..."
echo "📱 Access the application at: http://localhost:5000"
echo "⏹️  Press Ctrl+C to stop the server"
echo ""

# Run the application
streamlit run app.py --server.port 5000