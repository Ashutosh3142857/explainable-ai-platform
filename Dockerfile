FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY dependencies.txt .
RUN pip3 install streamlit>=1.28.0 \
                pandas>=1.5.0 \
                numpy>=1.24.0 \
                scikit-learn>=1.3.0 \
                plotly>=5.15.0 \
                lime>=0.2.0 \
                matplotlib>=3.7.0 \
                seaborn>=0.12.0

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 streamlit && chown -R streamlit:streamlit /app
USER streamlit

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK CMD curl --fail http://localhost:5000/_stcore/health

# Run the application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=5000", "--server.address=0.0.0.0", "--server.headless=true"]