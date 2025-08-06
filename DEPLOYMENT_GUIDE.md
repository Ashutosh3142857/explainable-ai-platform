# Deployment Guide - LLM Neural Pathway Visualizer

This guide provides comprehensive instructions for deploying the Real-Time LLM Neural Pathway Visualizer on various platforms.

## Table of Contents
1. [Local Development](#local-development)
2. [Server Deployment](#server-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Environment Configuration](#environment-configuration)
6. [Troubleshooting](#troubleshooting)

---

## Local Development

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning)

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/Ashutosh3142857/explainable-ai-platform.git
cd explainable-ai-platform
```

2. **Create virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables** (optional)
```bash
# Create .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
echo "ANTHROPIC_API_KEY=your_api_key_here" >> .env
echo "GEMINI_API_KEY=your_api_key_here" >> .env
```

5. **Run the application**
```bash
streamlit run app.py --server.port 8501
```

6. **Access the application**
Open your browser and navigate to `http://localhost:8501`

### Alternative Installation Methods

#### Using Poetry
```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Run application
poetry run streamlit run app.py --server.port 8501
```

#### Using Conda
```bash
# Create conda environment
conda create -n llm-visualizer python=3.9
conda activate llm-visualizer

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py --server.port 8501
```

---

## Server Deployment

### Ubuntu/Debian Server

1. **Update system and install Python**
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip python3-venv git nginx -y
```

2. **Clone and setup application**
```bash
cd /opt
sudo git clone https://github.com/Ashutosh3142857/explainable-ai-platform.git
sudo chown -R $USER:$USER explainable-ai-platform
cd explainable-ai-platform

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Create systemd service**
```bash
sudo nano /etc/systemd/system/llm-visualizer.service
```

Add the following content:
```ini
[Unit]
Description=LLM Neural Pathway Visualizer
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/explainable-ai-platform
Environment=PATH=/opt/explainable-ai-platform/venv/bin
ExecStart=/opt/explainable-ai-platform/venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

4. **Configure Nginx reverse proxy**
```bash
sudo nano /etc/nginx/sites-available/llm-visualizer
```

Add the following configuration:
```nginx
server {
    listen 80;
    server_name your-domain.com;  # Replace with your domain

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

5. **Enable and start services**
```bash
# Enable Nginx site
sudo ln -s /etc/nginx/sites-available/llm-visualizer /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Enable and start application service
sudo systemctl enable llm-visualizer
sudo systemctl start llm-visualizer
sudo systemctl status llm-visualizer
```

### CentOS/RHEL Server

```bash
# Install dependencies
sudo yum update -y
sudo yum install python3 python3-pip git nginx -y

# Follow similar steps as Ubuntu, adjusting package manager commands
```

---

## Cloud Deployment

### AWS EC2

1. **Launch EC2 Instance**
   - Choose Ubuntu 20.04 LTS AMI
   - Select appropriate instance type (t3.medium recommended)
   - Configure security group to allow HTTP (80), HTTPS (443), and SSH (22)

2. **Connect and deploy**
```bash
# Connect via SSH
ssh -i your-key.pem ubuntu@your-ec2-ip

# Follow server deployment steps above
```

3. **Optional: Set up SSL with Let's Encrypt**
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### Google Cloud Platform (GCP)

1. **Create Compute Engine instance**
```bash
gcloud compute instances create llm-visualizer \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --machine-type=e2-medium \
    --tags=http-server,https-server
```

2. **Configure firewall**
```bash
gcloud compute firewall-rules create allow-streamlit \
    --allow tcp:8501 \
    --source-ranges 0.0.0.0/0 \
    --description "Allow Streamlit"
```

3. **Deploy application** (follow server deployment steps)

### Azure Virtual Machine

1. **Create VM using Azure CLI**
```bash
az vm create \
    --resource-group myResourceGroup \
    --name llm-visualizer-vm \
    --image UbuntuLTS \
    --admin-username azureuser \
    --generate-ssh-keys
```

2. **Open necessary ports**
```bash
az vm open-port --port 80 --resource-group myResourceGroup --name llm-visualizer-vm
az vm open-port --port 8501 --resource-group myResourceGroup --name llm-visualizer-vm
```

3. **Deploy application** (follow server deployment steps)

### Heroku Deployment

1. **Create required files**

Create `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

Create `requirements.txt`:
```
streamlit>=1.28.0
numpy>=1.21.0
pandas>=1.3.0
plotly>=5.0.0
openai>=1.0.0
anthropic>=0.20.0
google-genai>=0.4.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
lime>=0.2.0
```

2. **Deploy to Heroku**
```bash
# Install Heroku CLI and login
heroku login

# Create Heroku app
heroku create your-app-name

# Set environment variables
heroku config:set OPENAI_API_KEY=your_key
heroku config:set ANTHROPIC_API_KEY=your_key

# Deploy
git push heroku main
```

### Streamlit Cloud

1. **Fork the repository** on GitHub
2. **Go to** [share.streamlit.io](https://share.streamlit.io)
3. **Connect your GitHub account**
4. **Deploy** by selecting your forked repository
5. **Add secrets** in the Streamlit Cloud dashboard:
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`
   - `GEMINI_API_KEY`

---

## Docker Deployment

### Build and Run Locally

1. **Build Docker image**
```bash
docker build -t llm-visualizer .
```

2. **Run container**
```bash
docker run -p 8501:8501 \
    -e OPENAI_API_KEY=your_api_key \
    -e ANTHROPIC_API_KEY=your_api_key \
    llm-visualizer
```

### Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  llm-visualizer:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - llm-visualizer
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

### Production Docker Setup

1. **Multi-stage Dockerfile** for optimization:
```dockerfile
# Build stage
FROM python:3.9-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

EXPOSE 8501

ENV PATH=/root/.local/bin:$PATH

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. **Deploy with Docker Swarm** (for high availability):
```bash
docker swarm init
docker stack deploy -c docker-compose.yml llm-visualizer-stack
```

---

## Environment Configuration

### Required Environment Variables

```bash
# API Keys (optional, for real LLM integration)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
PERPLEXITY_API_KEY=...
XAI_API_KEY=...

# Application Settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

### Streamlit Configuration

Create `.streamlit/config.toml`:
```toml
[server]
headless = true
address = "0.0.0.0"
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

---

## Troubleshooting

### Common Issues

1. **Port already in use**
```bash
# Find process using port 8501
lsof -i :8501
# Kill the process
kill -9 <PID>
```

2. **Module not found errors**
```bash
# Ensure virtual environment is activated
source venv/bin/activate
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

3. **Permission denied errors**
```bash
# Fix file permissions
chmod +x run.sh
# Or run with sudo for system-wide installation
sudo streamlit run app.py
```

4. **Memory issues on small servers**
```bash
# Add swap space
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Performance Optimization

1. **For production deployments**:
   - Use at least 2GB RAM
   - Enable caching in Streamlit
   - Use a reverse proxy (Nginx/Apache)
   - Implement SSL/TLS

2. **For high-traffic scenarios**:
   - Use load balancing
   - Implement Redis for session management
   - Consider container orchestration (Kubernetes)

### Monitoring and Logging

1. **Application logs**
```bash
# View systemd service logs
sudo journalctl -u llm-visualizer -f

# Docker logs
docker logs -f container_name
```

2. **Resource monitoring**
```bash
# Install monitoring tools
sudo apt install htop iotop

# Monitor resources
htop
```

---

## Security Considerations

1. **Environment Variables**: Never commit API keys to version control
2. **Firewall**: Only open necessary ports
3. **SSL/TLS**: Always use HTTPS in production
4. **Updates**: Keep dependencies updated regularly
5. **Access Control**: Implement authentication if needed

---

## Support and Updates

For issues or questions:
- Create an issue on [GitHub](https://github.com/Ashutosh3142857/explainable-ai-platform/issues)
- Check the troubleshooting section above
- Review Streamlit documentation for framework-specific issues

Regular updates will be pushed to the main branch. Monitor the repository for new releases and features.