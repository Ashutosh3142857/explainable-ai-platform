# Deployment Guide

This guide covers multiple deployment options for the ExplainableAI Platform, from local development to production cloud deployments.

## Table of Contents
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployments](#cloud-deployments)
- [Server Deployment](#server-deployment)
- [Configuration](#configuration)
- [Environment Variables](#environment-variables)
- [Production Considerations](#production-considerations)

## Local Development

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/Ashutosh3142857/explainable-ai-platform.git
cd explainable-ai-platform
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install streamlit>=1.28.0
pip install pandas>=1.5.0
pip install numpy>=1.24.0
pip install scikit-learn>=1.3.0
pip install plotly>=5.15.0
pip install lime>=0.2.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
```

4. **Run the application**
```bash
streamlit run app.py --server.port 5000
```

5. **Access the application**
Open your browser and navigate to `http://localhost:5000`

## Docker Deployment

### Create Dockerfile

Create a `Dockerfile` in the project root:

```dockerfile
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
RUN pip3 install -r dependencies.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK CMD curl --fail http://localhost:5000/_stcore/health

# Run the application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=5000", "--server.address=0.0.0.0"]
```

### Docker Compose (Optional)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  explainable-ai:
    build: .
    ports:
      - "5000:5000"
    environment:
      - STREAMLIT_SERVER_PORT=5000
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

### Build and Run

```bash
# Build the Docker image
docker build -t explainable-ai-platform .

# Run the container
docker run -p 5000:5000 explainable-ai-platform

# Or using docker-compose
docker-compose up -d
```

## Cloud Deployments

### 1. Replit Deployment (Recommended for quick setup)

The application is already configured for Replit deployment:

1. Fork or clone the repository in Replit
2. The application will automatically install dependencies
3. Click the "Run" button
4. Use Replit's deployment feature for production hosting

### 2. Heroku Deployment

Create the following files:

**Procfile:**
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

**runtime.txt:**
```
python-3.11.1
```

**requirements.txt:** (copy from dependencies.txt)

**Deploy commands:**
```bash
# Install Heroku CLI and login
heroku login

# Create new app
heroku create your-app-name

# Deploy
git push heroku main
```

### 3. AWS EC2 Deployment

**Launch EC2 Instance:**
1. Choose Ubuntu 20.04 LTS AMI
2. Select t3.medium or larger for better performance
3. Configure security group to allow HTTP (80) and custom port (5000)

**Setup on EC2:**
```bash
# Connect to EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip python3-venv nginx -y

# Clone repository
git clone https://github.com/Ashutosh3142857/explainable-ai-platform.git
cd explainable-ai-platform

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r dependencies.txt

# Install PM2 for process management
npm install -g pm2

# Create PM2 ecosystem file
cat > ecosystem.config.js << EOF
module.exports = {
  apps: [{
    name: 'explainable-ai',
    script: 'streamlit',
    args: 'run app.py --server.port 5000 --server.address 0.0.0.0',
    interpreter: 'python3',
    cwd: '/home/ubuntu/explainable-ai-platform',
    env: {
      PATH: '/home/ubuntu/explainable-ai-platform/venv/bin:' + process.env.PATH
    }
  }]
}
EOF

# Start with PM2
pm2 start ecosystem.config.js
pm2 save
pm2 startup
```

**Configure Nginx (Optional):**
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### 4. Google Cloud Platform (Cloud Run)

**cloudbuild.yaml:**
```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/explainable-ai', '.']
images:
  - 'gcr.io/$PROJECT_ID/explainable-ai'
```

**Deploy commands:**
```bash
# Enable required APIs
gcloud services enable cloudbuild.googleapis.com run.googleapis.com

# Submit build
gcloud builds submit --config cloudbuild.yaml

# Deploy to Cloud Run
gcloud run deploy explainable-ai \
  --image gcr.io/$PROJECT_ID/explainable-ai \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 5000
```

### 5. Azure Container Instances

```bash
# Create resource group
az group create --name explainable-ai-rg --location eastus

# Create container instance
az container create \
  --resource-group explainable-ai-rg \
  --name explainable-ai-app \
  --image your-docker-registry/explainable-ai-platform:latest \
  --dns-name-label explainable-ai \
  --ports 5000 \
  --cpu 2 \
  --memory 4
```

## Server Deployment (Ubuntu/CentOS)

### System Requirements
- Ubuntu 18.04+ or CentOS 7+
- 2+ CPU cores
- 4GB+ RAM
- 20GB+ disk space

### Production Setup

```bash
# Install system dependencies
sudo apt update
sudo apt install python3 python3-pip python3-venv nginx supervisor -y

# Create application user
sudo useradd -m -s /bin/bash streamlit
sudo su - streamlit

# Clone and setup application
git clone https://github.com/Ashutosh3142857/explainable-ai-platform.git
cd explainable-ai-platform
python3 -m venv venv
source venv/bin/activate
pip install -r dependencies.txt

# Exit back to root user
exit

# Create supervisor configuration
sudo tee /etc/supervisor/conf.d/explainable-ai.conf << EOF
[program:explainable-ai]
command=/home/streamlit/explainable-ai-platform/venv/bin/streamlit run app.py --server.port 5000 --server.address 0.0.0.0
directory=/home/streamlit/explainable-ai-platform
user=streamlit
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/explainable-ai.log
environment=PATH="/home/streamlit/explainable-ai-platform/venv/bin"
EOF

# Start supervisor
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start explainable-ai
```

## Configuration

### Streamlit Configuration

Create `.streamlit/config.toml`:

```toml
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
```

## Environment Variables

Set these environment variables for production:

```bash
export STREAMLIT_SERVER_PORT=5000
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

## Production Considerations

### Performance Optimization

1. **Resource Allocation:**
   - Minimum 2 CPU cores
   - 4GB RAM for small datasets
   - 8GB+ RAM for larger datasets

2. **Caching:**
   - Enable Streamlit's built-in caching
   - Consider Redis for distributed caching

3. **Load Balancing:**
   - Use Nginx or HAProxy for multiple instances
   - Enable session affinity if needed

### Security

1. **Network Security:**
   - Use HTTPS in production
   - Configure firewall rules
   - Restrict access to necessary ports

2. **Application Security:**
   - Validate all user inputs
   - Sanitize uploaded files
   - Implement rate limiting

### Monitoring and Logging

```bash
# View application logs
tail -f /var/log/explainable-ai.log

# Check system resources
htop
df -h
free -m

# Monitor with supervisor
sudo supervisorctl status
```

### SSL/TLS Setup (Let's Encrypt)

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## Backup and Recovery

```bash
# Backup application data
tar -czf backup-$(date +%Y%m%d).tar.gz /home/streamlit/explainable-ai-platform

# Database backup (if using external database)
# Configure based on your database system

# Automated backup script
cat > /home/streamlit/backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/home/streamlit/backups"
mkdir -p $BACKUP_DIR
tar -czf $BACKUP_DIR/explainable-ai-$DATE.tar.gz /home/streamlit/explainable-ai-platform
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
EOF

chmod +x /home/streamlit/backup.sh

# Add to crontab for daily backups
# 0 2 * * * /home/streamlit/backup.sh
```

## Troubleshooting

### Common Issues

1. **Port conflicts:**
   ```bash
   sudo netstat -tlnp | grep :5000
   sudo kill -9 <PID>
   ```

2. **Permission errors:**
   ```bash
   sudo chown -R streamlit:streamlit /home/streamlit/explainable-ai-platform
   ```

3. **Memory issues:**
   ```bash
   # Monitor memory usage
   free -m
   # Consider upgrading instance or optimizing code
   ```

4. **Dependency conflicts:**
   ```bash
   # Recreate virtual environment
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   pip install -r dependencies.txt
   ```

## Support

For deployment issues:
1. Check the application logs first
2. Verify all dependencies are installed
3. Ensure proper network configuration
4. Review security group/firewall settings

---

Choose the deployment method that best fits your needs and infrastructure requirements. For quick prototyping, use local development or Replit. For production applications, consider cloud deployments with proper monitoring and security measures.