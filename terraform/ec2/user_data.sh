#!/bin/bash

# Update system
apt-get update -y
apt-get upgrade -y

# Clean up package cache to free space
apt-get clean
apt-get autoremove -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker ubuntu

# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Configure Docker daemon for better resource management
cat > /etc/docker/daemon.json << EOF
{
  "storage-driver": "overlay2",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
EOF

# Restart Docker daemon with proper configuration
systemctl restart docker

# Wait for Docker to be ready
echo "Waiting for Docker daemon to be ready..."
for i in {1..30}; do
  if docker ps > /dev/null 2>&1; then
    echo "Docker daemon is ready!"
    break
  fi
  echo "Attempt $i/30: Docker not ready yet, waiting 2 seconds..."
  sleep 2
done

# Verify Docker is working
if ! docker ps > /dev/null 2>&1; then
  echo "ERROR: Docker daemon failed to start properly"
  exit 1
fi

# Clean up any existing Docker images to free space
docker system prune -af --volumes

# Configure AWS CLI for ECR
aws ecr get-login-password --region ${aws_region} | docker login --username AWS --password-stdin ${ecr_registry}

# Create application directory
mkdir -p /opt/inference
cd /opt/inference

# Create environment file
cat > .env << EOF
ENV=${environment}
MODEL_PATH=${model_path}
AWS_REGION=${aws_region}
HF_TOKEN=${hf_token}
HUGGINGFACEHUB_API_TOKEN=${hf_token}
EOF

# Create Docker Compose file
cat > docker-compose.yml << EOF
version: '3.8'
services:
  inference:
    image: ${ecr_repository_uri}
    container_name: nba-inference
    ports:
      - "8000:8000"
    environment:
      - ENV=${environment}
      - MODEL_PATH=${model_path}
      - HF_TOKEN=${hf_token}
      - HUGGINGFACEHUB_API_TOKEN=${hf_token}
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
EOF

# Create systemd service for Docker Compose
cat > /etc/systemd/system/nba-inference.service << EOF
[Unit]
Description=NBA Inference Service
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/inference
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
systemctl enable nba-inference.service

# Start the service with error handling
echo "Starting NBA inference service..."
if ! systemctl start nba-inference.service; then
  echo "ERROR: Failed to start nba-inference.service"
  systemctl status nba-inference.service --no-pager
  exit 1
fi

# Start the health check script in the background
chmod +x /opt/inference/health_check.sh
nohup /opt/inference/health_check.sh > /var/log/health_check.log 2>&1 &

# Wait a bit for the service to start
sleep 30

# Check service status
echo "Checking service status..."
systemctl status nba-inference.service --no-pager

# Check Docker containers
echo "Checking Docker containers..."
docker ps -a

# Check Docker logs
echo "Checking Docker logs..."
docker logs nba-inference 2>&1 | tail -50 || echo "No logs available yet"

# Check if the container is running
echo "Checking container status..."
docker ps -a | grep nba-inference || echo "Container not found"

# Check Docker Compose status
echo "Checking Docker Compose status..."
cd /opt/inference && docker-compose ps || echo "Docker Compose not running"

# Check systemd service logs
echo "Checking systemd service logs..."
journalctl -u nba-inference.service --no-pager -n 20 || echo "No service logs available"

# Create a comprehensive health check script
cat > /opt/inference/health_check.sh << 'EOF'
#!/bin/bash
echo "Starting health check script..."

# Function to check if port is open
check_port() {
    local port=$1
    if netstat -tln | grep ":$port " > /dev/null; then
        echo "✅ Port $port is open"
        return 0
    else
        echo "❌ Port $port is not open"
        return 1
    fi
}

# Function to check Docker container
check_docker() {
    if docker ps | grep nba-inference > /dev/null; then
        echo "✅ Docker container is running"
        return 0
    else
        echo "❌ Docker container is not running"
        docker ps -a | grep nba-inference
        return 1
    fi
}

# Function to check service health
check_service() {
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo "Health check attempt $attempt/$max_attempts"
        
        # Check if Docker container is running
        if ! check_docker; then
            echo "Docker container not running, waiting..."
            sleep 10
            ((attempt++))
            continue
        fi
        
        # Check if port is open
        if ! check_port 8000; then
            echo "Port 8000 not open, waiting..."
            sleep 10
            ((attempt++))
            continue
        fi
        
        # Try to hit the health endpoint
        if curl -f -s http://localhost:8000/health > /dev/null; then
            echo "✅ Service is healthy!"
            return 0
        else
            echo "Service not responding, waiting..."
            sleep 10
            ((attempt++))
        fi
    done
    
    echo "❌ Service failed to become healthy after $max_attempts attempts"
    return 1
}

# Run the health check
check_service

# Keep the script running for monitoring
while true; do
    if curl -f -s http://localhost:8000/health > /dev/null; then
        echo "$(date): Service is healthy"
    else
        echo "$(date): Service is not responding, restarting..."
        systemctl restart nba-inference.service
    fi
    sleep 60
done
EOF

chmod +x /opt/inference/health_check.sh

# Create systemd service for health check
cat > /etc/systemd/system/nba-health-check.service << EOF
[Unit]
Description=NBA Inference Health Check
After=nba-inference.service

[Service]
Type=simple
ExecStart=/opt/inference/health_check.sh
Restart=always
User=ubuntu

[Install]
WantedBy=multi-user.target
EOF

systemctl enable nba-health-check.service
systemctl start nba-health-check.service

# Log completion
echo "$(date): NBA Inference deployment completed" >> /var/log/nba-inference-deployment.log
