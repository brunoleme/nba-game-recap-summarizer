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

# Configure Docker to use more space and clean up unused images
echo '{"storage-driver": "overlay2", "storage-opts": ["overlay2.override_kernel_check=true"]}' > /etc/docker/daemon.json
systemctl restart docker

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
systemctl start nba-inference.service

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

# Create a simple health check script
cat > /opt/inference/health_check.sh << 'EOF'
#!/bin/bash
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
