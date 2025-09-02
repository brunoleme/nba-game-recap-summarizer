# EC2 Deployment Guide

This guide explains how to deploy the NBA Game Recap Summarizer inference service to EC2 for debugging and validation purposes.

## 🎯 Purpose

The EC2 deployment provides:
- **Easy debugging** with direct SSH access
- **Quick validation** of new models before ECS deployment
- **Cost-effective testing** with on-demand instances
- **Full control** over the environment

## 🚀 Quick Start

### 1. Deploy via GitHub Actions

1. Go to **Actions** → **Manual EC2 Deployment**
2. Click **Run workflow**
3. Fill in the parameters:
   - **Environment**: `dev`, `staging`, or `prod`
   - **Pipeline ID**: SageMaker pipeline ID (e.g., `ef723d78-6c3b-4293-b237-d65039a0dd92`)
   - **Instance Type**: Choose based on your needs:
     - `g4dn.xlarge` - 1 GPU, 4 vCPU, 16GB RAM (recommended for testing)
     - `g4dn.2xlarge` - 1 GPU, 8 vCPU, 32GB RAM
     - `g4dn.4xlarge` - 1 GPU, 16 vCPU, 64GB RAM
     - `g5.xlarge` - 1 GPU, 4 vCPU, 16GB RAM (newer generation)
     - `g5.2xlarge` - 1 GPU, 8 vCPU, 32GB RAM
   - **SSH Key Name**: Your EC2 key pair name

4. Click **Run workflow**

### 2. Monitor Deployment

The workflow will:
- ✅ Run inference tests
- 🏗️ Build and push Docker image to ECR
- 🚀 Deploy EC2 instance with Terraform
- ⏳ Wait for service to be ready
- 🏥 Run health checks
- 🧠 Test inference endpoint

### 3. Access Your Service

Once deployed, you'll get:
- **Health Check**: `http://<instance-ip>:8000/health`
- **API Documentation**: `http://<instance-ip>:8000/docs`
- **Inference Endpoint**: `http://<instance-ip>:8000/summarize_recap`

## 🔧 Manual Operations

### SSH Access
```bash
ssh -i your-key.pem ec2-user@<instance-ip>
```

### Check Service Status
```bash
# Check systemd service
sudo systemctl status nba-inference

# Check Docker containers
sudo docker-compose -f /opt/inference/docker-compose.yml ps

# View logs
sudo docker-compose -f /opt/inference/docker-compose.yml logs -f
```

### Restart Service
```bash
sudo systemctl restart nba-inference
```

### Update Model
```bash
# Edit environment file
sudo nano /opt/inference/.env

# Restart service
sudo systemctl restart nba-inference
```

## 📊 Monitoring

### Use the Monitoring Script
```bash
./scripts/monitor-ec2.sh <instance-ip> <environment>
```

### Manual Health Check
```bash
curl http://<instance-ip>:8000/health
```

### Test Inference
```bash
curl -X POST "http://<instance-ip>:8000/summarize_recap" \
  -H "Content-Type: application/json" \
  -d '{
    "game_recap": "Lakers beat Suns in overtime with a spectacular game-winning shot by Bryant.",
    "max_length": 100
  }'
```

## 🧹 Cleanup

### Automatic Cleanup via GitHub Actions

1. Go to **Actions** → **Cleanup EC2 Deployment**
2. Click **Run workflow**
3. Fill in:
   - **Environment**: The environment to cleanup
   - **Confirm Destroy**: Type `DESTROY` to confirm
4. Click **Run workflow**

### Manual Cleanup
```bash
cd terraform/ec2
terraform destroy
```

## 🏗️ Architecture

### Infrastructure Components
- **VPC**: Isolated network environment
- **Public Subnet**: EC2 instance with public IP
- **Security Group**: Allows SSH (22) and HTTP (8000)
- **IAM Role**: ECR and S3 access permissions
- **EC2 Instance**: GPU-enabled with Deep Learning AMI

### Application Stack
- **Docker**: Containerized inference service
- **Docker Compose**: Service orchestration
- **Systemd**: Service management and auto-restart
- **Health Check**: Automated monitoring and recovery

## 🔒 Security

### Network Security
- VPC with public subnet only
- Security group restricts access to ports 22 and 8000
- No private subnets (simplified for debugging)

### Access Control
- SSH key-based authentication
- IAM role with minimal required permissions
- ECR and S3 access only

## 💰 Cost Optimization

### Instance Types
- **g4dn.xlarge**: ~$0.526/hour (recommended for testing)
- **g4dn.2xlarge**: ~$0.752/hour
- **g4dn.4xlarge**: ~$1.204/hour

### Cost Management
- Deploy only when needed
- Use cleanup workflow to destroy resources
- Monitor usage in AWS Cost Explorer

## 🐛 Debugging

### Common Issues

#### Service Not Starting
```bash
# Check logs
sudo journalctl -u nba-inference -f

# Check Docker logs
sudo docker-compose -f /opt/inference/docker-compose.yml logs
```

#### Model Loading Issues
```bash
# Check model path
cat /opt/inference/.env

# Test S3 access
aws s3 ls s3://nba-recap-summarization-model-dev/
```

#### GPU Issues
```bash
# Check GPU availability
nvidia-smi

# Check Docker GPU support
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Log Locations
- **Application logs**: `/opt/inference/docker-compose.yml logs`
- **System logs**: `sudo journalctl -u nba-inference`
- **Health check logs**: `sudo journalctl -u nba-health-check`
- **Deployment logs**: `/var/log/nba-inference-deployment.log`

## 🔄 Workflow Comparison

| Feature | ECS Deployment | EC2 Deployment |
|---------|----------------|----------------|
| **Purpose** | Production | Debugging/Validation |
| **Access** | Load Balancer | Direct IP |
| **Scaling** | Auto-scaling | Manual |
| **Cost** | Always running | On-demand |
| **Debugging** | Limited | Full SSH access |
| **Setup Time** | ~10 minutes | ~5 minutes |
| **Cleanup** | Manual | Automated workflow |

## 📝 Best Practices

1. **Always cleanup** after testing to avoid costs
2. **Use appropriate instance types** for your testing needs
3. **Monitor logs** during deployment for issues
4. **Test thoroughly** before promoting to ECS
5. **Keep SSH keys secure** and rotate regularly

## 🆘 Support

If you encounter issues:
1. Check the GitHub Actions logs
2. SSH into the instance and check logs
3. Use the monitoring script for diagnostics
4. Review this documentation for common solutions
