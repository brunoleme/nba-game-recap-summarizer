# Terraform Infrastructure Documentation

This directory contains all Infrastructure as Code (IaC) for the NBA Game Recap Summarizer project using Terraform.

## 📁 Directory Structure

```
terraform/
├── alb/              # Application Load Balancer module
├── ec2/              # EC2 standalone deployment (testing)
├── ecs/              # ECS cluster and services (production)
├── envs/             # Environment-specific configurations
│   ├── dev/          # Development environment
│   ├── staging/      # Staging environment
│   └── prod/         # Production environment
├── iam/              # IAM roles and policies
└── network/          # VPC, subnets, security groups
```

## 🏗️ Architecture Overview

### Production Architecture (ECS)

```
Internet
    ↓
Application Load Balancer (ALB)
    ↓
[Canary Traffic Split]
    ├── 90% → Target Group V1 (Stable)
    │          ↓
    │      ECS Service V1
    │          ↓
    │      ECS Tasks (Container)
    └── 10% → Target Group V2 (Canary)
               ↓
           ECS Service V2
               ↓
           ECS Tasks (Container)

Auto Scaling Group (GPU Instances)
    ↓
ECS Capacity Provider
    ↓
Managed Scaling (Target: 80%)
```

### Testing Architecture (EC2)

```
Internet
    ↓
EC2 Instance (Public IP)
    ↓
Docker Container
    ↓
Inference API (Port 8000)
```

## 📦 Modules

### 1. Network Module (`network/`)

Creates VPC and networking infrastructure.

**Resources**:
- VPC (10.0.0.0/16)
- Public Subnets (2 AZs)
- Private Subnets (2 AZs)
- Internet Gateway
- NAT Gateway (for private subnet internet access)
- Route Tables
- Security Groups (ALB, ECS instances, ECS services)

**Outputs**:
- `vpc_id`
- `public_subnet_ids`
- `private_subnet_ids`
- `alb_sg_id`
- `ecs_sg_id`

**Security Groups**:
```hcl
# ALB Security Group
- Ingress: 80, 443 from 0.0.0.0/0
- Egress: All

# ECS Security Group
- Ingress: 8000 from ALB
- Egress: All
```

---

### 2. ALB Module (`alb/`)

Manages Application Load Balancer and target groups.

**Resources**:
- Application Load Balancer
- Target Group V1 (stable version)
- Target Group V2 (canary version, optional)
- Listener (port 80/443)
- Listener Rules (weighted routing for canary)

**Key Features**:
- Health checks on `/health` endpoint
- Sticky sessions (60 seconds)
- Weighted traffic splitting
- Deregistration delay: 30 seconds

**Variables**:
```hcl
enable_canary     = false        # Enable canary deployment
canary_weight_v1  = 90           # Traffic % to stable version
canary_weight_v2  = 10           # Traffic % to canary version
```

**Outputs**:
- `alb_dns_name`
- `alb_arn`
- `target_group_v1_arn`
- `target_group_v2_arn`

---

### 3. ECS Module (`ecs/`)

Manages ECS cluster, services, and auto-scaling.

**Resources**:
- ECS Cluster
- ECS Task Definition (V1 and V2)
- ECS Service (V1 and V2)
- Launch Template (GPU-enabled)
- Auto Scaling Group
- Capacity Provider
- CloudWatch Log Groups

**Task Definition**:
```hcl
cpu    = 4096      # 4 vCPU
memory = 14000     # 14 GB
gpu    = 1         # 1 GPU

container:
  port = 8000
  environment:
    - MODEL_PATH
    - HF_TOKEN
  resourceRequirements:
    - type: GPU, value: 1
```

**Auto Scaling**:
```hcl
asg_min_size         = 1
asg_max_size         = 10
asg_desired_capacity = 1

managed_scaling:
  target_capacity = 80%
  min_step        = 1
  max_step        = 1000
  warmup_period   = 300s
```

**Variables**:
```hcl
instance_type        = "g4dn.xlarge"  # GPU instance type
enable_canary        = false           # Enable V2 service
container_image      = "ecr-uri:tag"   # Docker image
model_path           = "s3://..."      # S3 model path
```

**Outputs**:
- `cluster_id`
- `service_v1_name`
- `service_v2_name`
- `log_group_name`

---

### 4. IAM Module (`iam/`)

Manages IAM roles and policies.

**Resources**:
- ECS Instance Role (for EC2 instances in ECS cluster)
- ECS Task Execution Role (for pulling images, logging)
- ECS Task Role (for application-level permissions)
- Instance Profile

**Permissions**:
```hcl
ECS Instance Role:
  - ECS agent registration
  - ECR image pulling
  - CloudWatch logging

Task Execution Role:
  - ECR GetAuthorizationToken
  - ECR pull operations
  - CloudWatch Logs

Task Role:
  - S3 GetObject (model artifacts)
  - S3 ListBucket
  - CloudWatch PutMetrics
```

**Variables**:
```hcl
name_prefix         = "nba-recap-summarizer"
model_bucket_prefix = "nba-recap-summarization-model-"
```

**Outputs**:
- `instance_profile_name`
- `task_execution_role_arn`
- `task_role_arn`

---

### 5. EC2 Module (`ec2/`)

Standalone EC2 deployment for testing.

**Resources**:
- VPC (separate from ECS)
- Public Subnet
- Internet Gateway
- Security Group (SSH + API)
- EC2 Instance (GPU-enabled)
- IAM Role and Instance Profile

**User Data**:
- Install Docker and docker-compose
- Login to ECR
- Pull inference image
- Create systemd service
- Start inference API
- Health check monitoring

**Variables**:
```hcl
environment         = "dev"
instance_type       = "g4dn.xlarge"
ssh_key_name        = "my-key"
inference_image_uri = "ecr-uri:tag"
model_path          = "s3://..."
hf_token            = "hf_..."
```

**Outputs**:
- `instance_id`
- `instance_public_ip`
- `instance_public_dns`

**Access**:
```bash
ssh -i your-key.pem ec2-user@<public-ip>
curl http://<public-ip>:8000/health
```

---

## 🌍 Environments

### Development (`envs/dev/`)

**Purpose**: Rapid development and testing

**Configuration**:
```hcl
env           = "dev"
instance_type = "g4dn.xlarge"   # 1 GPU, 16GB RAM
enable_canary = false

ASG:
  min_size     = 1
  max_size     = 2
  desired      = 1
```

**Usage**:
```bash
cd terraform/envs/dev
terraform init
terraform apply
```

---

### Staging (`envs/staging/`)

**Purpose**: Production-like validation

**Configuration**:
```hcl
env           = "staging"
instance_type = "g4dn.xlarge"
enable_canary = false

ASG:
  min_size     = 1
  max_size     = 3
  desired      = 1
```

**Usage**:
```bash
cd terraform/envs/staging
terraform init
terraform apply
```

---

### Production (`envs/prod/`)

**Purpose**: Live production serving

**Configuration**:
```hcl
env           = "prod"
instance_type = "g4dn.2xlarge"  # 1 GPU, 32GB RAM
enable_canary = true             # For canary deployments

ASG:
  min_size     = 2               # High availability
  max_size     = 10
  desired      = 2
```

**Usage**:
```bash
cd terraform/envs/prod
terraform init

# Initial deployment (no canary)
terraform apply

# Enable canary deployment
terraform apply -var="enable_canary=true"

# Adjust traffic weights
terraform apply \
  -var="enable_canary=true" \
  -var="canary_weight_v1=50" \
  -var="canary_weight_v2=50"
```

---

## 🚀 Deployment Workflows

### Initial ECS Deployment

```bash
# 1. Configure variables
cd terraform/envs/dev
vi terraform.tfvars

# 2. Initialize Terraform
terraform init

# 3. Review plan
terraform plan

# 4. Apply infrastructure
terraform apply

# 5. Get ALB DNS name
terraform output alb_dns_name

# 6. Test deployment
curl http://<alb-dns>/health
```

### Canary Deployment

```bash
# 1. Deploy new version to V2
cd terraform/envs/prod

# 2. Enable canary (10% traffic to new version)
terraform apply \
  -var="enable_canary=true" \
  -var="canary_weight_v1=90" \
  -var="canary_weight_v2=10"

# 3. Monitor metrics for 30-60 minutes
# Check error rates, latency, ROUGE scores

# 4. Gradually increase V2 traffic
terraform apply -var="canary_weight_v1=75" -var="canary_weight_v2=25"
terraform apply -var="canary_weight_v1=50" -var="canary_weight_v2=50"
terraform apply -var="canary_weight_v1=25" -var="canary_weight_v2=75"

# 5. Promote V2 to V1 (via GitHub Actions)
# This makes V2 the new stable version

# 6. Disable canary
terraform apply -var="enable_canary=false"
```

### EC2 Testing Deployment

```bash
# 1. Configure EC2 variables
cd terraform/ec2
cp terraform.tfvars.example terraform.tfvars
vi terraform.tfvars

# 2. Deploy
terraform init
terraform apply

# 3. Get instance IP
terraform output instance_public_ip

# 4. SSH into instance
ssh -i your-key.pem ec2-user@<instance-ip>

# 5. Check logs
sudo journalctl -u nba-inference -f

# 6. Test API
curl http://<instance-ip>:8000/health

# 7. Cleanup when done
terraform destroy
```

---

## 🔧 Configuration

### Required Variables

**All Environments**:
```hcl
# terraform.tfvars
app_name            = "nba-recap-summarizer"
env                 = "dev"
aws_region          = "us-east-1"
ssh_key_name        = "my-ecs-key"
instance_type       = "g4dn.xlarge"
inference_image_uri = "123456789.dkr.ecr.us-east-1.amazonaws.com/nba-recap:dev-latest"
model_path          = "s3://nba-recap-summarization-model-dev/pipeline-id/hf_model"
```

**Production Additional**:
```hcl
enable_canary    = true
canary_weight_v1 = 90
canary_weight_v2 = 10
```

### Backend Configuration

**S3 Backend** (recommended for team collaboration):
```hcl
# backend.tf
terraform {
  backend "s3" {
    bucket         = "nba-recap-model-tf-state-bucket"
    key            = "dev/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-locks"
    encrypt        = true
  }
}
```

**Setup S3 Backend**:
```bash
# Create S3 bucket
aws s3 mb s3://nba-recap-model-tf-state-bucket

# Create DynamoDB table for state locking
aws dynamodb create-table \
  --table-name terraform-locks \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST
```

---

## 📊 Cost Estimation

### ECS Production (Always Running)

```
# 2 x g4dn.xlarge instances
2 × $0.526/hour × 730 hours/month = $768/month

# ALB
$22.50/month + data transfer

# NAT Gateway
$32.85/month + data transfer

# Total: ~$850-900/month
```

### EC2 Testing (On-Demand)

```
# 1 x g4dn.xlarge instance
1 × $0.526/hour × hours_used

# Example: 8 hours of testing
8 × $0.526 = $4.21

# Cost-effective for intermittent testing
```

### Cost Optimization Tips

1. **Stop EC2 instances** when not testing
2. **Use spot instances** for non-critical ECS tasks (50-70% savings)
3. **Right-size instances** based on actual usage
4. **Scale down auto-scaling** during low traffic periods
5. **Use S3 lifecycle policies** to archive old models

---

## 🔒 Security Best Practices

### Network Security

```hcl
✅ ECS tasks in private subnets (no public IP)
✅ ALB in public subnets only
✅ Security groups with least privilege
✅ NAT Gateway for outbound traffic
❌ No direct internet access to ECS tasks
```

### IAM Security

```hcl
✅ Separate roles for instances and tasks
✅ Least privilege principle
✅ S3 bucket-specific access (not wildcard)
✅ No hard-coded credentials
❌ Don't use AdministratorAccess
```

### Secrets Management

```bash
# Use AWS Secrets Manager for sensitive values
aws secretsmanager create-secret \
  --name nba-recap/hf-token \
  --secret-string "hf_..."

# Reference in Terraform
data "aws_secretsmanager_secret_version" "hf_token" {
  secret_id = "nba-recap/hf-token"
}
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. ECS Service Won't Start

```bash
# Check ECS service events
aws ecs describe-services \
  --cluster nba-recap-dev-cluster \
  --services nba-recap-dev-v1-service

# Check task logs
aws logs tail /ecs/nba-recap-dev --follow

# Common causes:
- Image doesn't exist or can't be pulled
- Insufficient GPU capacity
- Model path incorrect
- Health check failing
```

#### 2. Terraform Apply Fails

```bash
# Check state lock
aws dynamodb get-item \
  --table-name terraform-locks \
  --key '{"LockID":{"S":"nba-recap-dev"}}'

# Force unlock (use with caution)
terraform force-unlock <lock-id>

# Common causes:
- State locked by another process
- AWS credentials expired
- Resource limits exceeded
- Dependency cycle
```

#### 3. Auto-Scaling Not Working

```bash
# Check capacity provider status
aws ecs describe-capacity-providers \
  --capacity-providers nba-recap-dev-gpu

# Check ASG status
aws autoscaling describe-auto-scaling-groups \
  --auto-scaling-group-names nba-recap-dev-gpu-asg

# Common causes:
- Managed scaling disabled
- Instance type unavailable in AZ
- Service limits reached
- Insufficient capacity reservation
```

#### 4. Health Check Failures

```bash
# Test health endpoint manually
curl http://<alb-dns>/health

# Check target group health
aws elbv2 describe-target-health \
  --target-group-arn <target-group-arn>

# Common causes:
- Model not loaded
- Container port mismatch (8000)
- Security group blocking traffic
- Application crashed
```

#### 5. High Costs

```bash
# Check resource usage
aws ce get-cost-and-usage \
  --time-period Start=2025-10-01,End=2025-10-11 \
  --granularity DAILY \
  --metrics BlendedCost

# Common causes:
- EC2 instances left running
- Auto-scaling not scaling down
- NAT Gateway data transfer
- ALB idle charges
```

---

## 📈 Monitoring

### CloudWatch Dashboards

```bash
# View ECS metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name CPUUtilization \
  --dimensions Name=ServiceName,Value=nba-recap-prod-v1-service \
  --start-time 2025-10-11T00:00:00Z \
  --end-time 2025-10-11T23:59:59Z \
  --period 300 \
  --statistics Average
```

### Key Metrics to Monitor

- **ECS Cluster**: CPU, Memory, GPU utilization
- **ALB**: Request count, latency, error rate, target health
- **Auto Scaling**: Desired capacity, running tasks, pending tasks
- **Application**: Inference latency, ROUGE scores, error logs

---

## 🔄 Maintenance

### Regular Updates

```bash
# Update Terraform providers
terraform init -upgrade

# Update task definitions (new image)
terraform apply -var="container_image=new-image:tag"

# Rotate SSH keys
terraform apply -var="ssh_key_name=new-key"
```

### Backup and Recovery

```bash
# Backup Terraform state
aws s3 cp s3://nba-recap-model-tf-state-bucket/dev/terraform.tfstate \
  terraform.tfstate.backup

# Restore from backup
aws s3 cp terraform.tfstate.backup \
  s3://nba-recap-model-tf-state-bucket/dev/terraform.tfstate
```

---

## 📚 Additional Resources

- [AWS ECS Best Practices](https://docs.aws.amazon.com/AmazonECS/latest/bestpracticesguide/intro.html)
- [Terraform AWS Provider Docs](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [GPU Instance Types](https://aws.amazon.com/ec2/instance-types/g4/)
- [ALB Target Groups](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/load-balancer-target-groups.html)

---

## 🤝 Contributing

When adding new infrastructure:

1. Create module in appropriate directory
2. Add variables with descriptions
3. Add outputs with descriptions
4. Update this README
5. Test in dev environment first
6. Document any new costs

---

## 📞 Support

For infrastructure issues:
1. Check this documentation
2. Review CloudWatch logs
3. Check AWS Service Health Dashboard
4. Review Terraform state

---

**Last Updated**: October 2025

