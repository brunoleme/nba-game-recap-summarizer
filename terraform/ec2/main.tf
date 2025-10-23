terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
  }
  
  backend "s3" {
    bucket         = "nba-recap-model-tf-state-bucket"
    key            = "ec2/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-locks"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region
}

# Random ID for unique resource naming
resource "random_id" "suffix" {
  byte_length = 4
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

# Try to find Deep Learning AMI with multiple patterns
data "aws_ami" "gpu_ami" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = [
      "Deep Learning AMI GPU PyTorch 2.* (Ubuntu 20.04) *",
      "Deep Learning AMI GPU PyTorch 2.* (Ubuntu 22.04) *",
      "Deep Learning AMI GPU PyTorch 3.* (Ubuntu 20.04) *",
      "Deep Learning AMI GPU PyTorch 3.* (Ubuntu 22.04) *",
      "Deep Learning AMI GPU PyTorch 2.* (Ubuntu 20.04) *",
      "Deep Learning AMI GPU PyTorch 2.* (Ubuntu 22.04) *"
    ]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

# Alternative Deep Learning AMI search with different naming pattern
# This is optional and will be skipped if no matching AMIs are found
data "aws_ami" "gpu_ami_alt" {
  count       = 0  # Disable this for now since the pattern doesn't exist
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = [
      "Deep Learning AMI (Ubuntu 20.04) Version *",
      "Deep Learning AMI (Ubuntu 22.04) Version *"
    ]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

# Fallback to standard Ubuntu AMI if Deep Learning AMI is not available
data "aws_ami" "ubuntu_ami" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = [
      "ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*",
      "ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"
    ]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

# Select the best available AMI with fallback logic
locals {
  selected_ami = data.aws_ami.gpu_ami.id != "" ? data.aws_ami.gpu_ami.id : data.aws_ami.ubuntu_ami.id
}

# VPC and Networking
resource "aws_vpc" "main" {
  cidr_block           = "10.1.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "${var.environment}-ec2-vpc"
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.environment}-ec2-igw"
  }
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.1.1.0/24"
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.environment}-ec2-public-subnet"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "${var.environment}-ec2-public-rt"
  }
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

# Security Group
resource "aws_security_group" "ec2_sg" {
  name_prefix = "${var.environment}-ec2-sg"
  vpc_id      = aws_vpc.main.id

  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # HTTP access for the API
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # All outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.environment}-ec2-sg"
  }
}

# IAM Role for EC2
resource "aws_iam_role" "ec2_role" {
  name = "${var.environment}-ec2-role-${random_id.suffix.hex}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "ec2_policy" {
  name = "${var.environment}-ec2-policy-${random_id.suffix.hex}"
  role = aws_iam_role.ec2_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::nba-recap-summarization-model-*",
          "arn:aws:s3:::nba-recap-summarization-model-*/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${var.environment}-ec2-profile-${random_id.suffix.hex}"
  role = aws_iam_role.ec2_role.name
}

# User data script
locals {
  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    ecr_repository_uri = var.inference_image_uri
    ecr_registry       = "${split("/", var.inference_image_uri)[0]}"
    model_path         = var.model_path
    environment        = var.environment
    aws_region         = var.aws_region
    hf_token           = var.hf_token
  }))
}

# EC2 Instance
resource "aws_instance" "inference" {
  ami                    = local.selected_ami
  instance_type          = var.instance_type
  key_name               = var.ssh_key_name
  vpc_security_group_ids = [aws_security_group.ec2_sg.id]
  subnet_id              = aws_subnet.public.id
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name

  # GPU instance requirements
  root_block_device {
    volume_type = "gp3"
    volume_size = 200
    encrypted   = true
  }

  user_data = local.user_data

  tags = {
    Name        = "${var.environment}-inference-instance"
    Environment = var.environment
  }

  # Ensure instance is running before proceeding
  lifecycle {
    create_before_destroy = true
  }
}

# Outputs
output "instance_id" {
  description = "ID of the EC2 instance"
  value       = aws_instance.inference.id
}

output "instance_public_ip" {
  description = "Public IP address of the EC2 instance"
  value       = aws_instance.inference.public_ip
}

output "instance_public_dns" {
  description = "Public DNS name of the EC2 instance"
  value       = aws_instance.inference.public_dns
}

output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "subnet_id" {
  description = "ID of the public subnet"
  value       = aws_subnet.public.id
}

output "selected_ami_id" {
  description = "ID of the selected AMI"
  value       = local.selected_ami
}

output "ami_name" {
  description = "Name of the selected AMI"
  value       = data.aws_ami.gpu_ami.id != "" ? data.aws_ami.gpu_ami.name : data.aws_ami.ubuntu_ami.name
}

output "ami_type" {
  description = "Type of AMI selected (GPU or Ubuntu)"
  value       = data.aws_ami.gpu_ami.id != "" ? "GPU" : "Ubuntu"
}
