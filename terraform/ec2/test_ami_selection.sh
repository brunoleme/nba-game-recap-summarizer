#!/bin/bash

# Test script to verify AMI selection before deployment
# This script tests the AMI data sources without creating any resources

set -e

echo "🔍 Testing AMI selection for EC2 deployment..."
echo "=============================================="

# Change to the terraform directory
cd "$(dirname "$0")"

# Initialize Terraform
echo "📦 Initializing Terraform..."
terraform init -reconfigure

# Test the AMI data sources
echo "🔍 Testing AMI data sources..."
echo ""

# Test GPU AMI
echo "1. Testing Deep Learning AMI (GPU PyTorch)..."
terraform console <<EOF
data.aws_ami.gpu_ami
EOF

echo ""

# Test alternative GPU AMI
echo "2. Testing Alternative Deep Learning AMI..."
terraform console <<EOF
data.aws_ami.gpu_ami_alt
EOF

echo ""

# Test Ubuntu AMI
echo "3. Testing Ubuntu AMI (fallback)..."
terraform console <<EOF
data.aws_ami.ubuntu_ami
EOF

echo ""

# Test selected AMI
echo "4. Testing selected AMI logic..."
terraform console <<EOF
local.selected_ami
EOF

echo ""
echo "✅ AMI selection test completed!"
echo "If any of the above commands failed, it means the AMI patterns need to be updated."
echo "The deployment will use the first available AMI in the fallback chain."
