variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "g4dn.xlarge"
}

variable "ssh_key_name" {
  description = "Name of the SSH key pair"
  type        = string
}

variable "inference_image_uri" {
  description = "Full ECR image URI for inference"
  type        = string
}

variable "model_path" {
  description = "S3 path to the model checkpoint"
  type        = string
}
