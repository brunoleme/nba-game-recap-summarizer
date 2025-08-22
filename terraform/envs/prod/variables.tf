variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "inference_image_uri" {
  description = "Full URI of the ECR image (stable version) to deploy to ECS"
  type        = string
}

variable "inference_image_v2_uri" {
  description = "Full URI of the ECR image (candidate version) to deploy to ECS"
  type        = string
}

variable "ssh_key_name" {
  description = "EC2 SSH key name"
  type        = string
  default     = "my-ecs-key"
}

variable "env" {
  description = "Deployment environment"
  type        = string
}

variable "app_name" {
  description = "Name of the application"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
}

variable "model_path" {
  type        = string
  description = "S3 path to packaged model"
}