variable "instance_type" {
  description = "EC2 instance type for ECS GPU nodes"
  type        = string
  default     = "g4dn.xlarge"
}

variable "asg_min_size" {
  default = 1
}

variable "asg_max_size" {
  default = 1
}

variable "asg_desired_capacity" {
  default = 1
}

variable "ssh_key_name" {
  description = "SSH key pair name"
  type        = string
}

variable "instance_profile_name" {
  description = "IAM instance profile name"
  type        = string
}

variable "ecs_instance_sg_id" {
  description = "Security group for ECS EC2"
  type        = string
}

variable "private_subnet_ids" {
  type        = list(string)
  description = "Private subnet IDs for ECS instances"
}

variable "app_name" {
  description = "Application name"
  type        = string
}

variable "env" {
  description = "Deployment environment"
  type        = string
}

variable "container_image" {
  description = "Full ECR image URI for inference"
  type        = string
}

variable "container_image_v2" {
  description = "ECR image URI for canary"
  type        = string
  default     = null
}

variable "model_path" {
  description = "Path to load the model inside container"
  type        = string
  default     = "/app/model"
}

variable "task_execution_role_arn" {
  description = "IAM role ARN for ECS task execution"
  type        = string
}

variable "ecs_service_sg_id" {
  description = "Security group for ECS service"
  type        = string
}

variable "target_group_v1_arn" {
  description = "Target group v1 ARN from ALB"
  type        = string
}

variable "target_group_v2_arn" {
  description = "Target group v2 ARN from ALB"
  type        = string
  default = null
}

variable "enable_canary" {
  type        = bool
  default = false
}

variable "aws_region" {
  description = "AWS region"
  type        = string
}

variable "cluster_name" {
  description = "Name of the ECS Cluster"
  type        = string
}

variable "task_role_arn" {
  type = string
}

variable "root_volume_size" {
  description = "Root EBS volume size (GiB) for ECS GPU instances"
  type        = number
  default     = 150
}
