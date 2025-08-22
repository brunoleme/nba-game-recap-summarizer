variable "app_name" {
  type        = string
  description = "App name for naming resources"
}

variable "vpc_id" {
  type        = string
  description = "VPC ID"
}

variable "public_subnet_ids" {
  type        = list(string)
  description = "List of public subnet IDs for ALB"
}

variable "alb_sg_id" {
  type        = string
  description = "Security group ID for ALB"
}

variable "env" {
  type        = string
  description = "Environment (dev, staging or prod)"
}

variable "enable_canary" {
  type        = bool
  default = false
}

variable "canary_weight_v1" {
  type = number
  default = 90
  validation {
    condition     = var.canary_weight_v1 >= 0 && var.canary_weight_v1 <= 100
    error_message = "canary_weight_v1 must be 0–100."
  }
}

variable "canary_weight_v2" {
  type = number
  default = 10
  validation {
    condition     = var.canary_weight_v2 >= 0 && var.canary_weight_v2 <= 100
    error_message = "canary_weight_v2 must be 0–100."
  }
}

locals {
  canary_weight_sum_valid = var.canary_weight_v1 + var.canary_weight_v2 == 100
}

resource "null_resource" "validate_canary_weights" {
  # keep or drop triggers; they're not needed for the check to work
  triggers = { sum_ok = local.canary_weight_sum_valid ? "ok" : "bad" }

  lifecycle {
    precondition {
      condition     = local.canary_weight_sum_valid
      error_message = "canary_weight_v1 + canary_weight_v2 must equal 100."
    }
  }
}