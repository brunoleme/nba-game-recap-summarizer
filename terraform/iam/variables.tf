variable "name_prefix" {
  type        = string
  description = "Prefix for IAM role names"
}

variable "model_bucket_prefix" {
  type        = string
  description = "Prefix for S3 buckets that store models"
  default     = "bl-portfolio-ml-sagemaker-"
}

variable "kms_key_arn" {
  type        = string
  description = "Optional KMS key ARN for model bucket encryption"
  default     = null
}