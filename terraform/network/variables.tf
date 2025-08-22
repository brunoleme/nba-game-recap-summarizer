variable "app_name" {
  type        = string
  description = "Application name for naming network resources"
}

variable "env" {
  type        = string
  description = "Environment (dev|staging|prod)"
}
