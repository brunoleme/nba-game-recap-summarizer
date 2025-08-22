resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${var.app_name}_${var.env}"
  retention_in_days = 14
}