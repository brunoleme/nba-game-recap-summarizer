output "alb_dns_name" {
  value = aws_lb.this.dns_name
}

output "target_group_v1_arn" {
  value = aws_lb_target_group.v1.arn
}

output "target_group_v2_arn" {
  value = var.enable_canary ? aws_lb_target_group.v2[0].arn : null
}