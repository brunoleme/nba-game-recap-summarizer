resource "aws_lb_listener_rule" "canary_weighted" {
  count        = var.enable_canary ? 1 : 0
  listener_arn = aws_lb_listener.frontend.arn
  priority     = 100

  action {
    type = "forward"
    forward {
      target_group {
        arn    = aws_lb_target_group.v1.arn
        weight = var.canary_weight_v1
      }
      target_group {
        arn    = aws_lb_target_group.v2[0].arn
        weight = var.canary_weight_v2
      }
      stickiness {
        enabled = true
        duration = 60
      }
    }
  }

  condition {
    path_pattern {
      values = ["/*"]
    }
  }
}