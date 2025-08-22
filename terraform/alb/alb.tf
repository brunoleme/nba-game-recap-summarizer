resource "aws_lb" "this" {
  name               = "${var.app_name}-${var.env}-alb"
  load_balancer_type = "application"
  subnets            = var.public_subnet_ids
  security_groups    = [var.alb_sg_id]
  idle_timeout       = 60
  internal           = false

  enable_deletion_protection = false
}