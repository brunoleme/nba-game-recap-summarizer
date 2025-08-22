resource "aws_ecs_cluster" "main" {
  name = "${var.app_name}-${var.env}-cluster"
}

output "ecs_cluster_id" {
  value = aws_ecs_cluster.main.id
}