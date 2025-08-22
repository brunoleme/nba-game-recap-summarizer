resource "aws_ecs_service" "v1" {
  name = "${var.app_name}-${var.env}-v1-service"
  cluster         = aws_ecs_cluster.main.id
  desired_count   = 1
  task_definition = aws_ecs_task_definition.gpu_api.arn
  health_check_grace_period_seconds = 60
  enable_execute_command = true

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [var.ecs_service_sg_id]
    assign_public_ip = false
  }

  capacity_provider_strategy {
    capacity_provider = aws_ecs_capacity_provider.gpu.name
    weight            = 1
    base              = 1
  }

  load_balancer {
    target_group_arn = var.target_group_v1_arn
    container_name   = "${var.app_name}-${var.env}-container"
    container_port   = 8000
  }
}

resource "aws_ecs_service" "v2" {
  count           = var.enable_canary ? 1 : 0
  name = "${var.app_name}-${var.env}-v2-service"
  cluster         = aws_ecs_cluster.main.id
  desired_count   = 1
  task_definition = aws_ecs_task_definition.gpu_api_v2[0].arn
  health_check_grace_period_seconds = 60
  enable_execute_command = true

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [var.ecs_service_sg_id]
    assign_public_ip = false
  }

  capacity_provider_strategy {
    capacity_provider = aws_ecs_capacity_provider.gpu.name
    weight            = 1
    base              = 1
  }

  load_balancer {
    target_group_arn = var.target_group_v2_arn
    container_name   = "${var.app_name}-${var.env}-container"
    container_port   = 8000
  }
}