resource "aws_ecs_task_definition" "gpu_api" {
  family                   = "${var.app_name}-${var.env}-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["EC2"]
  cpu                      = "4096"  # adjust as needed
  memory                   = "15360"
  execution_role_arn       = var.task_execution_role_arn
  task_role_arn      = var.task_role_arn

  container_definitions = jsonencode([
    {
      name      = "${var.app_name}-${var.env}-container"
      image     = var.container_image
      resourceRequirements = [{ type = "GPU", value = "1" }]
      essential = true
      portMappings = [
        {
          containerPort = 8000
        }
      ]
      environment = [
        {
          name  = "ENV"
          value = var.env
        },
        {
          name  = "MODEL_PATH"
          value = var.model_path
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = "/ecs/${var.app_name}_${var.env}"
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
}

resource "aws_ecs_task_definition" "gpu_api_v2" {
  count                    = var.enable_canary ? 1 : 0
  family                   = "${var.app_name}-${var.env}-task-v2"
  network_mode             = "awsvpc"
  requires_compatibilities = ["EC2"]
  cpu                      = "4096"
  memory                   = "15360"
  execution_role_arn       = var.task_execution_role_arn
  task_role_arn            = var.task_role_arn

  container_definitions = jsonencode([{
    name         = "${var.app_name}-${var.env}-container"
    image        = var.container_image_v2   # new var
    resourceRequirements = [{ type = "GPU", value = "1" }]
    essential    = true
    portMappings = [{ containerPort = 8000 }]
    environment  = [
      { name = "ENV",        value = var.env },
      { name = "MODEL_PATH", value = var.model_path }
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = "/ecs/${var.app_name}_${var.env}"
        awslogs-region        = var.aws_region
        awslogs-stream-prefix = "ecs"
      }
    }
  }])
}