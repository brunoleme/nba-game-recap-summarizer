provider "aws" {
  region = var.aws_region
}

module "network" {
  source = "../../network"
  app_name = var.app_name
  env      = var.env
}

module "alb" {
  source = "../../alb"
  app_name          = var.app_name
  env               = var.env
  vpc_id            = module.network.vpc_id
  public_subnet_ids = module.network.public_subnet_ids
  alb_sg_id         = module.network.alb_sg_id
  enable_canary     = false
}

module "ecs" {
  source = "../../ecs"

  app_name                 = var.app_name
  env                      = var.env
  aws_region               = var.aws_region
  cluster_name = "${var.app_name}-${var.env}-cluster"

  # VPC & Subnet setup
  private_subnet_ids       = module.network.private_subnet_ids

  # Security groups
  ecs_instance_sg_id  = module.network.ecs_sg_id
  ecs_service_sg_id   = module.network.ecs_sg_id

  # IAM
  instance_profile_name    = module.iam.instance_profile_name
  task_execution_role_arn  = module.iam.task_execution_role_arn
  task_role_arn           = module.iam.task_role_arn
  ssh_key_name             = var.ssh_key_name

  # ECS Capacity
  instance_type            = var.instance_type
  asg_min_size             = 1
  asg_max_size             = 2
  asg_desired_capacity     = 1

  # Image
  container_image          = var.inference_image_uri

  # setting MODEL_PATH env var
  model_path = var.model_path

  # ALB
  enable_canary       = false
  target_group_v1_arn = module.alb.target_group_v1_arn
  # target_group_v2_arn not required
}

module "iam" {
  source      = "../../iam"
  name_prefix = var.app_name
}