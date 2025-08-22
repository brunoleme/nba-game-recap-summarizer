resource "aws_launch_template" "gpu" {
  name_prefix   = "${var.cluster_name}-${var.env}-gpu-"
  image_id      = data.aws_ami.amazon_linux_gpu.id
  instance_type = var.instance_type

  key_name = var.ssh_key_name

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = var.root_volume_size
      volume_type           = "gp3"
      delete_on_termination = true
      encrypted             = true
    }
  }

  user_data = base64encode(<<-EOF
              #!/bin/bash
              echo ECS_CLUSTER=${aws_ecs_cluster.main.name} >> /etc/ecs/ecs.config
              echo ECS_ENABLE_GPU_SUPPORT=true >> /etc/ecs/ecs.config
              EOF
  )

  network_interfaces {
    associate_public_ip_address = false
    security_groups             = [var.ecs_instance_sg_id]
  }

  iam_instance_profile {
    name = var.instance_profile_name
  }
}

data "aws_ami" "amazon_linux_gpu" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-ecs-gpu-hvm-*-x86_64-ebs"]
  }
}

resource "aws_autoscaling_group" "gpu" {
  name_prefix        = "${var.cluster_name}-${var.env}-gpu-asg-"
  desired_capacity   = var.asg_desired_capacity
  min_size           = var.asg_min_size
  max_size           = var.asg_max_size
  vpc_zone_identifier = var.private_subnet_ids

  launch_template {
    id      = aws_launch_template.gpu.id
    version = "$Latest"
  }

  instance_refresh {
    strategy = "Rolling"
    triggers = ["launch_template"]
  }
  
  tag {
    key                 = "AmazonECSCluster"
    value               = aws_ecs_cluster.main.name
    propagate_at_launch = true
  }
}

resource "aws_ecs_capacity_provider" "gpu" {
  name = "${var.cluster_name}-${var.env}-gpu"

  auto_scaling_group_provider {
    auto_scaling_group_arn = aws_autoscaling_group.gpu.arn

    managed_scaling {
      status                    = "ENABLED"
      target_capacity           = 80
      minimum_scaling_step_size = 1
      maximum_scaling_step_size = 1000
      instance_warmup_period    = 300
    }

    managed_termination_protection = "DISABLED"
  }
}

resource "aws_ecs_cluster_capacity_providers" "gpu" {
  cluster_name = aws_ecs_cluster.main.name

  capacity_providers = [aws_ecs_capacity_provider.gpu.name]

  default_capacity_provider_strategy {
    capacity_provider = aws_ecs_capacity_provider.gpu.name
    weight            = 1
    base              = 1
  }
}
