terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ---------------------------------------------------------------------------
# Networking (minimal: default VPC's public subnets, good enough for a demo/
# portfolio deployment; swap for a proper private-subnet VPC in production)
# ---------------------------------------------------------------------------
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# ---------------------------------------------------------------------------
# S3 — data lake for the raw CSVs (also used as the RDS load source)
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "data_lake" {
  bucket = var.s3_data_bucket
}

resource "aws_s3_bucket_versioning" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_public_access_block" "data_lake" {
  bucket                  = aws_s3_bucket.data_lake.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ---------------------------------------------------------------------------
# RDS — PostgreSQL warehouse (pgvector-capable engine version)
# ---------------------------------------------------------------------------
resource "aws_db_subnet_group" "nexus" {
  name       = "nexus-db-subnet-group"
  subnet_ids = data.aws_subnets.default.ids
}

resource "aws_security_group" "rds" {
  name        = "nexus-rds-sg"
  description = "Allow Postgres from the ECS tasks only"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs_tasks.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_db_instance" "nexus" {
  identifier              = "nexus-db"
  engine                  = "postgres"
  engine_version          = "16.3"
  instance_class          = var.db_instance_class
  allocated_storage       = 20
  max_allocated_storage   = 100
  db_name                 = "nexus_db"
  username                = var.db_master_username
  password                = var.db_master_password
  db_subnet_group_name    = aws_db_subnet_group.nexus.name
  vpc_security_group_ids  = [aws_security_group.rds.id]
  publicly_accessible     = false
  skip_final_snapshot     = true
  backup_retention_period = 7
  storage_encrypted       = true
  # pgvector ships as an extension inside standard Postgres 16 on RDS (CREATE EXTENSION vector;)
}

# ---------------------------------------------------------------------------
# ECR — image repos for the API and dashboard
# ---------------------------------------------------------------------------
resource "aws_ecr_repository" "api" {
  name                 = var.ecr_repo_api
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
}

resource "aws_ecr_repository" "dashboard" {
  name                 = var.ecr_repo_dashboard
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
}

# ---------------------------------------------------------------------------
# ECS Fargate cluster + services
# ---------------------------------------------------------------------------
resource "aws_ecs_cluster" "nexus" {
  name = "nexus-cluster"
}

resource "aws_security_group" "ecs_tasks" {
  name        = "nexus-ecs-tasks-sg"
  description = "Allow inbound from the ALB only"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  ingress {
    from_port       = 8501
    to_port         = 8501
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "alb" {
  name        = "nexus-alb-sg"
  description = "Public HTTP(S) ingress for the ALB"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_lb" "nexus" {
  name               = "nexus-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = data.aws_subnets.default.ids
}

resource "aws_lb_target_group" "api" {
  name        = "nexus-api-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.default.id
  target_type = "ip"
  health_check { path = "/health" }
}

resource "aws_lb_target_group" "dashboard" {
  name        = "nexus-dash-tg"
  port        = 8501
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.default.id
  target_type = "ip"
  health_check { path = "/" }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.nexus.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.dashboard.arn
  }
}

resource "aws_lb_listener_rule" "api" {
  listener_arn = aws_lb_listener.http.arn
  priority     = 100

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }

  condition {
    path_pattern { values = ["/investigate*", "/health*", "/docs*"] }
  }
}

resource "aws_iam_role" "ecs_execution" {
  name = "nexus-ecs-execution-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/nexus-api"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "dashboard" {
  name              = "/ecs/nexus-dashboard"
  retention_in_days = 14
}

resource "aws_ecs_task_definition" "api" {
  family                   = "nexus-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = aws_iam_role.ecs_execution.arn

  container_definitions = jsonencode([{
    name  = "nexus-api"
    image = "${aws_ecr_repository.api.repository_url}:latest"
    portMappings = [{ containerPort = 8000 }]
    environment = [
      { name = "DATABASE_URL", value = "postgresql://${var.db_master_username}:${var.db_master_password}@${aws_db_instance.nexus.address}:5432/nexus_db" },
      { name = "READONLY_DATABASE_URL", value = "postgresql://nexus_readonly:${var.readonly_password}@${aws_db_instance.nexus.address}:5432/nexus_db" },
      { name = "LLM_PROVIDER", value = var.llm_provider },
      { name = "LLM_MODEL", value = var.llm_model },
    ]
    secrets = [
      { name = "ANTHROPIC_API_KEY", valueFrom = var.anthropic_api_key_secret_arn },
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.api.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "api"
      }
    }
  }])
}

resource "aws_ecs_task_definition" "dashboard" {
  family                   = "nexus-dashboard"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "256"
  memory                   = "512"
  execution_role_arn       = aws_iam_role.ecs_execution.arn

  container_definitions = jsonencode([{
    name  = "nexus-dashboard"
    image = "${aws_ecr_repository.dashboard.repository_url}:latest"
    portMappings = [{ containerPort = 8501 }]
    environment = [
      { name = "DATABASE_URL", value = "postgresql://${var.db_master_username}:${var.db_master_password}@${aws_db_instance.nexus.address}:5432/nexus_db" },
      { name = "NEXUS_API_URL", value = "http://${aws_lb.nexus.dns_name}" },
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.dashboard.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "dashboard"
      }
    }
  }])
}

resource "aws_ecs_service" "api" {
  name            = "nexus-api"
  cluster         = aws_ecs_cluster.nexus.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.api_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name    = "nexus-api"
    container_port    = 8000
  }

  depends_on = [aws_lb_listener.http]
}

resource "aws_ecs_service" "dashboard" {
  name            = "nexus-dashboard"
  cluster         = aws_ecs_cluster.nexus.id
  task_definition = aws_ecs_task_definition.dashboard.arn
  desired_count   = var.dashboard_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.dashboard.arn
    container_name    = "nexus-dashboard"
    container_port    = 8501
  }

  depends_on = [aws_lb_listener.http]
}
