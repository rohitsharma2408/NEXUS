variable "aws_region" {
  default = "us-east-1"
}

variable "s3_data_bucket" {
  description = "Globally-unique S3 bucket name for the raw CSV data lake"
  type        = string
}

variable "db_instance_class" {
  default = "db.t4g.micro"
}

variable "db_master_username" {
  default = "nexus_user"
}

variable "db_master_password" {
  description = "Master password for RDS (use a secret manager / tfvars, never commit this)"
  type        = string
  sensitive   = true
}

variable "readonly_password" {
  description = "Password for the nexus_readonly role used by the agentic layer"
  type        = string
  sensitive   = true
}

variable "ecr_repo_api" {
  default = "nexus-api"
}

variable "ecr_repo_dashboard" {
  default = "nexus-dashboard"
}

variable "llm_provider" {
  default = "anthropic"
}

variable "llm_model" {
  default = "claude-sonnet-4-6"
}

variable "anthropic_api_key_secret_arn" {
  description = "ARN of a Secrets Manager secret holding the Anthropic API key"
  type        = string
}

variable "api_desired_count" {
  default = 1
}

variable "dashboard_desired_count" {
  default = 1
}
