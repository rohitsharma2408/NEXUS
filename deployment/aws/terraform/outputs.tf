output "alb_dns_name" {
  value = aws_lb.nexus.dns_name
}

output "rds_endpoint" {
  value = aws_db_instance.nexus.address
}

output "ecr_api_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "ecr_dashboard_repository_url" {
  value = aws_ecr_repository.dashboard.repository_url
}

output "s3_data_bucket" {
  value = aws_s3_bucket.data_lake.bucket
}
