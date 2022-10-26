output "redshift_db_url_address" {
  value     = aws_redshiftserverless_workgroup.das_workgroup.endpoint[0].address
  sensitive = true
}

output "redshift_db_user" {
  value = var.redshift_db_user
}

output "redshift_db_name" {
  value = var.redshift_db_name
}
