output "random_password" {
  value     = random_password.redshift_password.result
  sensitive = true
}
