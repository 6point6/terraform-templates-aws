# Generate a random password
resource "random_password" "redshift_password" {
  length           = 16
  special          = true
  override_special = "!#$%&*()-_=+[]{}<>:?"
  min_numeric      = 2
  min_special      = 2
  min_upper        = 2
}

resource "aws_secretsmanager_secret" "redshift_password" {
  name                    = "redshift_db_password"
  description             = "Redshift DB password"
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "redshift_password" {
  secret_id     = aws_secretsmanager_secret.redshift_password.id
  secret_string = random_password.redshift_password.result
  depends_on = [
    random_password.redshift_password
  ]
}
