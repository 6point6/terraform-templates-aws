# ======================================================================================================================
# Redshift 
# ======================================================================================================================

# Redshift Serveless Name Space
resource "aws_redshiftserverless_namespace" "das_namespace" {
  namespace_name      = var.redshiftserverless_namespace
  iam_roles           = [var.iam_role]
  db_name             = var.redshift_db_name
  admin_username      = var.redshift_db_user
  admin_user_password = var.admin_user_password
}

# Redshift Serveless Workgrout
resource "aws_redshiftserverless_workgroup" "das_workgroup" {
  namespace_name = var.redshiftserverless_namespace
  workgroup_name = var.redshiftserverless_workgroup
  subnet_ids     = aws_redshift_subnet_group.redshift_subnet_group.subnet_ids
  depends_on = [
    aws_redshiftserverless_namespace.das_namespace,
    aws_redshift_subnet_group.redshift_subnet_group
  ]
}

# Redshift Subnet Group
resource "aws_redshift_subnet_group" "redshift_subnet_group" {
  name       = "redshift-subnet-group"
  subnet_ids = ["${var.subnet_1}", "${var.subnet_2}", "${var.subnet_3}"]
  tags = {
    environment = "dev"
    Name        = "redshift-subnet-group"
  }
}
