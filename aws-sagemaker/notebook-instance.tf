# ======================================================================================================================
# Notebook 
# ======================================================================================================================


resource "aws_sagemaker_notebook_instance" "notebook_instance" {
  name                = var.notebook_name
  role_arn            = var.iam_role
  instance_type       = var.ec2_instance_type
  security_groups     = [var.aws_default_security_group]
  subnet_id           = var.subnet_1
  platform_identifier = "notebook-al2-v2"
  volume_size         = 16
}
