variable "notebook_name" {
  type    = string
  default = "test-notebook"
}

variable "ec2_instance_type" {
  type    = string
  default = "ml.t3.xlarge"
}

variable "iam_role" {
  type = string
}

variable "aws_default_security_group" {
  type = string
}

variable "subnet_1" {
  type = string
}
