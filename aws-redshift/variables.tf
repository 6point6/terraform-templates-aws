variable "redshiftserverless_namespace" {
  type    = string
  default = "ds-namespace"
}

variable "redshiftserverless_workgroup" {
  type    = string
  default = "ds-workgroup"
}

variable "redshift_db_name" {
  type    = string
  default = "ds_redshift_db"
}

variable "redshift_db_user" {
  type    = string
  default = "postgres"
}

variable "iam_role" {
  type = string
}

variable "admin_user_password" {
  type = string
}

variable "subnet_1" {
  type = string
}

variable "subnet_2" {
  type = string
}

variable "subnet_3" {
  type = string
}
