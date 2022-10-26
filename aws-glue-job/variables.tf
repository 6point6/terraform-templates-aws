variable "ds_job_language" {
  type    = string
  default = "3"
}

variable "ds_glue_version" {
  type    = string
  default = "3.0"
}

variable "ds_glue_job_name" {
  type    = string
  default = "ds-data-ingestion"
}

variable "ds_glue_worker_type" {
  type    = string
  default = "G.1X"
}

variable "ds_gule_num_workers" {
  type    = number
  default = 10
}

variable "ds_glue_jdbc_connection" {
  type    = string
  default = "DSJDBCConnector"
}

variable "redshift_db_url_address" {
  type = string
}

variable "redshift_db_name" {
  type = string
}

variable "iam_role" {
  type = string
}

variable "redshift_db_user" {
  type = string
}

variable "admin_user_password" {
  type = string
}

variable "subnet_1" {
  type = string
}

variable "subnet_region_1" {
  type = string
}

variable "redshift_role" {
  type = string
}

variable "aws_default_security_group" {
  type = string
}

variable "s3_bucket_name" {
  type = string
}

variable "glue_job_file" {
  type = string
}

variable "source_bucket" {
  type = string
}

variable "source_files" {
  type = string
}
