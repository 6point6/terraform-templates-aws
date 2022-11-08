variable "tools_bucket" {
  type    = string
  default = "ds-glue-tools"
}

variable "glue_job_file" {
  type    = string
  default = "glue_job_file.py"
}

variable "path" {
  type    = string
  default = "./aws-s3/files/"
}
