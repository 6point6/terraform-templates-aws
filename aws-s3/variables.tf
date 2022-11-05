variable "code_base_path" {
  type    = string
  default = "~/Documents/DAS/Repos/terraform-modules/aws-s3/files/"
}

variable "tools_bucket" {
  type    = string
  default = "ds-glue-tools"
}

variable "glue_job_file" {
  type    = string
  default = "glue_job_file.py"
}
