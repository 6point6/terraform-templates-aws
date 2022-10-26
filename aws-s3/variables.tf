variable "code_base_path" {
  type    = string
  default = "~/Documents/Visual Studio/terraform-modules/aws-s3/files/"
}

variable "tools_bucket" {
  type    = string
  default = "ds-glue-tools"
}

variable "requirements" {
  type    = string
  default = "requirements.txt"
}

variable "notebook_ingestion" {
  type    = string
  default = "ingestion.ipynb"
}

variable "glue_job_file" {
  type    = string
  default = "glue_job_file.py"
}
