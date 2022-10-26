variable "vpc_cidr" {
  type    = string
  default = "10.0.0.0/16"
}

variable "subnet_cidr_1" {
  type    = string
  default = "10.0.1.0/24"
}

variable "subnet_cidr_2" {
  type    = string
  default = "10.0.2.0/24"
}

variable "subnet_cidr_3" {
  type    = string
  default = "10.0.3.0/24"
}

variable "subnet_region_1" {
  type    = string
  default = "eu-west-2a"
}

variable "subnet_region_2" {
  type    = string
  default = "eu-west-2b"
}

variable "subnet_region_3" {
  type    = string
  default = "eu-west-2c"
}
