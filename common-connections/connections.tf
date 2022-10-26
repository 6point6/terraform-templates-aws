# ======================================================================================================================
# CONNECTIOONS
# ======================================================================================================================

# Create VPC
resource "aws_vpc" "common_vpc" {
  cidr_block       = var.vpc_cidr
  instance_tenancy = "default"
  tags = {
    Name = "common-vpc"
  }
}

# Create Endpoint
resource "aws_vpc_endpoint" "s3_endpoint" {
  vpc_id            = aws_vpc.common_vpc.id
  service_name      = "com.amazonaws.eu-west-2.s3"
  vpc_endpoint_type = "Gateway"
  tags = {
    Environment = "s3_endpoint"
  }
  depends_on = [
    aws_vpc.common_vpc
  ]
}

# Create Route Tables
resource "aws_route_table" "ds_route_table" {
  vpc_id = aws_vpc.common_vpc.id

  depends_on = [
    aws_vpc.common_vpc
  ]
}

# Create Endpoint Route Association
resource "aws_vpc_endpoint_route_table_association" "ds_route_table_association" {
  route_table_id  = aws_route_table.ds_route_table.id
  vpc_endpoint_id = aws_vpc_endpoint.s3_endpoint.id
}

# Create Route Association
resource "aws_route_table_association" "ds_route_table_association" {
  subnet_id      = aws_subnet.subnet_1.id
  route_table_id = aws_route_table.ds_route_table.id
}

# Internet Gateway (for VPC)
resource "aws_internet_gateway" "common_vpc_gw" {
  vpc_id = aws_vpc.common_vpc.id
  depends_on = [
    aws_vpc.common_vpc
  ]
}

# Security Group for VPC
resource "aws_default_security_group" "security_group" {
  vpc_id = aws_vpc.common_vpc.id

  ingress {
    self        = true
    from_port   = 5439
    to_port     = 5439
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    self        = true
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = {
    Name = "sg"
  }
  depends_on = [
    aws_vpc.common_vpc
  ]
}

# Subnets
resource "aws_subnet" "subnet_1" {
  vpc_id                  = aws_vpc.common_vpc.id
  cidr_block              = var.subnet_cidr_1
  availability_zone       = var.subnet_region_1
  map_public_ip_on_launch = "true"
  tags = {
    Name = "subnet-1"
  }
  depends_on = [
    aws_vpc.common_vpc
  ]
}
resource "aws_subnet" "subnet_2" {
  vpc_id                  = aws_vpc.common_vpc.id
  cidr_block              = var.subnet_cidr_2
  availability_zone       = var.subnet_region_2
  map_public_ip_on_launch = "true"
  tags = {
    Name = "subnet-2"
  }
  depends_on = [
    aws_vpc.common_vpc
  ]
}
resource "aws_subnet" "subnet_3" {
  vpc_id                  = aws_vpc.common_vpc.id
  cidr_block              = var.subnet_cidr_3
  availability_zone       = var.subnet_region_3
  map_public_ip_on_launch = "true"
  tags = {
    Name = "subnet-3"
  }
  depends_on = [
    aws_vpc.common_vpc
  ]
}
