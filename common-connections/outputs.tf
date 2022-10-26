output "aws_default_security_group" {
  value = aws_default_security_group.security_group.id
}

output "subnet_1" {
  value = aws_subnet.subnet_1.id
}

output "subnet_2" {
  value = aws_subnet.subnet_2.id
}

output "subnet_3" {
  value = aws_subnet.subnet_3.id
}

output "subnet_region_1" {
  value = var.subnet_region_1
}
