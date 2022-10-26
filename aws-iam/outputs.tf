output "redshift_iam_role" {
  value = aws_iam_role.redshift_iam_role.arn
}

output "notebook_iam_role" {
  value = aws_iam_role.notebook_iam_role.arn
}

output "glue_iam_role" {
  value = aws_iam_role.glue_iam_role.arn
}
