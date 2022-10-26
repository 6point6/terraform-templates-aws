# Defining the Redshift "Assume Role" policy
data "aws_iam_policy_document" "redshift_assume_role_policy" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["redshift.amazonaws.com"]
    }
  }
}

# Defining the Redshift IAM role
resource "aws_iam_role" "redshift_iam_role" {
  name               = "redshift_role"
  assume_role_policy = data.aws_iam_policy_document.redshift_assume_role_policy.json
}


# Attaching the AWS default policy, "AmazonRedshiftAllCommandsFullAccess"
resource "aws_iam_policy_attachment" "redshift_full_access_attach" {
  name       = "redshift_full_access_attach"
  roles      = [aws_iam_role.redshift_iam_role.name]
  policy_arn = "arn:aws:iam::aws:policy/AmazonRedshiftAllCommandsFullAccess"
}

# Attaching the AWS default policy, "AmazonRedshiftAllCommandsFullAccess"
resource "aws_iam_policy_attachment" "redshift_command_access_attach" {
  name       = "redshift_command_access_attach"
  roles      = [aws_iam_role.redshift_iam_role.name]
  policy_arn = "arn:aws:iam::425768274696:policy/service-role/AmazonRedshift-CommandsAccessPolicy-20220914T122554"
}
