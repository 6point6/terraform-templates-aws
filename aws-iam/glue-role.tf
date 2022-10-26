# Defining the Glue "Assume Role" policy
data "aws_iam_policy_document" "glue_assume_role_policy" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type = "Service"
      identifiers = [
        "glue.amazonaws.com",
        "ec2.amazonaws.com",
        "s3.amazonaws.com"
      ]
    }
  }
}

# Defining the Glue IAM role
resource "aws_iam_role" "glue_iam_role" {
  name               = "glue_role"
  assume_role_policy = data.aws_iam_policy_document.glue_assume_role_policy.json
}


# Attaching the AWS default policy, "AmazonS3FullAccess"
resource "aws_iam_policy_attachment" "s3_full_access_attach" {
  name       = "s3_full_access_attach"
  roles      = [aws_iam_role.glue_iam_role.name]
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

# Attaching the AWS default policy, "AmazonRDSFullAccess"
resource "aws_iam_policy_attachment" "rds_full_access_attach" {
  name       = "rds_full_access_attach"
  roles      = [aws_iam_role.glue_iam_role.name]
  policy_arn = "arn:aws:iam::aws:policy/AmazonRDSFullAccess"
}

# Attaching the AWS default policy, "AmazonRDSDataFullAccess"
resource "aws_iam_policy_attachment" "rds_data_full_access_attach" {
  name       = "rds_data_full_access_attach"
  roles      = [aws_iam_role.glue_iam_role.name]
  policy_arn = "arn:aws:iam::aws:policy/AmazonRDSDataFullAccess"
}

# Attaching the AWS default policy, "AWSGlueServiceRole"
resource "aws_iam_policy_attachment" "glue_full_access_attach" {
  name       = "glue_full_access_attach"
  roles      = [aws_iam_role.glue_iam_role.name]
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole"
}
