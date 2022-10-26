# ======================================================================================================================
# S3 
# ======================================================================================================================

# Local variables based on user inputs
locals {
  artifacts = {
    import_module_path  = join("", [var.code_base_path, "asn1crypto-1.5.1-py2.py3-none-any.whl"]),
    import_module_path2 = join("", [var.code_base_path, "scramp-1.4.1-py3-none-any.whl"]),
    import_module_path3 = join("", [var.code_base_path, "redshift_connector-2.0.909-py3-none-any.whl"]),
    requirements        = join("", [var.code_base_path, var.requirements]),
    notebook_ingestion  = join("", [var.code_base_path, var.notebook_ingestion]),
    glue_job_file       = join("", [var.code_base_path, var.glue_job_file])
  }
}

# ------- S3 Config -------

# create S3 buckets
resource "aws_s3_bucket" "create_buckets" {
  bucket = var.tools_bucket
  # acl           = "private"
  force_destroy = true
}

# Configure bucket ACLs (mainly block public access)
resource "aws_s3_bucket_public_access_block" "configure_bucket_acls" {
  bucket                  = var.tools_bucket
  block_public_acls       = true
  block_public_policy     = true
  restrict_public_buckets = true
  ignore_public_acls      = true

  depends_on = [
    aws_s3_bucket.create_buckets
  ]
}

# Upload tool artifacts
resource "aws_s3_object" "upload_tool_artifacts" {

  for_each = local.artifacts
  bucket   = var.tools_bucket
  key      = join("", [element(split("/", each.value), length(split("/", each.value)) - 1)])
  source   = each.value
  etag     = filemd5(each.value)
  depends_on = [
    aws_s3_bucket.create_buckets,
    aws_s3_bucket_public_access_block.configure_bucket_acls
  ]
}
