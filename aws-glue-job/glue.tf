# ======================================================================================================================
# Glue 
# ======================================================================================================================


# Create Glue connection to Redshift
resource "aws_glue_connection" "ds_glue_jdbc_connection" {
  name = var.ds_glue_jdbc_connection
  connection_properties = {
    JDBC_CONNECTION_URL = "jdbc:redshift://${var.redshift_db_url_address}:5439/${var.redshift_db_name}"
    USERNAME            = var.redshift_db_user
    PASSWORD            = var.admin_user_password
  }
  physical_connection_requirements {
    availability_zone      = var.subnet_region_1
    security_group_id_list = [var.aws_default_security_group]
    subnet_id              = var.subnet_1
  }
}

# Create a glue job to ingest data into redshift
resource "aws_glue_job" "ds_glue_job_ingestion" {
  name              = var.ds_glue_job_name
  role_arn          = var.iam_role
  glue_version      = var.ds_glue_version
  connections       = [var.ds_glue_jdbc_connection]
  worker_type       = var.ds_glue_worker_type
  number_of_workers = var.ds_gule_num_workers

  command {
    script_location = join("", ["s3://", var.s3_bucket_name, "/", var.glue_job_file])
    python_version  = var.ds_job_language
  }

  default_arguments = {
    "--extra-py-files"                   = join(",", [join("", ["s3://", var.s3_bucket_name, "/", "asn1crypto-1.5.1-py2.py3-none-any.whl"])], [join("", ["s3://", var.s3_bucket_name, "/", "scramp-1.4.1-py3-none-any.whl"])], [join("", ["s3://", var.s3_bucket_name, "/", "redshift_connector-2.0.909-py3-none-any.whl"])])
    "--rs_host"                          = var.redshift_db_url_address
    "--rd_db_name"                       = var.redshift_db_name
    "--rd_db_user"                       = var.redshift_db_user
    "--source_bucket"                    = var.source_bucket
    "--source_files"                     = "${var.source_files}/"
    "--rs_password"                      = var.admin_user_password
    "--iam_role_redshift"                = var.redshift_role
    "--enable-continuous-cloudwatch-log" = "true"
    "--enable-continuous-log-filter"     = "true"
    "--enable-job-insights"              = "true"
  }

  depends_on = [
    aws_glue_connection.ds_glue_jdbc_connection
  ]
}

# Create a glue trigger
resource "aws_glue_trigger" "ds_glue_job_trigger" {
  name = "ds_glue_job_trigger"
  type = "ON_DEMAND"

  actions {
    job_name = aws_glue_job.ds_glue_job_ingestion.name
  }
  depends_on = [
    aws_glue_job.ds_glue_job_ingestion
  ]
}
