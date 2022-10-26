import sys
from awsglue.utils import getResolvedOptions
import redshift_connector


# CONFIGURATION
RS_SCHEMA = "public"
RS_TABLE = "db_data"
DELIMITER = "\n"
REGION = "us-east-2"

# ARGUMENTS
args = getResolvedOptions(
    sys.argv, ["JOB_NAME", "rs_host", "rd_db_name", "rd_db_user", "source_bucket", "source_files", "rs_password", "iam_role_redshift"])

RS_HOST = args["rs_host"]
RS_DATABASE = args["rd_db_name"]
RS_USER = args["rd_db_user"]
S3_BUCKET = args["source_bucket"]
S3_OBJECT = args["source_files"]
RS_PASSWORD = args["rs_password"]
RS_IAM_ROLE = args["iam_role_redshift"]


conn = redshift_connector.connect(
    host=RS_HOST,
    database=RS_DATABASE,
    user=RS_USER,
    password=RS_PASSWORD,
    ssl=False
)
cursor = conn.cursor()

cursor.execute("set statement_timeout = 360000")

cursor.execute("CREATE TABLE IF NOT EXISTS %s (commons SUPER);" % (RS_TABLE))

copy_query = "COPY %s.%s from 's3://%s/%s' iam_role '%s' delimiter '%s'  GZIP REGION '%s';" % (
    RS_SCHEMA, RS_TABLE, S3_BUCKET, S3_OBJECT, RS_IAM_ROLE, DELIMITER, REGION)

cursor.execute(copy_query)
conn.commit()
cursor.close()
conn.close()
