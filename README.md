# Modules to Deploy

### Templates modules included:

- [IAM](./aws-iam) - Deploy the IAM roles.
- [Glue](./aws-glue-job) - Deploy the Glue job.
- [Redshift](./aws-redshift) - Deploy the Redshift Serverless.
- [S3](./aws-s3) - Deploy S3 Bucket and upload files into.
- [Sagemaker](./aws-sagemaker) - Deploy the Sagemaker with notebook.
- [Secrets Manages](./aws-secrets-manager) - Deploy the Secrets Manages.
- [Common Connections](./common-connections) - Deploy the all the necessary connections.

### To delploy all the modules

```bash
terraform get
terraform init
terraform plan
terraform apply
```

### To destroy all the modules

```bash
terraform destroy
```
