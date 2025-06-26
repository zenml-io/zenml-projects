## ☁️ Cloud Deployment

CreditScorer supports storing artifacts remotely and executing pipelines on cloud infrastructure. For this example, we'll use AWS, but you can use any cloud provider you want. You can also refer to the [AWS Integration Guide](https://docs.zenml.io/stacks/popular-stacks/aws-guide) for detailed instructions.

### AWS Setup

1. **Install required integrations**:

   ```bash
   zenml integration install aws s3
   ```

2. **Set up your AWS credentials**:

   - Create an IAM role with appropriate permissions (S3, ECR, SageMaker)
   - Configure your role ARN and region

3. **Register an AWS service connector**:

   ```bash
   zenml service-connector register aws_connector \
     --type aws \
     --auth-method iam-role \
     --role_arn=<ROLE_ARN> \
     --region=<YOUR_REGION> \
     --aws_access_key_id=<YOUR_ACCESS_KEY_ID> \
     --aws_secret_access_key=<YOUR_SECRET_ACCESS_KEY>
   ```

4. **Configure stack components**:

   a. **S3 Artifact Store**:

   ```bash
   zenml artifact-store register s3_artifact_store \
     -f s3 \
     --path=s3://<YOUR_BUCKET_NAME> \
     --connector aws_connector
   ```

   b. **SageMaker Orchestrator** (Optional):

   ```bash
   zenml orchestrator register sagemaker_orchestrator \
     --flavor=sagemaker \
     --region=<YOUR_REGION> \
     --execution_role=<ROLE_ARN>
   ```

   c. **ECR Container Registry**:

   ```bash
   zenml container-registry register ecr_registry \
     --flavor=aws \
     --uri=<ACCOUNT_ID>.dkr.ecr.<YOUR_REGION>.amazonaws.com \
     --connector aws_connector
   ```

5. **Register and activate your stack**:
   ```bash
   zenml stack register aws_stack \
     -a s3_artifact_store \
     -o sagemaker_orchestrator \
     -c ecr_registry \
     --set
   ```

### Other Cloud Providers

Similar setup processes can be followed for other cloud providers:

- **Azure**: Install the Azure integration (`zenml integration install azure`) and set up Azure Blob Storage, AzureML, and Azure Container Registry
- **Google Cloud**: Install the GCP integration (`zenml integration install gcp gcs`) and set up GCS, Vertex AI, and GCR
- **Kubernetes**: Install the Kubernetes integration (`zenml integration install kubernetes`) and set up a Kubernetes cluster

For detailed configuration options for these providers, refer to the ZenML documentation:

- [GCP Integration Guide](https://docs.zenml.io/stacks/popular-stacks/gcp-guide)
- [Azure Integration Guide](https://docs.zenml.io/stacks/popular-stacks/azure-guide)
- [Kubernetes Integration Guide](https://docs.zenml.io/stacks/popular-stacks/kubernetes)
