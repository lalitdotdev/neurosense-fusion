# deploy_endpoint.py
import os
import sagemaker
from sagemaker.pytorch import PyTorchModel


def deploy_endpoint():
    sagemaker.Session()

    # Load config from environment variables
    role = os.environ.get("SAGEMAKER_DEPLOY_ROLE_ARN")
    s3_bucket = os.environ.get("S3_MODEL_BUCKET")
    s3_key = os.environ.get("S3_MODEL_KEY")
    endpoint_name = os.environ.get("ENDPOINT_NAME", "sentiment-analysis-endpoint")

    if not all([role, s3_bucket, s3_key]):
        raise EnvironmentError(
            "Missing required environment variables: "
            "SAGEMAKER_DEPLOY_ROLE_ARN, S3_MODEL_BUCKET, S3_MODEL_KEY"
        )

    model_uri = f"s3://{s3_bucket}/{s3_key}"

    model = PyTorchModel(
        model_data=model_uri,
        role=role,
        framework_version="2.5.1",
        py_version="py311",
        entry_point="inference.py",
        source_dir=".",
        name="sentiment-analysis-model",
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.g5.xlarge",
        endpoint_name=endpoint_name,
    )


if __name__ == "__main__":
    deploy_endpoint()
