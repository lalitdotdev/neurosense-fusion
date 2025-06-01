# sagemaker_launcher.py
#
# Secure SageMaker PyTorch Training Launcher Script
# - Configures and launches a PyTorch training job on Amazon SageMaker using environment variables for all sensitive info.
# - Sets up TensorBoard logging, specifies S3 data channels, and passes hyperparameters to the training script.
# - Uses PyTorch Estimator for managed training, logs, and distributed support.
# - After training, model artifacts and logs are saved to S3 for deployment or further analysis.

import os
from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig


def start_training():
    # Load sensitive config from environment variables for security
    role = os.environ.get("SAGEMAKER_ROLE_ARN")
    s3_tensorboard = os.environ.get("S3_TENSORBOARD_PATH")
    s3_train = os.environ.get("S3_TRAIN_PATH")
    s3_val = os.environ.get("S3_VAL_PATH")
    s3_test = os.environ.get("S3_TEST_PATH")

    if not all([role, s3_tensorboard, s3_train, s3_val, s3_test]):
        raise EnvironmentError(
            "Missing one or more required environment variables: "
            "SAGEMAKER_ROLE_ARN, S3_TENSORBOARD_PATH, S3_TRAIN_PATH, S3_VAL_PATH, S3_TEST_PATH"
        )

    tensorboard_config = TensorBoardOutputConfig(
        s3_output_path=s3_tensorboard,
        container_local_output_path="/opt/ml/output/tensorboard",
    )

    estimator = PyTorch(
        entry_point="train.py",
        source_dir="training",
        role=role,
        framework_version="2.5.1",
        py_version="py311",
        instance_count=1,
        instance_type="ml.g5.xlarge",
        hyperparameters={
            "batch-size": 32,
            "epochs": 25,
        },
        tensorboard_config=tensorboard_config,
    )

    estimator.fit(
        {
            "training": s3_train,
            "validation": s3_val,
            "test": s3_test,
        }
    )


if __name__ == "__main__":
    start_training()
