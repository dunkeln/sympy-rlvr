import os

from sagemaker.pytorch import PyTorch


TRAINING_ROLE = "arn:aws:iam::979667333968:role/studio-admin"
TRAINING_DATA_URI = (
    "s3://training-artifacts-979667333968-us-east-1-an/rlvr_sympy/shared/data"
)


def main() -> None:
    estimator = PyTorch(
        entry_point="sft/trainer.py",
        source_dir=".",
        role=TRAINING_ROLE,
        instance_count=int(os.environ.get("INSTANCE_COUNT", "1")),
        instance_type=os.environ.get("INSTANCE_TYPE", "ml.g5.xlarge"),
        framework_version="2.0.0",
        py_version="py310",
        hyperparameters={
            "--epochs": os.environ.get("EPOCHS", "1"),
            "--alpha": os.environ.get("ALPHA", "0.001"),
            "--batch": os.environ.get("BATCH", "64"),
            "--patience": os.environ.get("PATIENCE", "3"),
            "--delta": os.environ.get("DELTA", "0.01"),
        },
        environment={
            "TRAINING_DATA_URI": TRAINING_DATA_URI,
            "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", ""),
        },
    )

    estimator.fit(
        TRAINING_DATA_URI,
        logs=True,
        wait=True,
    )


if __name__ == "__main__":
    main()
