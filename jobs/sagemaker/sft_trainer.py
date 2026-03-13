import os

from sagemaker.train import ModelTrainer
from sagemaker.train.configs import InputData


def _build_trainer():
    """Return a ModelTrainer configured for this project."""
    training_image = os.environ.get("TRAINING_IMAGE", "sympy-rlvr-img")
    role = os.environ.get(
        "SAGEMAKER_ROLE", "arn:aws:iam::979667333968:role/studio-admin"
    )
    return ModelTrainer(
        training_image=training_image,
        role=role,
    )


def _make_input_data() -> list[InputData] | None:
    """Construct InputData only if TRAINING_DATA_URI is set."""
    training_data_uri = os.environ.get(
        "TRAINING_DATA_URI",
        "s3://training-artifacts-979667333968-us-east-1-an/rlvr_sympy/shared/data",
    )
    if not training_data_uri:
        return None
    return [
        InputData(
            channel_name="training",
            data_source=training_data_uri,
        )
    ]


def run_training() -> None:
    trainer = _build_trainer()
    input_data_config = _make_input_data()
    trainer.train(input_data_config=input_data_config)


if __name__ == "__main__":
    run_training()
