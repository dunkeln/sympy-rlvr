from sagemaker.pytorch import PyTorch

params = {
    "epochs": 1,
    "alpha": 0.001,
    "batch": 64,
    "patience": 3,
    "delta": 0.01,
}


estimator = PyTorch(
    entry_point="../../sft/trainer.py",
    role="",
    instance_type="ml.g5.xlarge",
    instance_count=1,
    hyperparameters={**params},
)

estimator.fit()
