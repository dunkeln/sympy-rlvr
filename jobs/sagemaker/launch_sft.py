from sagemaker.pytorch import PyTorch

params = {
    "epochs": 1,
    "alpha": 0.001,
    "batch": 64,
    "patience": 3,
    "delta": 0.01,
}


estimator = PyTorch(
    entry_point="trainer.py",
    source_dir="sft",
    role="arn:aws:iam::979667333968:role/studio-admin",
    # INFO: latest versions
    framework_version="2.8",
    py_version="py312",
    instance_type="ml.g5.xlarge",
    instance_count=1,
    hyperparameters={**params},
)

estimator.fit()
