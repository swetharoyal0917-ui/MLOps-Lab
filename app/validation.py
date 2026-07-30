import pandas as pd


REQUIRED_COLUMNS = [
    "diagnosis",
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
]


def validate_dataset(df: pd.DataFrame):
    """
    Validate dataset structure and quality.
    Raises ValueError if validation fails.
    """

    # Check empty dataset
    if df.empty:
        raise ValueError("Dataset is empty.")

    # Check null values
    if df.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains null values.")

    # Check required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return True


def validate_quality(metrics: dict):
    """
    Validate model quality gates.
    """

    thresholds = {
        "accuracy": 0.90,
        "precision": 0.90,
        "recall": 0.90,
        "f1": 0.90,
    }

    failed = []

    for metric, threshold in thresholds.items():
        if metrics.get(metric, 0) < threshold:
            failed.append(
                f"{metric}={metrics.get(metric):.4f} < {threshold}"
            )

    if failed:
        raise ValueError(
            "Quality gate failed:\n" + "\n".join(failed)
        )

    return True