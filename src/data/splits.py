from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def create_patient_splits(
    clinical_data: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Create stratified train, validation, and test splits
    at the patient level.

    The final proportions are approximately:
    70% training, 15% validation, and 15% test.
    """
    data = clinical_data[
        [
            "patient_id",
            "study_id",
            "case_csPCa",
        ]
    ].copy()

    train_data, temporary_data = train_test_split(
        data,
        test_size=0.30,
        stratify=data["case_csPCa"],
        random_state=random_state,
    )

    validation_data, test_data = train_test_split(
        temporary_data,
        test_size=0.50,
        stratify=temporary_data["case_csPCa"],
        random_state=random_state,
    )

    train_data["split"] = "train"
    validation_data["split"] = "validation"
    test_data["split"] = "test"

    split_data = pd.concat(
        [
            train_data,
            validation_data,
            test_data,
        ],
        ignore_index=True,
    )

    return split_data
