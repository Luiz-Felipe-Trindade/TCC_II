from pathlib import Path

from src.data.paths import PICAI_DIR, WHOLE_GLAND_DIR


def build_case_id(patient_id: str, study_id: str) -> str:
    """
    Build the PI-CAI case identifier.

    Example
    -------
    patient_id = "10000"
    study_id = "1000000"

    Returns
    -------
    "10000_1000000"
    """
    return f"{patient_id}_{study_id}"


def get_case_image_paths(
    patient_id: str,
    study_id: str,
) -> dict[str, Path]:
    """
    Return the available MRI paths for a PI-CAI case.
    """
    case_id = build_case_id(patient_id, study_id)

    case_dir = PICAI_DIR / patient_id

    return {
        "t2w": case_dir / f"{case_id}_t2w.mha",
        "adc": case_dir / f"{case_id}_adc.mha",
        "hbv": case_dir / f"{case_id}_hbv.mha",
        "cor": case_dir / f"{case_id}_cor.mha",
        "sag": case_dir / f"{case_id}_sag.mha",
    }


def find_whole_gland_mask(
    patient_id: str,
    study_id: str,
) -> Path | None:
    """
    Search for the whole-gland segmentation mask
    associated with a PI-CAI case.
    """
    case_id = build_case_id(patient_id, study_id)
    filename = f"{case_id}.nii.gz"

    matches = list(
        WHOLE_GLAND_DIR.rglob(filename)
    )

    if not matches:
        return None
    return matches[0]

def discover_t2w_cases() -> list[tuple[str, str]]:
    """
    Discover PI-CAI cases that contain a T2W MRI file.

    Returns
    -------
    list[tuple[str, str]]
        List of (patient_id, study_id) pairs.
    """
    cases = []

    for patient_dir in PICAI_DIR.iterdir():
        if not patient_dir.is_dir():
            continue

        patient_id = patient_dir.name

        for t2w_path in patient_dir.glob("*_t2w.mha"):
            case_id = t2w_path.name.replace("_t2w.mha", "")

            parts = case_id.split("_")

            if len(parts) != 2:
                continue

            _, study_id = parts

            cases.append(
                (patient_id, study_id)
            )
    return sorted(cases)