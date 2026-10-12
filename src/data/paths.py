from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "src" / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

PICAI_DIR = RAW_DATA_DIR / "picai"
PICAI_LABELS_DIR = RAW_DATA_DIR / "picai_labels"

ANATOMICAL_DELINEATIONS_DIR = (
    PICAI_LABELS_DIR
    / "anatomical_delineations"
)

WHOLE_GLAND_DIR = (
    ANATOMICAL_DELINEATIONS_DIR
    / "whole_gland"
    / "AI"
)

CLINICAL_INFORMATION_DIR = (
    PICAI_LABELS_DIR
    / "clinical_information"
)

CSPCA_LESION_DIR = (
    PICAI_LABELS_DIR
    / "csPCa_lesion_delineations"
)

PROCESSED_DIR = DATA_DIR / "processed"

FEATURES_DIR = DATA_DIR / "features"

SPLITS_DIR = PROCESSED_DIR / "splits"

CHECKPOINTS_DIR = PROJECT_ROOT / "results" / "checkpoints"
