from pathlib import Path
import numpy as np

from src.data.loader import (
    load_medical_image,
    validate_image_geometry,
)
from src.data.mapping import (
    get_case_image_paths,
    find_whole_gland_mask,
)
from src.preprocessing.slicing import (
    select_slices_with_context
)
from src.preprocessing.preprocessing import (
    normalize_t2w,
    resize_image_and_mask,
)


def process_case(
    patient_id: str,
    study_id: str,
    output_dir: str | Path,
    context: int = 2,
    image_size: tuple[int, int] = (256, 256),
) -> int:
    """
    Process one PI-CAI case and save selected 2D image-mask pairs.

    Returns
    -------
    int
        Number of saved slices.
    """
    output_dir = Path(output_dir)

    images_output_dir = output_dir / "images"
    masks_output_dir = output_dir / "masks"

    images_output_dir.mkdir(parents=True, exist_ok=True)
    masks_output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = get_case_image_paths(
        patient_id,
        study_id,
    )

    t2w_path = image_paths["t2w"]

    mask_path = find_whole_gland_mask(
        patient_id,
        study_id,
    )

    if mask_path is None:
        print(
            f"Mask not found for case "
            f"{patient_id}_{study_id}"
        )
        return 0

    _, t2w = load_medical_image(t2w_path)
    _, mask = load_medical_image(mask_path)

    selected_slices = select_slices_with_context(
        mask,
        context=context,
    )

    case_id = f"{patient_id}_{study_id}"

    saved_slices = 0

    for slice_index in selected_slices:
        image_slice = t2w[slice_index]
        mask_slice = mask[slice_index]

        normalized_image = normalize_t2w(
            image_slice
        )

        resized_image, resized_mask = (
            resize_image_and_mask(
                normalized_image,
                mask_slice,
                size=image_size,
            )
        )

        image_filename = (
            f"{case_id}_slice_{slice_index:03d}.npy"
        )

        mask_filename = (
            f"{case_id}_slice_{slice_index:03d}.npy"
        )

        np.save(
            images_output_dir / image_filename,
            resized_image.astype(np.float32),
        )

        np.save(
            masks_output_dir / mask_filename,
            resized_mask.astype(np.uint8),
        )

        saved_slices += 1

    return saved_slices

def validate_case(
    patient_id: str,
    study_id: str,
) -> dict:
    """
    Validate the T2W image and whole-gland mask of a PI-CAI case.

    Returns
    -------
    dict
        Validation information for the case.
    """
    case_id = f"{patient_id}_{study_id}"

    image_paths = get_case_image_paths(
        patient_id,
        study_id,
    )

    t2w_path = image_paths["t2w"]

    mask_path = find_whole_gland_mask(
        patient_id,
        study_id,
    )

    result = {
        "case_id": case_id,
        "valid": False,
        "same_shape": False,
        "same_geometry": False,
        "image_shape": None,
        "mask_shape": None,
        "error": None,
    }

    if mask_path is None:
        result["error"] = "Whole-gland mask not found"
        return result

    try:
        t2w_image, t2w = load_medical_image(
            t2w_path
        )

        mask_image, mask = load_medical_image(
            mask_path
        )

        result["image_shape"] = t2w.shape
        result["mask_shape"] = mask.shape

        result["same_shape"] = (
            t2w.shape == mask.shape
        )

        result["same_geometry"] = validate_image_geometry(
            t2w_image,
            mask_image,
        )

        result["valid"] = (
            result["same_shape"]
            and result["same_geometry"]
        )

    except Exception as error:
        result["error"] = str(error)

    return result

def process_dataset(
    cases: list[tuple[str, str]],
    output_dir: str | Path,
    context: int = 2,
    image_size: tuple[int, int] = (256, 256),
) -> dict:
    """
    Process multiple PI-CAI cases and build the 2D dataset.

    Returns
    -------
    dict
        Processing summary containing the number of successful cases,
        failed cases, and saved slices.
    """
    successful_cases = 0
    failed_cases = []
    total_slices = 0

    total_cases = len(cases)

    for index, (patient_id, study_id) in enumerate(
        cases,
        start=1,
    ):
        case_id = f"{patient_id}_{study_id}"

        try:
            saved_slices = process_case(
                patient_id=patient_id,
                study_id=study_id,
                output_dir=output_dir,
                context=context,
                image_size=image_size,
            )

            if saved_slices > 0:
                successful_cases += 1
                total_slices += saved_slices
            else:
                failed_cases.append(
                    {
                        "case_id": case_id,
                        "error": "No slices saved",
                    }
                )

        except Exception as error:
            failed_cases.append(
                {
                    "case_id": case_id,
                    "error": str(error),
                }
            )

        if index % 50 == 0 or index == total_cases:
            print(
                f"Processed {index}/{total_cases} cases "
                f"| Saved slices: {total_slices}"
            )

    return {
        "total_cases": total_cases,
        "successful_cases": successful_cases,
        "failed_cases": failed_cases,
        "total_slices": total_slices,
    }