from pathlib import Path

import SimpleITK as sitk


def load_medical_image(
    path: str | Path,
) -> tuple[sitk.Image, object]:
    """
    Load a medical image and convert it to a NumPy array.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Medical image not found: {path}"
        )

    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)

    return image, array


def get_image_metadata(
    image: sitk.Image,
) -> dict:
    """
    Return spatial metadata from a medical image.
    """
    return {
        "size": image.GetSize(),
        "spacing": image.GetSpacing(),
        "origin": image.GetOrigin(),
        "direction": image.GetDirection(),
    }


def validate_image_geometry(
    image_a: sitk.Image,
    image_b: sitk.Image,
    tolerance: float = 1e-4,
) -> bool:
    """
    Check whether two medical images have compatible geometry.
    """
    size_match = (
        image_a.GetSize()
        == image_b.GetSize()
    )

    spacing_match = all(
        abs(a - b) <= tolerance
        for a, b in zip(
            image_a.GetSpacing(),
            image_b.GetSpacing(),
        )
    )

    origin_match = all(
        abs(a - b) <= tolerance
        for a, b in zip(
            image_a.GetOrigin(),
            image_b.GetOrigin(),
        )
    )

    direction_match = all(
        abs(a - b) <= tolerance
        for a, b in zip(
            image_a.GetDirection(),
            image_b.GetDirection(),
        )
    )

    return (
        size_match
        and spacing_match
        and origin_match
        and direction_match
    )