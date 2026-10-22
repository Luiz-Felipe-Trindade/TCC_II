import cv2
import numpy as np


def normalize_t2w(image: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalization to non-zero pixels
    of a T2-weighted MRI slice.

    Parameters
    ----------
    image : np.ndarray
        Two-dimensional T2W image.

    Returns
    -------
    np.ndarray
        Normalized image as float32.
    """
    image = image.astype(np.float32)

    non_zero_pixels = image[image > 0]

    if non_zero_pixels.size == 0:
        return image

    mean = non_zero_pixels.mean()
    std = non_zero_pixels.std()

    if std == 0:
        return image

    return (image - mean) / std


def resize_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    size: tuple[int, int] = (256, 256),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resize an MRI slice and its segmentation mask.

    Linear interpolation is used for the image.
    Nearest-neighbor interpolation is used for the mask.

    Parameters
    ----------
    image : np.ndarray
        Two-dimensional MRI image.
    mask : np.ndarray
        Corresponding binary segmentation mask.
    size : tuple[int, int], default=(256, 256)
        Target image size.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Resized image and binary mask.
    """
    resized_image = cv2.resize(
        image,
        size,
        interpolation=cv2.INTER_LINEAR,
    )

    resized_mask = cv2.resize(
        mask.astype(np.uint8),
        size,
        interpolation=cv2.INTER_NEAREST,
    )

    resized_mask = (resized_mask > 0).astype(np.uint8)

    return resized_image, resized_mask