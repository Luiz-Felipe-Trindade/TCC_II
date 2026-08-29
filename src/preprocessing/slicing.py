import numpy as np


def get_positive_slices(mask: np.ndarray) -> list[int]:
    """
    Return the indices of slices containing at least
    one positive pixel in the segmentation mask.

    Parameters
    ----------
    mask : np.ndarray
        Three-dimensional segmentation mask with shape (z, y, x).

    Returns
    -------
    list[int]
        Indices of slices containing the region of interest.
    """
    return [
        index
        for index in range(mask.shape[0])
        if np.any(mask[index] > 0)
    ]


def get_positive_pixel_count(mask: np.ndarray) -> dict[int, int]:
    """
    Count positive mask pixels for each slice containing
    the region of interest.

    Parameters
    ----------
    mask : np.ndarray
        Three-dimensional segmentation mask.

    Returns
    -------
    dict[int, int]
        Slice index mapped to its number of positive pixels.
    """
    positive_slices = get_positive_slices(mask)

    return {
        index: int(np.sum(mask[index] > 0))
        for index in positive_slices
    }


def select_slices_with_context(
    mask: np.ndarray,
    context: int = 2,
) -> list[int]:
    """
    Select all slices containing the region of interest
    and include additional contextual slices before and after it.

    Parameters
    ----------
    mask : np.ndarray
        Three-dimensional segmentation mask.
    context : int, default=2
        Number of additional slices included before and after
        the positive region.

    Returns
    -------
    list[int]
        Selected slice indices.
    """
    positive_slices = get_positive_slices(mask)

    if not positive_slices:
        return []

    first_positive = positive_slices[0]
    last_positive = positive_slices[-1]

    start = max(0, first_positive - context)
    end = min(mask.shape[0] - 1, last_positive + context)

    return list(range(start, end + 1))