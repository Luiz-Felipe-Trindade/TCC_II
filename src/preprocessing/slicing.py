import numpy as np


def get_positive_slices(mask: np.ndarray) -> list[int]:
    """
    Return slice indices containing at least one positive mask pixel.

    Parameters
    ----------
    mask : np.ndarray
        Three-dimensional binary mask with shape (z, y, x).

    Returns
    -------
    list[int]
        Slice indices containing the region of interest.
    """
    return [
        index
        for index in range(mask.shape[0])
        if np.any(mask[index] > 0)
    ]


def get_positive_pixel_count(mask: np.ndarray) -> dict[int, int]:
    """
    Count positive mask pixels for each positive slice.

    Parameters
    ----------
    mask : np.ndarray
        Three-dimensional binary mask.

    Returns
    -------
    dict[int, int]
        Mapping between slice index and number of positive pixels.
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
    Select all positive slices and include contextual slices
    before and after the region of interest.

    Parameters
    ----------
    mask : np.ndarray
        Three-dimensional binary mask.
    context : int, default=2
        Number of additional slices before and after the ROI.

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