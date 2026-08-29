import matplotlib.pyplot as plt
import numpy as np


def plot_slice_with_mask(
    image: np.ndarray,
    mask: np.ndarray,
    slice_index: int,
    alpha: float = 0.3,
) -> None:
    """
    Display a volume slice with its segmentation mask overlay.

    Parameters
    ----------
    image : np.ndarray
        Three-dimensional MRI volume.
    mask : np.ndarray
        Three-dimensional segmentation mask.
    slice_index : int
        Slice index to display.
    alpha : float, default=0.3
        Mask transparency.
    """
    plt.figure(figsize=(8, 8))

    plt.imshow(image[slice_index], cmap="gray")
    plt.imshow(mask[slice_index], alpha=alpha)

    plt.title(f"Slice {slice_index}")
    plt.axis("off")

    plt.show()


def compare_original_and_processed(
    original_image: np.ndarray,
    original_mask: np.ndarray,
    processed_image: np.ndarray,
    processed_mask: np.ndarray,
    alpha: float = 0.3,
) -> None:
    """
    Compare the original image/mask pair with the processed pair.

    Parameters
    ----------
    original_image : np.ndarray
        Original MRI slice.
    original_mask : np.ndarray
        Original segmentation mask.
    processed_image : np.ndarray
        Processed MRI slice.
    processed_mask : np.ndarray
        Processed segmentation mask.
    alpha : float, default=0.3
        Mask transparency.
    """
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(original_image, cmap="gray")
    axes[0].imshow(original_mask, alpha=alpha)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(processed_image, cmap="gray")
    axes[1].imshow(processed_mask, alpha=alpha)
    axes[1].set_title("Processed")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()