import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_slice_with_mask(
    image: np.ndarray,
    mask: np.ndarray,
    slice_index: int,
    alpha: float = 0.3,
) -> None:
    """
    Display a volume slice with its segmentation mask overlay.
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
    Compare the original image and mask with the processed versions.
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
    
def plot_segmentation_prediction(
    image: torch.Tensor,
    mask: torch.Tensor,
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> None:
    """
    Plot MRI image, ground-truth mask, and predicted mask.
    """
    image_np = image.squeeze().cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()

    probability = torch.sigmoid(logits)
    prediction = (
        probability >= threshold
    ).float()

    prediction_np = (
        prediction.squeeze().cpu().numpy()
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    axes[0].imshow(
        image_np,
        cmap="gray",
    )
    axes[0].set_title("T2W MRI")

    axes[1].imshow(
        image_np,
        cmap="gray",
    )
    axes[1].imshow(
        mask_np,
        alpha=0.4,
    )
    axes[1].set_title("Ground Truth")

    axes[2].imshow(
        image_np,
        cmap="gray",
    )
    axes[2].imshow(
        prediction_np,
        alpha=0.4,
    )
    axes[2].set_title("Prediction")

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()
    plt.show()