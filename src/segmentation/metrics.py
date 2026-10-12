import torch


def dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> float:
    """
    Compute the Dice score for binary segmentation.
    """
    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= threshold
    ).float()

    predictions = predictions.view(
        predictions.size(0),
        -1,
    )

    targets = targets.view(
        targets.size(0),
        -1,
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    dice = (
        2.0 * intersection
        + smooth
    ) / (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
        + smooth
    )

    return dice.mean().item()

def positive_dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> tuple[float, int]:
    """
    Compute Dice score only for samples with positive ground-truth masks.
    """
    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= threshold
    ).float()

    predictions = predictions.view(
        predictions.size(0),
        -1,
    )

    targets = targets.view(
        targets.size(0),
        -1,
    )

    positive_mask = (
        targets.sum(dim=1) > 0
    )

    if positive_mask.sum() == 0:
        return 0.0, 0

    predictions = predictions[positive_mask]
    targets = targets[positive_mask]

    intersection = (
        predictions * targets
    ).sum(dim=1)

    dice = (
        2.0 * intersection
        + smooth
    ) / (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
        + smooth
    )

    return (
        dice.sum().item(),
        dice.size(0),
    )
    
def iou_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> float:
    """
    Compute the Intersection over Union score
    for binary segmentation.
    """
    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= threshold
    ).float()

    predictions = predictions.view(
        predictions.size(0),
        -1,
    )

    targets = targets.view(
        targets.size(0),
        -1,
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    union = (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
        - intersection
    )

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return iou.mean().item()