import torch
from pathlib import Path
from src.segmentation.metrics import (
    dice_score,
    positive_dice_score,
    iou_score,
)

def train_one_epoch(
    model: torch.nn.Module,
    dataloader,
    criterion,
    optimizer,
    device: torch.device,
) -> float:
    """
    Train the model for one epoch.

    Returns
    -------
    float
        Average training loss.
    """
    model.train()

    total_loss = 0.0
    total_batches = 0
    
    


    for images, masks in dataloader:
        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            masks,
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / total_batches

def validate_one_epoch(
    model: torch.nn.Module,
    dataloader,
    criterion,
    device: torch.device,
) -> float:
    """
    Evaluate the model for one validation epoch.

    Returns
    -------
    float
        Average validation loss.
    """
    model.eval()

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(
                device,
                non_blocking=True,
            )

            masks = masks.to(
                device,
                non_blocking=True,
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                masks,
            )

            total_loss += loss.item()
            total_batches += 1

    return total_loss / total_batches

def train_model(
    model: torch.nn.Module,
    train_loader,
    validation_loader,
    criterion,
    optimizer,
    device: torch.device,
    epochs: int,
    checkpoint_path: str | Path,
) -> dict:
    """
    Train the segmentation model and save the best checkpoint
    according to validation loss.

    Returns
    -------
    dict
        Training history containing train and validation losses.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = {
        "train_loss": [],
        "validation_loss": [],
    }

    best_validation_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation_loss = validate_one_epoch(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(
            train_loss
        )

        history["validation_loss"].append(
            validation_loss
        )

        print(
            f"Epoch {epoch}/{epochs} "
            f"| Train loss: {train_loss:.4f} "
            f"| Validation loss: {validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": validation_loss,
                },
                checkpoint_path,
            )

            print(
                f"Best model saved "
                f"(validation loss: {validation_loss:.4f})"
            )

    return history


def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    criterion,
    device: torch.device,
) -> dict:
    """
    Evaluate the model using loss and Dice score.
    """
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_batches = 0

    total_positive_dice = 0.0
    total_positive_samples = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                masks,
            )

            dice = dice_score(
                outputs,
                masks,
            )
            iou = iou_score(
                outputs,
                masks,
            )

            total_iou += iou           

            positive_dice_sum, positive_count = positive_dice_score(
                outputs,
                masks,
            )

            total_loss += loss.item()
            total_dice += dice
            total_batches += 1

            total_positive_dice += positive_dice_sum
            total_positive_samples += positive_count
            
    return {
    "loss": total_loss / total_batches,
    "dice": total_dice / total_batches,
    "iou": total_iou / total_batches,
    "positive_dice": (
        total_positive_dice / total_positive_samples
        if total_positive_samples > 0
        else 0.0
    ),
}