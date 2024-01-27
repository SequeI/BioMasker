from typing import List
import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision


def mask_to_rle(mask: np.ndarray):
    """
    Convert a binary mask to RLE format.
    :param mask: numpy array, 1 - mask, 0 - background
    :return: RLE array
    """
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return [int(x) for x in runs]


def mask_to_rgb(mask: np.ndarray, label_to_color: dict):
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    mask_red = np.zeros_like(mask, dtype=np.uint8)
    mask_green = np.zeros_like(mask, dtype=np.uint8)
    mask_blue = np.zeros_like(mask, dtype=np.uint8)

    for l in label_to_color:
        mask_red[mask == l] = label_to_color[l][0]
        mask_green[mask == l] = label_to_color[l][1]
        mask_blue[mask == l] = label_to_color[l][2]

    mask_colors = (
        np.stack([mask_red, mask_green, mask_blue]).astype(np.uint8).transpose(1, 2, 0)
    )
    return mask_colors

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()