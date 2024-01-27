import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from datasets import SegDataModule
from models import ConvNet
from lightning import seed_everything
from tqdm import tqdm
import torch.optim as optim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

seed_everything(2023)

def TrainFunc(loader, model, optim, loss, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():

    ValLoader = SegDataset().val_dataloader()
    TrainLoader = SegDataModule().train_dataloader()
    Model = ConvNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(15):
        TrainFunc(TrainLoader, Model, optimizer, loss_fn, scaler)

        checkpoint = {
            "state_dict": Model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }

        save_checkpoint(checkpoint)

        check_accuracy(ValLoader, Model, device=DEVICE)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        monitor="val_high_vegetation_IoU",
        mode="max",
        filename="{epoch}-{val_loss:.2f}-{val_high_vegetation_IoU:.2f}-{val_mIoU:.2f}",
        save_top_k=3,
    )

if __name__ == "__main__":
    main()