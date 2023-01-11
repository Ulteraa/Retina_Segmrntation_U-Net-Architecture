import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
import torch.optim as optim
import config
import dataset_
from forward import Unet_Architecture
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image


def train(data, device, optimizer, model, loss_fn):
    loop = tqdm(data, leave=True)
    for _, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE); y = y.to(config.DEVICE)
        predict_ = model(x).squeeze(1)
        optimizer.zero_grad()
        loss = loss_fn(predict_, y)
        loss.backward()
        optimizer.step()
        # test(model)
    return loss


def train_fn():
    training_data = dataset_.Retina(image_dir='retina/DRIVE/training/images',
                  mask_dir='retina/DRIVE/training/1st_manual', transform=config.train_transform)
    training_loader = DataLoader(training_data, batch_size=config.BATCH_SIZE, shuffle=True)
    model = Unet_Architecture().to(device=config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    print("Starting training...\n")
    losses = []
    for epoch in range(config.NUM_EPOCHS):
        loss = train(training_loader, config.DEVICE, optimizer, model, loss_fn)
        print(f"\n Epoch {epoch} | Train Loss {loss} \n")
    test(model)


def test(model):
    training_data = dataset_.Retina(image_dir='retina/DRIVE/test/images',
                                    mask_dir='retina/DRIVE/test/images', transform=config.train_transform)

    test_data = DataLoader(training_data, batch_size=1, shuffle=True)

    with torch.no_grad():
        loop = tqdm(test_data)
        for _, (x, y) in enumerate(loop):
            x = x.to(config.DEVICE);
            y = y.to(config.DEVICE)
            predict_ = model(x)
            predict_ = (predict_ > 0.5).float()
            path = 'Test_Results/' + 'image_' + str(_) + '.png'
            save_image(predict_, path)


if __name__ == '__main__':
    train_fn()
