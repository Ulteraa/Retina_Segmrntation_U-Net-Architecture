import torch
import albumentations as transform
import cv2
import albumentations as transform
from albumentations.pytorch import ToTensorV2
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 4
IMAGE_HEIGHT = 584; IMAGE_WIDTH = 565

train_transform = transform.Compose(
    [
        transform.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        transform.Rotate(limit=35, p=1.0),
        transform.HorizontalFlip(p=0.5),
        transform.VerticalFlip(p=0.1),
        transform.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)