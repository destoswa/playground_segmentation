import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import rasterio
import cv2
import torch


class SegmentationDataset(Dataset):
    """
    PyTorch dataset for loading segmentation images and masks with optional preprocessing and augmentation.
    Parameters: 
        data_dir (str) - root directory containing image and mask folders; 
        processor (callable | None) - HuggingFace processor used to prepare model inputs; 
        num_layers (int) - number of image channels to load; 
        is_rgb (bool) - whether the images correspond to standard RGB inputs; 
        transform (callable | None) - optional augmentation transform applied to images and masks.
    Returns: 
        Dataset - dataset object providing image tensors, segmentation labels, and filenames.
    """

    def __init__(self, data_dir, processor=None, num_layers=3, transform=None):
        self.data_dir = data_dir
        self.processor = processor
        self.num_layers = num_layers
        self.transform = transform

        self.images = []
        self.masks = []
        for dataset in [r for r,_,_ in os.walk(data_dir) if os.path.basename(r) in ["images", "masks"]]:
            if os.path.basename(dataset) == 'images':
                self.images.append([os.path.join(dataset, x) for x in os.listdir(dataset)])
            else:
                self.masks.append([os.path.join(dataset, x) for x in os.listdir(dataset)])

        self.images = [x for row in self.images for x in row]
        self.masks = [x for row in self.masks for x in row]

        assert len(self.images) == len(self.masks), "Image/mask count mismatch"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = rasterio.open(self.images[idx]).read()[:self.num_layers, ...]
        mask = rasterio.open(self.masks[idx]).read().squeeze(0)

        # Apply augmentation
        if self.transform is not None:
            augmented = self.transform(image=np.moveaxis(image, 0, 2), mask=mask)
            image = np.moveaxis(augmented["image"], 2, 0)
            mask = augmented["mask"]

        if self.processor is not None:
            inputs = self.processor(images=image, segmentation_maps=mask.squeeze(-1), return_tensors="pt")
            inputs["pixel_values"] = inputs["pixel_values"].squeeze(0)  # HF returns tensors with extra batch dim, we remove manually
            inputs["labels"] = inputs["labels"].squeeze(0)              # HF returns tensors with extra batch dim, we remove manually
            inputs['filename'] = self.images[idx]
        else:
            imgs = image.astype(np.float32) / 255
            mean = np.mean(imgs, axis=(1,2), keepdims=True)
            std = np.std(imgs, axis=(1,2), keepdims=True)
            std = np.clip(std, 1e-3, None)
            imgs = (imgs - mean) / std
            inputs = {
                "pixel_values": torch.from_numpy(imgs).float(),
                'labels': torch.from_numpy(mask).long(),
                'filename': self.images[idx]
            }
            
        return inputs
    
    def get_images(self):
        return self.images
    

if __name__ == "__main__":
    pass
