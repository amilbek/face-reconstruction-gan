import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random


class FaceDataset(Dataset):
    def __init__(self, image_dir, image_size=128, mask_type='random'):
        self.image_dir = image_dir
        self.image_size = image_size
        self.mask_type = mask_type
        
        self.image_files = [
            f for f in os.listdir(image_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def create_random_mask(self, img_size):
        mask = torch.ones(3, img_size, img_size)
        
        mask_h = random.randint(int(img_size * 0.2), int(img_size * 0.6))
        mask_w = random.randint(int(img_size * 0.2), int(img_size * 0.6))
        
        y = random.randint(0, img_size - mask_h)
        x = random.randint(0, img_size - mask_w)
        
        mask[:, y:y+mask_h, x:x+mask_w] = 0
        
        return mask
    
    def create_center_mask(self, img_size):
        mask = torch.ones(3, img_size, img_size)
        
        quarter = img_size // 4
        mask[:, quarter:3*quarter, quarter:3*quarter] = 0
        
        return mask
    
    def create_half_mask(self, img_size):
        mask = torch.ones(3, img_size, img_size)
        
        if random.random() > 0.5:
            mask[:, :, :img_size//2] = 0 
        else:
            mask[:, :, img_size//2:] = 0 
        
        return mask
    
    def create_eyes_mask(self, img_size):
        mask = torch.ones(3, img_size, img_size)
        
        eye_region_height = int(img_size * 0.4)
        mask[:, :eye_region_height, :] = 0
        
        return mask
    
    def create_mouth_mask(self, img_size):
        mask = torch.ones(3, img_size, img_size)
        
        start_y = int(img_size * 0.7)
        mask[:, start_y:, :] = 0
        
        return mask
    
    def create_scattered_mask(self, img_size, num_patches=5):
        mask = torch.ones(3, img_size, img_size)
        
        for _ in range(num_patches):
            patch_size = random.randint(10, 30)
            y = random.randint(0, img_size - patch_size)
            x = random.randint(0, img_size - patch_size)
            mask[:, y:y+patch_size, x:x+patch_size] = 0
        
        return mask
    
    def get_mask(self, img_size):
        if self.mask_type == 'random':
            return self.create_random_mask(img_size)
        elif self.mask_type == 'center':
            return self.create_center_mask(img_size)
        elif self.mask_type == 'half':
            return self.create_half_mask(img_size)
        elif self.mask_type == 'eyes':
            return self.create_eyes_mask(img_size)
        elif self.mask_type == 'mouth':
            return self.create_mouth_mask(img_size)
        elif self.mask_type == 'scattered':
            return self.create_scattered_mask(img_size)
        elif self.mask_type == 'mixed':
            mask_types = ['random', 'center', 'half', 'eyes', 'mouth', 'scattered']
            chosen_type = random.choice(mask_types)
            self.mask_type = chosen_type
            mask = self.get_mask(img_size)
            self.mask_type = 'mixed'
            return mask
        else:
            return self.create_random_mask(img_size)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        image_tensor = self.transform(image)
        
        mask = self.get_mask(self.image_size)
        
        masked_image = image_tensor * mask
        
        return {
            'real_image': image_tensor,
            'masked_image': masked_image,
            'mask': mask
        }


def get_dataloaders(image_dir, batch_size=16, image_size=128, mask_type='mixed', 
                    train_split=0.8, num_workers=2):
    full_dataset = FaceDataset(image_dir, image_size, mask_type)
    
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset loaded: {len(train_dataset)} training, {len(val_dataset)} validation samples")
    
    return train_loader, val_loader


def visualize_masks(save_path='mask_examples.png'):
    import matplotlib.pyplot as plt
    
    mask_types = ['random', 'center', 'half', 'eyes', 'mouth', 'scattered']
    
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    
    for idx, mask_type in enumerate(mask_types):
        dummy_img = torch.rand(3, 128, 128)
        
        dataset = FaceDataset('.', mask_type=mask_type)
        mask = dataset.get_mask(128)
        masked_img = dummy_img * mask
        
        axes[0, idx].imshow(dummy_img.permute(1, 2, 0))
        axes[0, idx].set_title(f'{mask_type.capitalize()} - Original')
        axes[0, idx].axis('off')
        
        axes[1, idx].imshow(masked_img.permute(1, 2, 0))
        axes[1, idx].set_title(f'{mask_type.capitalize()} - Masked')
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Mask examples saved to {save_path}")
