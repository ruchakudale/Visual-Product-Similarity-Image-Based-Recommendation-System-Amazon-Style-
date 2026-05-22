import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from config import IMAGE_DIR, IMAGE_SIZE

class ProductImageDataset(Dataset):
    def __init__(self):
        self.image_paths = []
        
        for root, _, files in os.walk(IMAGE_DIR):
            for file in files:
                if file.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.image_paths.append(os.path.join(root, file))

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, img_path