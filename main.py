import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

class BrainMRIDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        # Generate 20 synthetic images for initial baseline pipeline testing
        self.data = [Image.new('RGB', (224, 224), color='gray') for _ in range(20)]
        self.labels = [torch.randint(0, 3, (1,)).item() for _ in range(20)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def build_model(num_classes=3):
    # Load pretrained ResNet18 backbone
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Freeze feature extraction layers
    for param in model.parameters():
        param.requires_grad = False
    # Swap out final fully-connected classifier head
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = BrainMRIDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=3).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    # Run a quick baseline optimization step
    model.train()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    print(f"Pipeline executed successfully on device: {device}")

if __name__ == "__main__":
    main()
