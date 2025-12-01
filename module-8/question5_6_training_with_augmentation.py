"""
Questions 5 & 6: Training with data augmentation
Question 5: Mean of test loss for all epochs (with augmentation)
Question 6: Average test accuracy for last 5 epochs (with augmentation)

This script continues training from the model trained in questions 3-4.
We train for 10 MORE epochs (11-20) with data augmentation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ============================================================================
# REPRODUCIBILITY SETUP
# ============================================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================================
# DEVICE SETUP
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# MODEL DEFINITION
# ============================================================================
class HairTypeCNN(nn.Module):
    """
    CNN Model for Binary Classification (Straight vs Curly Hair)
    - Input: (3, 200, 200)
    - Conv2d: 32 filters, 3x3 kernel, ReLU
    - MaxPool2d: 2x2
    - Flatten
    - Linear: 64 neurons, ReLU
    - Linear: 1 neuron (output)
    """
    
    def __init__(self):
        super(HairTypeCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=0
        )
        self.relu1 = nn.ReLU()
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten_size = 32 * 99 * 99  # 313,632
        
        self.fc1 = nn.Linear(self.flatten_size, 64)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.conv1(x)        # (batch, 32, 198, 198)
        x = self.relu1(x)
        x = self.pool(x)         # (batch, 32, 99, 99)
        x = x.view(x.size(0), -1)  # (batch, 313632)
        x = self.fc1(x)          # (batch, 64)
        x = self.relu2(x)
        x = self.fc2(x)          # (batch, 1)
        return x

# ============================================================================
# CREATE MODEL (CONTINUE FROM PREVIOUS TRAINING)
# ============================================================================
model = HairTypeCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.8)

# Try to load model from previous training (questions 3-4)
# We need to recreate the training for first 10 epochs, then continue with augmentation
# Since the model weights from Q3-4 aren't saved, we'll train all 20 epochs:
# First 10 without augmentation, then 10 with augmentation

print("=" * 70)
print("TRAINING PHASE 1: First 10 epochs WITHOUT augmentation")
print("=" * 70)
print()

# ============================================================================
# PHASE 1: DATA TRANSFORMS WITHOUT AUGMENTATION
# ============================================================================
train_transforms_no_aug = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset_phase1 = datasets.ImageFolder(root='data/train', transform=train_transforms_no_aug)
validation_dataset = datasets.ImageFolder(root='data/test', transform=test_transforms)

train_loader_phase1 = DataLoader(train_dataset_phase1, batch_size=20, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=20, shuffle=False)

# Train first 10 epochs without augmentation
history_phase1 = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []}

for epoch in range(10):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for images, labels in train_loader_phase1:
        images, labels = images.to(device), labels.to(device)
        labels = labels.float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataset_phase1)
    epoch_acc = correct_train / total_train
    history_phase1['loss'].append(epoch_loss)
    history_phase1['acc'].append(epoch_acc)

    model.eval()
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_epoch_loss = val_running_loss / len(validation_dataset)
    val_epoch_acc = correct_val / total_val
    history_phase1['val_loss'].append(val_epoch_loss)
    history_phase1['val_acc'].append(val_epoch_acc)

    print(f"Epoch {epoch+1}/10, "
          f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
          f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")

print()
print("=" * 70)
print("TRAINING PHASE 2: Next 10 epochs WITH augmentation (epochs 11-20)")
print("=" * 70)
print()

# ============================================================================
# PHASE 2: DATA TRANSFORMS WITH AUGMENTATION
# ============================================================================
train_transforms_aug = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.RandomRotation(50),
    transforms.RandomResizedCrop(200, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset_phase2 = datasets.ImageFolder(root='data/train', transform=train_transforms_aug)
train_loader_phase2 = DataLoader(train_dataset_phase2, batch_size=20, shuffle=True)

# Continue training for 10 more epochs WITH augmentation
history_phase2 = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []}

for epoch in range(10):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for images, labels in train_loader_phase2:
        images, labels = images.to(device), labels.to(device)
        labels = labels.float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataset_phase2)
    epoch_acc = correct_train / total_train
    history_phase2['loss'].append(epoch_loss)
    history_phase2['acc'].append(epoch_acc)

    model.eval()
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_epoch_loss = val_running_loss / len(validation_dataset)
    val_epoch_acc = correct_val / total_val
    history_phase2['val_loss'].append(val_epoch_loss)
    history_phase2['val_acc'].append(val_epoch_acc)

    print(f"Epoch {epoch+11}/20, "  # Showing as epochs 11-20
          f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
          f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")

print()
print("=" * 70)

# ============================================================================
# QUESTION 5: MEAN OF TEST LOSS FOR ALL EPOCHS (WITH AUGMENTATION)
# ============================================================================
# Test loss for epochs 11-20 (phase 2 with augmentation)
test_losses_aug = history_phase2['val_loss']  # validation loss = test loss
mean_test_loss = np.mean(test_losses_aug)

print("Question 5: Mean of Test Loss (all epochs with augmentation)")
print("-" * 70)
print(f"Test losses (epochs 11-20): {[f'{loss:.4f}' for loss in test_losses_aug]}")
print(f"Mean test loss: {mean_test_loss:.4f}")
print()

# ============================================================================
# QUESTION 6: AVERAGE TEST ACCURACY FOR LAST 5 EPOCHS
# ============================================================================
# Last 5 epochs of augmented training: epochs 16-20 (indices 5-9 in history_phase2)
last_5_test_acc = history_phase2['val_acc'][-5:]
avg_last_5_test_acc = np.mean(last_5_test_acc)

print("Question 6: Average Test Accuracy (last 5 epochs with augmentation)")
print("-" * 70)
print(f"Test accuracies (all 10 epochs with aug): {[f'{acc:.4f}' for acc in history_phase2['val_acc']]}")
print(f"Last 5 test accuracies (epochs 16-20): {[f'{acc:.4f}' for acc in last_5_test_acc]}")
print(f"Average test accuracy (last 5 epochs): {avg_last_5_test_acc:.4f}")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Question 5 - Mean test loss (epochs 11-20): {mean_test_loss:.4f}")
print(f"Question 6 - Average test accuracy (epochs 16-20): {avg_last_5_test_acc:.4f}")
print("=" * 70)
