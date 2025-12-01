"""
Questions 3 & 4: Training the model and calculating metrics
Question 3: Median of training accuracy for all epochs
Question 4: Standard deviation of training loss for all epochs
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
# CREATE MODEL, LOSS FUNCTION, OPTIMIZER
# ============================================================================
model = HairTypeCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.8)

# ============================================================================
# DATA TRANSFORMS
# ============================================================================
train_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )  # ImageNet normalization
])

test_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )  # ImageNet normalization
])

# ============================================================================
# LOAD DATASETS
# ============================================================================
train_dataset = datasets.ImageFolder(root='data/train', transform=train_transforms)
validation_dataset = datasets.ImageFolder(root='data/test', transform=test_transforms)

# ============================================================================
# CREATE DATA LOADERS
# ============================================================================
train_loader = DataLoader(
    train_dataset,
    batch_size=20,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=20,
    shuffle=False
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(validation_dataset)}")
print()

# ============================================================================
# TRAINING LOOP
# ============================================================================
num_epochs = 10
history = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []}

print("Starting training...")
print("=" * 70)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        labels = labels.float().unsqueeze(1)  # Ensure labels are float and have shape (batch_size, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        # For binary classification with BCEWithLogitsLoss, apply sigmoid to outputs before thresholding for accuracy
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct_train / total_train
    history['loss'].append(epoch_loss)
    history['acc'].append(epoch_acc)

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
    history['val_loss'].append(val_epoch_loss)
    history['val_acc'].append(val_epoch_acc)

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
          f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")

print("=" * 70)
print()

# ============================================================================
# QUESTION 3: MEDIAN OF TRAINING ACCURACY
# ============================================================================
train_accuracies = history['acc']
median_train_acc = np.median(train_accuracies)

print("Question 3: Median of Training Accuracy")
print("-" * 50)
print(f"Training accuracies: {[f'{acc:.4f}' for acc in train_accuracies]}")
print(f"Median training accuracy: {median_train_acc:.4f}")
print()

# ============================================================================
# QUESTION 4: STANDARD DEVIATION OF TRAINING LOSS
# ============================================================================
train_losses = history['loss']
std_train_loss = np.std(train_losses)

print("Question 4: Standard Deviation of Training Loss")
print("-" * 50)
print(f"Training losses: {[f'{loss:.4f}' for loss in train_losses]}")
print(f"Standard deviation of training loss: {std_train_loss:.4f}")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Question 3 - Median training accuracy: {median_train_acc:.4f}")
print(f"Question 4 - Std dev of training loss: {std_train_loss:.4f}")
print("=" * 70)
print()

# ============================================================================
# SAVE MODEL FOR CONTINUED TRAINING
# ============================================================================
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'history': history,
}, 'model_after_10_epochs.pth')
print("Model saved to 'model_after_10_epochs.pth' for continued training")

