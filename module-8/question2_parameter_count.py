"""
Question 2: What's the total number of parameters of the model?
"""

import numpy as np
import torch
import torch.nn as nn

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
# CREATE MODEL
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HairTypeCNN().to(device)

# ============================================================================
# QUESTION 2: PARAMETER COUNT
# ============================================================================

# Method 1: Manual counting
total_params = sum(p.numel() for p in model.parameters())
print("Question 2: Total Number of Parameters")
print("-" * 50)
print(f"Method 1 (Manual): {total_params:,} parameters")
print()

# Method 2: Using torchsummary (if available)
try:
    from torchsummary import summary
    print("Method 2 (torchsummary):")
    print("-" * 50)
    summary(model, input_size=(3, 200, 200))
except ImportError:
    print("Method 2: torchsummary not available")
    print("Install with: pip install torchsummary")
    print()

# Detailed breakdown
print("Parameter breakdown by layer:")
print("-" * 50)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name:20s}: {param.numel():>12,} parameters")
print()
print(f"{'TOTAL:':>20s} {total_params:>12,} parameters")
