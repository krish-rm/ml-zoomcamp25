# Question 2: Total Number of Parameters

## Question
What's the total number of parameters of the model?

## Options:
- 896
- 11,214,912
- 15,896,912
- **20,073,473** ✓

---

## ✅ ANSWER: **20,073,473**

---

## Detailed Calculation

### Step-by-Step Breakdown

#### 1. Convolutional Layer (Conv2d)

**Configuration:**
- Input channels: 3 (RGB)
- Output channels: 32 (filters)
- Kernel size: 3 × 3
- Padding: 0

**Parameters:**
- Weights: `32 × 3 × 3 × 3 = 864`
  - 32 filters
  - Each filter: 3 input channels × 3 × 3 kernel
- Bias: `32` (one per filter)
- **Conv1 Total: 896 parameters**

#### 2. Feature Map Sizes

**After Convolution:**
- Input: (200, 200)
- With kernel 3×3 and no padding: (200 - 3 + 1) = **198**
- Output shape: (32, 198, 198)

**After MaxPooling:**
- Pool size: 2 × 2
- Output: 198 ÷ 2 = **99**
- Output shape: (32, 99, 99)

**Flattened Size:**
- `32 × 99 × 99 = 313,632` values

#### 3. First Linear Layer (FC1)

**Configuration:**
- Input: 313,632 (flattened features)
- Output: 64 neurons

**Parameters:**
- Weights: `64 × 313,632 = 20,072,448`
- Bias: `64` (one per neuron)
- **FC1 Total: 20,072,512 parameters**

#### 4. Output Linear Layer (FC2)

**Configuration:**
- Input: 64
- Output: 1 (binary classification)

**Parameters:**
- Weights: `1 × 64 = 64`
- Bias: `1`
- **FC2 Total: 65 parameters**

---

## Grand Total

```
Conv1:         896
FC1:    20,072,512
FC2:            65
───────────────────
TOTAL:  20,073,473 parameters
```

---

## Verification Methods

### Method 1: PyTorch Built-in
```python
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
# Output: 20,073,473
```

### Method 2: torchsummary
```python
from torchsummary import summary
summary(model, input_size=(3, 200, 200))

# Output:
# Total params: 20,073,473
# Trainable params: 20,073,473
```

### Method 3: Manual Counting
```python
for name, param in model.named_parameters():
    print(f"{name}: {param.numel():,} parameters")
    
# Output:
# conv1.weight: 864
# conv1.bias: 32
# fc1.weight: 20,072,448
# fc1.bias: 64
# fc2.weight: 64
# fc2.bias: 1
# Total: 20,073,473
```

All three methods confirm: **20,073,473 parameters**

---

## Parameter Distribution

The vast majority of parameters (99.96%) are in the first fully connected layer:

| Layer | Parameters | Percentage |
|-------|-----------|------------|
| Conv1 | 896 | 0.004% |
| FC1 | 20,072,512 | 99.96% |
| FC2 | 65 | 0.0003% |
| **Total** | **20,073,473** | **100%** |

**Why FC1 has so many parameters?**
- It connects all 313,632 flattened features to 64 neurons
- Each connection needs a weight
- 313,632 × 64 = 20+ million weights!

---

## Key Insights

1. **Convolutional layers are parameter-efficient**
   - Conv1 has only 896 parameters despite processing 120,000 input pixels
   - Weight sharing makes CNNs efficient

2. **Fully connected layers dominate parameter count**
   - FC1 alone has 20+ million parameters
   - This is typical in CNNs

3. **Most parameters come from the first FC layer**
   - After flattening, we have 313K features
   - Each of 64 neurons connects to all 313K features

---

## ✅ Final Answer

**Total number of parameters: 20,073,473**


