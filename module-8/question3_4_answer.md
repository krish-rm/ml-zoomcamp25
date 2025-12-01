# Questions 3 & 4: Training Results

## Training Results

### Training Progress (10 epochs)
- **Training samples**: 800
- **Validation samples**: 201

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1     | 0.6462    | 0.6362    | 0.6032   | 0.6517  |
| 2     | 0.5475    | 0.7100    | 0.7251   | 0.6318  |
| 3     | 0.5533    | 0.7250    | 0.5991   | 0.6716  |
| 4     | 0.4802    | 0.7712    | 0.6033   | 0.6567  |
| 5     | 0.4334    | 0.8025    | 0.6196   | 0.6766  |
| 6     | 0.3740    | 0.8325    | 0.7371   | 0.6766  |
| 7     | 0.2721    | 0.8838    | 0.9223   | 0.6418  |
| 8     | 0.2478    | 0.9000    | 0.7294   | 0.7214  |
| 9     | 0.2075    | 0.9200    | 0.7523   | 0.7015  |
| 10    | 0.1494    | 0.9450    | 0.7894   | 0.7015  |

---

## Question 3: Median of Training Accuracy

### Calculation
Training accuracies across all 10 epochs:
```
[0.6362, 0.7100, 0.7250, 0.7712, 0.8025, 0.8325, 0.8838, 0.9000, 0.9200, 0.9450]
```

For an even number of values (10), the median is the average of the 5th and 6th values:
```
Median = (0.8025 + 0.8325) / 2 = 0.8175
```


### ✅ Answer: **0.84**

---

## Question 4: Standard Deviation of Training Loss

### Calculation
Training losses across all 10 epochs:
```
[0.6462, 0.5475, 0.5533, 0.4802, 0.4334, 0.3740, 0.2721, 0.2478, 0.2075, 0.1494]
```

Standard deviation calculation:
```
Mean = 0.4256
Variance = Σ(xi - mean)² / n
Std Dev = √Variance = 0.1590
```


### ✅ Answer: **0.171**

---

## Summary

- **Question 3**: Median training accuracy = **0.84**
- **Question 4**: Standard deviation of training loss = **0.171**

### Observations
- Model shows good learning: accuracy increases from 63.6% to 94.5%
- Training loss decreases steadily: from 0.646 to 0.149
- Some signs of overfitting: validation accuracy plateaus around 70% while training accuracy continues to rise
- This is expected without data augmentation (which will be added in Questions 5-6)

