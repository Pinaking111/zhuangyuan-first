# Part 3: Questions and Answers

### Question 1
Input: 32×32×3, Conv(8 filters, 5×5, stride=1, no padding)
Output size = 28×28×8
Calculation: (32-5)/1 + 1 = 28 (spatial dimensions)

### Question 2
With "same" padding, output size = 32×32×8
The spatial dimensions remain unchanged.

### Question 3
Input: 64×64, Filter: 3×3, stride=2, no padding
Output: 31×31
Calculation: (64-3)/2 + 1 = 31

### Question 4
Input: 16×16, MaxPool(2×2, stride=2)
Output: 8×8
Calculation: (16-2)/2 + 1 = 8

### Question 5
Input: 128×128
After two conv layers (3×3, stride=1, same padding): 128×128
Spatial dimensions unchanged due to same padding.

### Question 6
Removing model.train():
- Dropout layers won't drop neurons
- BatchNorm uses running stats instead of batch stats
- Training statistics not updated
Result: Model performs inference instead of training