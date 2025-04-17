# Logistic Regression from Scratch — Cat vs Non-Cat Classification

This project is a **binary image classification model** implemented from scratch using **NumPy and h5py**. It trains a **logistic regression model** to distinguish between images of **cats** and **non-cats**, without using any machine learning libraries like Scikit-learn or TensorFlow.

---

## Dataset

The model uses the `catvsnoncat.h5` dataset which contains:
- `train_set_x`: Training images of shape (m_train, height, width, 3)
- `train_set_y`: Training labels (0 = non-cat, 1 = cat)
- `test_set_x`: Test images
- `test_set_y`: Test labels

Files used:
- `train_catvsnoncat.h5`
- `test_catvsnoncat.h5`

---

## Logistic Regression Overview

Logistic Regression uses the **sigmoid function**:

<div align="center">

![sigmoid](https://latex.codecogs.com/svg.image?\Large&space;\sigma(z)&space;=&space;\frac{1}{1&plus;e^{-z}})

</div>

Where:

\[
z = w^T x + b
\]

The model is trained using **Gradient Descent** by updating parameters:

<div align="center">

![grad](https://latex.codecogs.com/svg.image?\Large&space;w:=w-\alpha\cdot\frac{1}{m}\cdot\sum_{i=1}^{m}(a^{(i)}-y^{(i)})x^{(i)})

![grad-b](https://latex.codecogs.com/svg.image?\Large&space;b:=b-\alpha\cdot\frac{1}{m}\cdot\sum_{i=1}^{m}(a^{(i)}-y^{(i)}))

</div>

---

## Features

- Loads `.h5` image dataset
- Preprocesses RGB images into column vectors
- Implements forward & backward propagation manually
- Optimizes using gradient descent
- Predicts and evaluates accuracy
- Entire pipeline written from scratch

---

## Data Preprocessing
Each image is originally shaped as (height, width, 3) (RGB). It is flattened into a column vector like this:
<div align="center">

(https://latex.codecogs.com/svg.image?x_{1}=\begin{bmatrix}x_{1R}\\|\\x_{1G}\\|\\x_{1B}\\|\\\end{bmatrix})

</div>

Then, all image vectors are stacked together to form a matrix X where each column represents one image:
<div align="center">

(https://latex.codecogs.com/svg.image?&space;x=\begin{bmatrix}|&|&|...\\x_{1}&x_{2}&x_{3}...\\|&|&|...\\\end{bmatrix})
</div>

---

## Model Architecture

```plaintext
Input Layer (flattened image vector)
       ↓
Linear Combination: z = wᵀx + b
       ↓
Activation: a = sigmoid(z)
       ↓
Loss: cross-entropy
       ↓
Backpropagation: compute gradients
       ↓
Gradient Descent: update weights and bias
```

## Requirements
-Python3
-Numpy
-h5py

## How to run
Run the given command
```Bash
python logistic_regression.py
```
Make sure test_catvsnoncat.h5 and train_catvsnoncat.h5 are in the same folder

##Sample Output
```plaintext
model trained
Total data: 50
Correct prediction number: 36
The accuracy of model is: 72.0%
```





