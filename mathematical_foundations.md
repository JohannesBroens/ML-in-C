# Mathematical Foundations

This document provides an explanation of the mathematical principles underlying the machine learning models implemented in this project, focusing primarily on the Multi-Layer Perceptron (MLP). It includes detailed mathematical derivations and proofs to offer a deeper understanding of the algorithms.

## Table of Contents

- [Mathematical Foundations](#mathematical-foundations)
  - [Table of Contents](#table-of-contents)
- [Pseudo-Code for Multi-Layer Perceptron (MLP) Training](#pseudo-code-for-multi-layer-perceptron-mlp-training)
  - [MLP Training Algorithm](#mlp-training-algorithm)
  - [Detailed Steps](#detailed-steps)
    - [1. Initialization](#1-initialization)
    - [2. Forward Propagation](#2-forward-propagation)
    - [3. Backpropagation](#3-backpropagation)
    - [4. Gradient Computation](#4-gradient-computation)
    - [5. Parameter Update](#5-parameter-update)
  - [Mathematical Symbols and Notations](#mathematical-symbols-and-notations)
  - [Multi-Layer Perceptron (MLP)](#multi-layer-perceptron-mlp)
    - [Model Architecture](#model-architecture)
    - [Forward Propagation](#forward-propagation)
    - [Activation Functions](#activation-functions)
    - [Loss Function](#loss-function)
    - [Backpropagation Algorithm](#backpropagation-algorithm)
    - [Gradient Descent Optimization](#gradient-descent-optimization)
    - [Detailed Proof of Backpropagation](#detailed-proof-of-backpropagation)
      - [Derivative of the Loss with Respect to Output Layer Weights $\\mathbf{W}^{(2)}$](#derivative-of-the-loss-with-respect-to-output-layer-weights-mathbfw2)
      - [Derivative of the Loss with Respect to Hidden Layer Weights $\\mathbf{W}^{(1)}$](#derivative-of-the-loss-with-respect-to-hidden-layer-weights-mathbfw1)
      - [Summary](#summary)
  - [Convolutional Neural Networks (CNNs)](#convolutional-neural-networks-cnns)
    - [Convolution Operation](#convolution-operation)
    - [Pooling Layers](#pooling-layers)
  - [GPU Acceleration vs. CPU Parallelization](#gpu-acceleration-vs-cpu-parallelization)

---

# Pseudo-Code for Multi-Layer Perceptron (MLP) Training

This pseudo-code outlines the training process of a Multi-Layer Perceptron (MLP) model using mathematical notation. It captures the essence of forward propagation, loss computation, backpropagation, and parameter updates.

## MLP Training Algorithm

```math
\begin{align*}
&\textbf{Inputs:} \
&\quad \text{Training dataset: } \{ (\mathbf{x}^{(i)}, \mathbf{y}^{(i)}) \}_{i=1}^{N} \
&\quad \text{Number of epochs: } T \
&\quad \text{Learning rate: } \eta \
&\quad \text{Network architecture:} \
&\quad \quad \text{Input size: } n \
&\quad \quad \text{Hidden layer size: } m \
&\quad \quad \text{Output size: } k \
&\textbf{Initialize Parameters:} \
&\quad \mathbf{W}^{(1)} \in \mathbb{R}^{m \times n} \sim \mathcal{N}(0, \sigma^2) \
&\quad \mathbf{b}^{(1)} \in \mathbb{R}^{m} \leftarrow \mathbf{0} \
&\quad \mathbf{W}^{(2)} \in \mathbb{R}^{k \times m} \sim \mathcal{N}(0, \sigma^2) \
&\quad \mathbf{b}^{(2)} \in \mathbb{R}^{k} \leftarrow \mathbf{0} \
&\textbf{Training Loop:} \
&\quad \text{FOR } \text{epoch} = 1 \text{ TO } T \text{ DO} \
&\quad \quad \text{FOR each training example } (\mathbf{x}, \mathbf{y}) \text{ DO} \
&\quad \quad \quad \textbf{Forward Propagation:} \
&\quad \quad \quad \quad \mathbf{z}^{(1)} = \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)} \
&\quad \quad \quad \quad \mathbf{h} = f(\mathbf{z}^{(1)}) \
&\quad \quad \quad \quad \mathbf{z}^{(2)} = \mathbf{W}^{(2)} \mathbf{h} + \mathbf{b}^{(2)} \
&\quad \quad \quad \quad \hat{\mathbf{y}} = g(\mathbf{z}^{(2)}) \
&\quad \quad \quad \quad L = \text{Loss}(\mathbf{y}, \hat{\mathbf{y}}) \
&\quad \quad \quad \textbf{Backpropagation:} \
&\quad \quad \quad \quad \delta^{(2)} = \nabla_{\hat{\mathbf{y}}} L \odot g'(\mathbf{z}^{(2)}) \
&\quad \quad \quad \quad \delta^{(1)} = (\mathbf{W}^{(2)^\top} \delta^{(2)}) \odot f'(\mathbf{z}^{(1)}) \
&\quad \quad \quad \textbf{Gradient Computation:} \
&\quad \quad \quad \quad \nabla_{\mathbf{W}^{(2)}} L = \delta^{(2)} \mathbf{h}^\top \
&\quad \quad \quad \quad \nabla_{\mathbf{b}^{(2)}} L = \delta^{(2)} \
&\quad \quad \quad \quad \nabla_{\mathbf{W}^{(1)}} L = \delta^{(1)} \mathbf{x}^\top \
&\quad \quad \quad \quad \nabla_{\mathbf{b}^{(1)}} L = \delta^{(1)} \
&\quad \quad \quad \textbf{Parameter Update:} \
&\quad \quad \quad \quad \mathbf{W}^{(2)} \leftarrow \mathbf{W}^{(2)} - \eta \nabla_{\mathbf{W}^{(2)}} L \
&\quad \quad \quad \quad \mathbf{b}^{(2)} \leftarrow \mathbf{b}^{(2)} - \eta \nabla_{\mathbf{b}^{(2)}} L \
&\quad \quad \quad \quad \mathbf{W}^{(1)} \leftarrow \mathbf{W}^{(1)} - \eta \nabla_{\mathbf{W}^{(1)}} L \
&\quad \quad \quad \quad \mathbf{b}^{(1)} \leftarrow \mathbf{b}^{(1)} - \eta \nabla_{\mathbf{b}^{(1)}} L \
&\quad \quad \text{END FOR} \
&\quad \text{END FOR} \
&\textbf{Output:} \
&\quad \text{Trained parameters } \mathbf{W}^{(1)}, \mathbf{b}^{(1)}, \mathbf{W}^{(2)}, \mathbf{b}^{(2)}
\end{align*}
```
## Detailed Steps

### 1. Initialization
```math
\begin{align*}
\mathbf{W}^{(1)} &\sim \mathcal{N}(0, \sigma^2) \quad \text{(Initialize input-to-hidden weights)} \\
\mathbf{b}^{(1)} &= \mathbf{0} \quad \text{(Initialize hidden layer biases)} \\
\mathbf{W}^{(2)} &\sim \mathcal{N}(0, \sigma^2) \quad \text{(Initialize hidden-to-output weights)} \\
\mathbf{b}^{(2)} &= \mathbf{0} \quad \text{(Initialize output layer biases)}
\end{align*}
```

### 2. Forward Propagation
```math
\begin{align*}
\mathbf{z}^{(1)} &= \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)} \\
\mathbf{h} &= f(\mathbf{z}^{(1)}) \\
\mathbf{z}^{(2)} &= \mathbf{W}^{(2)} \mathbf{h} + \mathbf{b}^{(2)} \\
\hat{\mathbf{y}} &= g(\mathbf{z}^{(2)}) \\
L &= \text{Loss}(\mathbf{y}, \hat{\mathbf{y}})
\end{align*}
```

### 3. Backpropagation
```math
\begin{align*}
\delta^{(2)} &= \nabla_{\hat{\mathbf{y}}} L \odot g'(\mathbf{z}^{(2)}) \\
\delta^{(1)} &= (\mathbf{W}^{(2)^\top} \delta^{(2)}) \odot f'(\mathbf{z}^{(1)})
\end{align*}
```

### 4. Gradient Computation
```math
\begin{align*}
\nabla_{\mathbf{W}^{(2)}} L &= \delta^{(2)} \mathbf{h}^\top \\
\nabla_{\mathbf{b}^{(2)}} L &= \delta^{(2)} \\
\nabla_{\mathbf{W}^{(1)}} L &= \delta^{(1)} \mathbf{x}^\top \\
\nabla_{\mathbf{b}^{(1)}} L &= \delta^{(1)}
\end{align*}
```

### 5. Parameter Update
```math
\begin{align*}
\mathbf{W}^{(2)} &\leftarrow \mathbf{W}^{(2)} - \eta \nabla_{\mathbf{W}^{(2)}} L \\
\mathbf{b}^{(2)} &\leftarrow \mathbf{b}^{(2)} - \eta \nabla_{\mathbf{b}^{(2)}} L \\
\mathbf{W}^{(1)} &\leftarrow \mathbf{W}^{(1)} - \eta \nabla_{\mathbf{W}^{(1)}} L \\
\mathbf{b}^{(1)} &\leftarrow \mathbf{b}^{(1)} - \eta \nabla_{\mathbf{b}^{(1)}} L
\end{align*}
```

## Mathematical Symbols and Notations
- $\mathbf{x} \in \mathbb{R}^{n}$: Input vector.
- $\mathbf{y} \in \mathbb{R}^{k}$: True label (one-hot encoded).
- $\hat{\mathbf{y}} \in \mathbb{R}^{k}$: Predicted output vector.
- $f$, $g$: Activation functions (e.g., ReLU, Softmax).
- $L$: Loss function (e.g., Cross-Entropy Loss).
- $\eta$: Learning rate.
- $\odot$: Element-wise multiplication.
- $\mathcal{N}(0, \sigma^2)$: Normal distribution with mean $0$ and variance $\sigma^2$.

---


## Multi-Layer Perceptron (MLP)

An MLP is a type of feedforward artificial neural network that consists of multiple layers of nodes in a directed graph, with each layer fully connected to the next one. MLPs are capable of approximating complex nonlinear functions and are widely used for classification and regression tasks.

### Model Architecture

An MLP typically consists of:

- **Input Layer**: Receives the input features.
- **Hidden Layers**: One or more layers where computations are performed.
- **Output Layer**: Produces the final output (e.g., class probabilities).

Mathematically, an MLP with one hidden layer can be represented as:

```math
\begin{align*}
\mathbf{h} &= f\left(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}\right) \\
\mathbf{\hat{y}} &= g\left(\mathbf{W}^{(2)} \mathbf{h} + \mathbf{b}^{(2)}\right)
\end{align*}
```

Where:

- $\mathbf{x} \in \mathbb{R}^{n}$: Input vector.
- $\mathbf{W}^{(1)} \in \mathbb{R}^{m \times n}$: Weight matrix for the input-to-hidden layer.
- $\mathbf{b}^{(1)} \in \mathbb{R}^{m}$: Bias vector for the hidden layer.
- $f$: Activation function for the hidden layer.
- $\mathbf{h} \in \mathbb{R}^{m}$: Hidden layer activations.
- $\mathbf{W}^{(2)} \in \mathbb{R}^{k \times m}$: Weight matrix for the hidden-to-output layer.
- $\mathbf{b}^{(2)} \in \mathbb{R}^{k}$: Bias vector for the output layer.
- $g$: Activation function for the output layer.
- $\mathbf{\hat{y}} \in \mathbb{R}^{k}$: Predicted output vector.

### Forward Propagation

Forward propagation involves computing the output of the network given an input. It is a sequence of matrix multiplications and function applications.

1. **Hidden Layer Computation**:

   ```math
   \mathbf{z}^{(1)} = \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}
   ```

   ```math
   \mathbf{h} = f\left(\mathbf{z}^{(1)}\right)
   ```

2. **Output Layer Computation**:

   ```math
   \mathbf{z}^{(2)} = \mathbf{W}^{(2)} \mathbf{h} + \mathbf{b}^{(2)}
   ```

   ```math
   \mathbf{\hat{y}} = g\left(\mathbf{z}^{(2)}\right)
   ```

### Activation Functions

Activation functions introduce non-linearity into the network, allowing it to learn complex patterns.

- **ReLU (Rectified Linear Unit)**:
  ```math
  f(z) = \max(0, z)
  ```

- **Sigmoid Function**:
  ```math
  g(z) = \frac{1}{1 + e^{-z}}
  ```

- **Softmax Function** (for multi-class classification):
  ```math
  g_i(\mathbf{z}) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
  ```

### Loss Function
The loss function quantifies the difference between the predicted output and the true output.
- **Mean Squared Error (MSE)** (for regression):
```math
L(\mathbf{y}, \mathbf{\hat{y}}) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
```
- **Cross-Entropy Loss** (for classification):
```math
L(\mathbf{y}, \mathbf{\hat{y}}) = -\sum_{i=1}^{k} y_i \log(\hat{y}_i)
```
  Where:
  - $\mathbf{y}$: True labels (one-hot encoded).
  - $\mathbf{\hat{y}}$: Predicted probabilities.

### Backpropagation Algorithm
Backpropagation is an algorithm used to compute the gradient of the loss function with respect to the weights of the network. It applies the chain rule of calculus to compute these gradients efficiently.
1. **Compute Output Error**:
```math
\delta^{(2)} = \nabla_{\mathbf{\hat{y}}} L \odot g'\left(\mathbf{z}^{(2)}\right)
```
   - $\delta^{(2)} \in \mathbb{R}^{k}$: Error at the output layer.
   - $\mathbf{z}^{(2)} = \mathbf{W}^{(2)} \mathbf{h} + \mathbf{b}^{(2)}$
   - $g'$: Derivative of the activation function $g$.
   - $\odot$: Element-wise multiplication.

2. **Compute Hidden Layer Error**:
```math
\delta^{(1)} = \left(\mathbf{W}^{(2)^\top} \delta^{(2)}\right) \odot f'\left(\mathbf{z}^{(1)}\right)
```
   - $\delta^{(1)} \in \mathbb{R}^{m}$: Error at the hidden layer.
   - $\mathbf{z}^{(1)} = \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}$
   - $f'$: Derivative of the activation function $f$.

3. **Compute Gradients**:
```math
\begin{align*}
\nabla_{\mathbf{W}^{(2)}} L &= \delta^{(2)} \mathbf{h}^\top \\
\nabla_{\mathbf{b}^{(2)}} L &= \delta^{(2)} \\
\nabla_{\mathbf{W}^{(1)}} L &= \delta^{(1)} \mathbf{x}^\top \\
\nabla_{\mathbf{b}^{(1)}} L &= \delta^{(1)}
\end{align*}
```
### Gradient Descent Optimization
Weights are updated using gradient descent:
```math
\theta = \theta - \eta \nabla_{\theta} L
```
- $\theta$: Model parameters (weights and biases).
- $\eta$: Learning rate.
- $\nabla_{\theta} L$: Gradient of the loss function with respect to $\theta$.

### Detailed Proof of Backpropagation

We will provide detailed mathematical derivations for the gradients with respect to the weights and biases in both layers.

#### Derivative of the Loss with Respect to Output Layer Weights $\mathbf{W}^{(2)}$

**Objective**: Compute $\nabla_{\mathbf{W}^{(2)}} L$.

**Proof**:

1. **Loss Function**:
   For a single training example, the loss function using cross-entropy loss is:
```math
L = -\sum_{i=1}^{k} y_i \log(\hat{y}_i)
```
2. **Predicted Output**:
   The predicted output is:
```math
\hat{y}_i = g_i(\mathbf{z}^{(2)}) = \frac{e^{z_i^{(2)}}}{\sum_{j=1}^{k} e^{z_j^{(2)}}}
```
3. **Compute $\frac{\partial L}{\partial z_i^{(2)}}$**
   The derivative of the loss with respect to $z_i^{(2)}$ is:
```math
\frac{\partial L}{\partial z_i^{(2)}} = \hat{y}_i - y_i
```

   **Proof**:

   - Using the chain rule:

```math
\frac{\partial L}{\partial z_i^{(2)}} = \sum_{j=1}^{k} \frac{\partial L}{\partial \hat{y}_j} \frac{\partial \hat{y}_j}{\partial z_i^{(2)}}
```

   - For cross-entropy loss and softmax activation, this simplifies to:

```math
\frac{\partial L}{\partial z_i^{(2)}} = \hat{y}_i - y_i
```

4. **Compute $\frac{\partial L}{\partial \mathbf{W}^{(2)}}$**

   The gradient with respect to the weights is:

```math
\frac{\partial L}{\partial \mathbf{W}^{(2)}} = \delta^{(2)} \mathbf{h}^\top
```

   Where $\delta^{(2)} = \hat{\mathbf{y}} - \mathbf{y}$.

#### Derivative of the Loss with Respect to Hidden Layer Weights $\mathbf{W}^{(1)}$

**Objective**: Compute $\nabla_{\mathbf{W}^{(1)}} L$.

**Proof**:

1. **Compute $\frac{\partial L}{\partial \mathbf{h}}$**

   From the chain rule:

```math
\frac{\partial L}{\partial \mathbf{h}} = \left(\mathbf{W}^{(2)}\right)^\top \delta^{(2)}
```

2. **Compute $\delta^{(1)}$**

   Applying the element-wise multiplication with the derivative of the activation function:

```math
\delta^{(1)} = \frac{\partial L}{\partial \mathbf{h}} \odot f'\left(\mathbf{z}^{(1)}\right)
```

3. **Compute $\frac{\partial L}{\partial \mathbf{W}^{(1)}}$**

   The gradient with respect to the weights is:

```math
\frac{\partial L}{\partial \mathbf{W}^{(1)}} = \delta^{(1)} \mathbf{x}^\top
```

#### Summary

By systematically applying the chain rule, we compute the gradients of the loss with respect to each parameter in the network. This allows us to update the weights and biases during training to minimize the loss function.

---

## Convolutional Neural Networks (CNNs)

While not the primary focus of this project, CNNs are another class of neural networks particularly effective for image and spatial data processing.

### Convolution Operation

The convolution operation applies a kernel (filter) over the input data to extract features.

Mathematically, for a 2D convolution:

```math
S(i,j) = (I * K)(i,j) = \sum_{m} \sum_{n} I(i - m, j - n) K(m, n)
```

- $I$: Input image.
- $K$: Kernel (filter).
- $S$: Output feature map.

### Pooling Layers

Pooling layers reduce the spatial dimensions of the data, helping to reduce overfitting and computation.

- **Max Pooling**:

```math
S(i, j) = \max_{(m, n) \in \mathcal{P}(i, j)} I(m, n)
```

- **Average Pooling**:

```math
S(i, j) = \frac{1}{|\mathcal{P}(i, j)|} \sum_{(m, n) \in \mathcal{P}(i, j)} I(m, n)
```

Where $\mathcal{P}(i, j)$ is the pooling region corresponding to output position $(i, j)$.

---

## GPU Acceleration vs. CPU Parallelization

While CPUs are optimized for sequential serial processing with complex control logic, GPUs are designed for parallel processing of large blocks of data, making them ideal for the computations involved in neural network training and inference.

---
