# Mathematical Foundations

This document provides a detailed explanation of the mathematical principles underlying the machine learning models implemented in this project, focusing primarily on the Multi-Layer Perceptron (MLP). It includes detailed mathematical derivations and proofs to offer a deeper understanding of the algorithms.

## Table of Contents

- [Mathematical Foundations](#mathematical-foundations)
  - [Table of Contents](#table-of-contents)
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

## Multi-Layer Perceptron (MLP)

An MLP is a type of feedforward artificial neural network that consists of multiple layers of nodes in a directed graph, with each layer fully connected to the next one. MLPs are capable of approximating complex nonlinear functions and are widely used for classification and regression tasks.

### Model Architecture

An MLP typically consists of:

- **Input Layer**: Receives the input features.
- **Hidden Layers**: One or more layers where computations are performed.
- **Output Layer**: Produces the final output (e.g., class probabilities).

Mathematically, an MLP with one hidden layer can be represented as:

$$
\begin{align*}
\mathbf{h} &= f\left(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}\right) \\
\mathbf{\hat{y}} &= g\left(\mathbf{W}^{(2)} \mathbf{h} + \mathbf{b}^{(2)}\right)
\end{align*}
$$

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

   $$
   \mathbf{z}^{(1)} = \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}
   $$

   $$
   \mathbf{h} = f\left(\mathbf{z}^{(1)}\right)
   $$

2. **Output Layer Computation**:

   $$
   \mathbf{z}^{(2)} = \mathbf{W}^{(2)} \mathbf{h} + \mathbf{b}^{(2)}
   $$

   $$
   \mathbf{\hat{y}} = g\left(\mathbf{z}^{(2)}\right)
   $$

### Activation Functions

Activation functions introduce non-linearity into the network, allowing it to learn complex patterns.

- **ReLU (Rectified Linear Unit)**:

  $$
  f(z) = \max(0, z)
  $$

- **Sigmoid Function**:

  $$
  g(z) = \frac{1}{1 + e^{-z}}
  $$

- **Softmax Function** (for multi-class classification):

  $$
  g_i(\mathbf{z}) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
  $$

### Loss Function

The loss function quantifies the difference between the predicted output and the true output.

- **Mean Squared Error (MSE)** (for regression):

  $$
  L(\mathbf{y}, \mathbf{\hat{y}}) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  $$

- **Cross-Entropy Loss** (for classification):

  $$
  L(\mathbf{y}, \mathbf{\hat{y}}) = -\sum_{i=1}^{k} y_i \log(\hat{y}_i)
  $$

  Where:

  - $\mathbf{y}$: True labels (one-hot encoded).
  - $\mathbf{\hat{y}}$: Predicted probabilities.

### Backpropagation Algorithm

Backpropagation is an algorithm used to compute the gradient of the loss function with respect to the weights of the network. It applies the chain rule of calculus to compute these gradients efficiently.

1. **Compute Output Error**:

   $$
   \delta^{(2)} = \nabla_{\mathbf{\hat{y}}} L \odot g'\left(\mathbf{z}^{(2)}\right)
   $$

   - $\delta^{(2)} \in \mathbb{R}^{k}$: Error at the output layer.
   - $\mathbf{z}^{(2)} = \mathbf{W}^{(2)} \mathbf{h} + \mathbf{b}^{(2)}$
   - $g'$: Derivative of the activation function $g$.
   - $\odot$: Element-wise multiplication.

2. **Compute Hidden Layer Error**:

   $$
   \delta^{(1)} = \left(\mathbf{W}^{(2)^\top} \delta^{(2)}\right) \odot f'\left(\mathbf{z}^{(1)}\right)
   $$

   - $\delta^{(1)} \in \mathbb{R}^{m}$: Error at the hidden layer.
   - $\mathbf{z}^{(1)} = \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}$
   - $f'$: Derivative of the activation function $f$.

3. **Compute Gradients**:

   $$
   \begin{align*}
   \nabla_{\mathbf{W}^{(2)}} L &= \delta^{(2)} \mathbf{h}^\top \\
   \nabla_{\mathbf{b}^{(2)}} L &= \delta^{(2)} \\
   \nabla_{\mathbf{W}^{(1)}} L &= \delta^{(1)} \mathbf{x}^\top \\
   \nabla_{\mathbf{b}^{(1)}} L &= \delta^{(1)}
   \end{align*}
   $$

### Gradient Descent Optimization

Weights are updated using gradient descent:

$$
\theta = \theta - \eta \nabla_{\theta} L
$$

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

   $$
   L = -\sum_{i=1}^{k} y_i \log(\hat{y}_i)
   $$

2. **Predicted Output**:

   The predicted output is:

   $$
   \hat{y}_i = g_i(\mathbf{z}^{(2)}) = \frac{e^{z_i^{(2)}}}{\sum_{j=1}^{k} e^{z_j^{(2)}}}
   $$

3. **Compute $\frac{\partial L}{\partial z_i^{(2)}}$**

   The derivative of the loss with respect to $z_i^{(2)}$ is:

   $$
   \frac{\partial L}{\partial z_i^{(2)}} = \hat{y}_i - y_i
   $$

   **Proof**:

   - Using the chain rule:

     $$
     \frac{\partial L}{\partial z_i^{(2)}} = \sum_{j=1}^{k} \frac{\partial L}{\partial \hat{y}_j} \frac{\partial \hat{y}_j}{\partial z_i^{(2)}}
     $$

   - For cross-entropy loss and softmax activation, this simplifies to:

     $$
     \frac{\partial L}{\partial z_i^{(2)}} = \hat{y}_i - y_i
     $$

4. **Compute $\frac{\partial L}{\partial \mathbf{W}^{(2)}}$**

   The gradient with respect to the weights is:

   $$
   \frac{\partial L}{\partial \mathbf{W}^{(2)}} = \delta^{(2)} \mathbf{h}^\top
   $$

   Where $\delta^{(2)} = \hat{\mathbf{y}} - \mathbf{y}$.

#### Derivative of the Loss with Respect to Hidden Layer Weights $\mathbf{W}^{(1)}$

**Objective**: Compute $\nabla_{\mathbf{W}^{(1)}} L$.

**Proof**:

1. **Compute $\frac{\partial L}{\partial \mathbf{h}}$**

   From the chain rule:

   $$
   \frac{\partial L}{\partial \mathbf{h}} = \left(\mathbf{W}^{(2)}\right)^\top \delta^{(2)}
   $$

2. **Compute $\delta^{(1)}$**

   Applying the element-wise multiplication with the derivative of the activation function:

   $$
   \delta^{(1)} = \frac{\partial L}{\partial \mathbf{h}} \odot f'\left(\mathbf{z}^{(1)}\right)
   $$

3. **Compute $\frac{\partial L}{\partial \mathbf{W}^{(1)}}$**

   The gradient with respect to the weights is:

   $$
   \frac{\partial L}{\partial \mathbf{W}^{(1)}} = \delta^{(1)} \mathbf{x}^\top
   $$

#### Summary

By systematically applying the chain rule, we compute the gradients of the loss with respect to each parameter in the network. This allows us to update the weights and biases during training to minimize the loss function.

---

## Convolutional Neural Networks (CNNs)

While not the primary focus of this project, CNNs are another class of neural networks particularly effective for image and spatial data processing.

### Convolution Operation

The convolution operation applies a kernel (filter) over the input data to extract features.

Mathematically, for a 2D convolution:

$$
S(i,j) = (I * K)(i,j) = \sum_{m} \sum_{n} I(i - m, j - n) K(m, n)
$$

- $I$: Input image.
- $K$: Kernel (filter).
- $S$: Output feature map.

### Pooling Layers

Pooling layers reduce the spatial dimensions of the data, helping to reduce overfitting and computation.

- **Max Pooling**:

  $$
  S(i, j) = \max_{(m, n) \in \mathcal{P}(i, j)} I(m, n)
  $$

- **Average Pooling**:

  $$
  S(i, j) = \frac{1}{|\mathcal{P}(i, j)|} \sum_{(m, n) \in \mathcal{P}(i, j)} I(m, n)
  $$

Where $\mathcal{P}(i, j)$ is the pooling region corresponding to output position $(i, j)$.

---

## GPU Acceleration vs. CPU Parallelization

While CPUs are optimized for sequential serial processing with complex control logic, GPUs are designed for parallel processing of large blocks of data, making them ideal for the computations involved in neural network training and inference.

---
