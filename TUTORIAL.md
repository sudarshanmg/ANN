# Neural Network Tutorial

In this tutorial, I'll guide you through creating a simple neural network using the classes implemented in the project. The neural network consists of layers, activations, and a loss function.

## Setup

Make sure you have the required Python environment set up and the necessary libraries installed. You can install the required libraries using:

```bash
pip install numpy scikit-learn
```

- scikit-learn has just been used to measure the accuracy.

## Classes Overview

### 1. `Dense`

The `Dense` layer represents a fully connected layer in the neural network.

#### Parameters:

- `input_size`: Number of input neurons.
- `output_size`: Number of output neurons.

#### Usage:

```python
from ANN.Layer import Dense

# Example: Creating a Dense layer
dense_layer = Dense(input_size=784, output_size=16)
```

### 2. `ActivationLayer`

The `ActivationLayer` applies an activation function to the layer's output.

#### Parameters:

- `activation`: Activation function.
- `activation_prime`: Derivative of the activation function.

#### Usage:

```python
from ANN.Activation import Activation
from ANN.Activation_functions.Tanh import tanh, tanh_prime

# Example: Creating an ActivationLayer with Tanh activation
activation_layer = Activation(activation=tanh, activation_prime=tanh_prime)
```

### 3. `Network`

The `Network` class represents the entire neural network, comprising layers and a loss function.

#### Usage:

```python
from ANN.Network import Network
from ANN.Loss_functions.MSE import mse, mse_prime

# Example: Creating a neural network
neural_network = Network()

# Adding layers to the network
neural_network.add(Dense(input_size=784, output_size=16))
neural_network.add(ActivationLayer(activation=tanh, activation_prime=tanh_prime))
neural_network.add(Dense(input_size=16, output_size=10))
neural_network.add(ActivationLayer(activation=tanh, activation_prime=tanh_prime))

# Setting the loss function
neural_network.use(loss=mse, loss_prime=mse_prime)
```

### 4. Training the Network

To train the network, you need to provide training data (`x_train` and `y_train`), specify the number of epochs, and set the learning rate.

#### Usage:

```python
# Training the network
neural_network.fit(x_train, y_train, epochs=35, learning_rate=0.1)
```

### 5. Making Predictions

You can use the trained network to make predictions on new data.

#### Usage:

```python
# Making predictions
predictions = neural_network.predict(new_data)
print(predictions)
```

## Conclusion

You've created a simple neural network using the classes provided in this project. Experiment with different architectures, activation functions, and hyperparameters to optimize your network's performance.
