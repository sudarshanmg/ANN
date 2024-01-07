# ANN in Python from SCRATCH

This project implements a neural network class without using any ML library.

# Project Structure

```plaintext
.
|-- ANN/
|   |-- Layer.py
|   |-- Dense.py
|   |-- Network.py
|   |-- Activation.py
|   |-- Activation_functions/
|       |-- Tanh.py
|   |-- Loss_functions/
|       |-- MSE.py
|-- mnist_digits.py
```

- **`ANN/`**: Directory for your neural network classes.

  - **`Layer.py`**: Implementation of the Base `Layer` class.
  - **`Dense.py`**: Implementation of the `Dense` class.
  - **`Network.py`**: Implementation of the `Network` class.
  - **`Activation.py`**: Implementation of the `Activation` class.
  - **`Activation_functions/`**: Contains activation function implementations.

    - **`Tanh.py`**: Implementation of the hyperbolic tangent activation function.

  - **`Loss_functions/`**: Contains loss function implementations.

    - **`MSE.py`**: Implementation of the Mean Squared Error loss function.

- **`mnist_digits.py`**: Main script or application where you use the neural network.

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/sudarshanmg/ANN.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the MNIST digits script:

   ```bash
   python mnist_digits.py
   ```

## Usage

- Modify and run `mnist_digits.py` to experiment with the neural network on the MNIST dataset.

## Tutorial

Head to the [Tutorial Section](TUTORIAL.md).

## Contributing

Feel free to contribute to the development of this project. Create an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
