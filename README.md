# ANN in Python from SCRATCH

This project implements a neural network for the MNIST dataset using Python.

## Project Structure

- `ANN/`: Directory containing neural network components
  - `Activation_functions/`: Implementation of activation functions
  - `Loss_functions/`: Implementation of loss functions
  - `Activation.py`: Implementation of the activation layer
  - `Network.py`: Implementation of the neural network
  - `Layer.py`: Base class for network layers
- `mnist_digits.py`: Main script for working with the MNIST dataset
- `model_cache.joblib`: Cached model file
- `model_dataset.joblib`: Cached dataset file

## Getting Started

1. Clone the repository:

   ```bash
   git clone <repository-url>
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
