import numpy as np
from ANN.Network import Network 
from ANN.Activation import Activation
from ANN.Activation_functions import Tanh
from ANN.Loss_functions import MSE
from ANN.Dense import Dense
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path



# Load the dataset from cache (if available)
def load_mnist_cached():
    try:
        # Attempt to load the cached dataset
        mnist = joblib.load('mnist_dataset.joblib')
        print("Dataset loaded from cache.")
    except FileNotFoundError:
        # If the cache file is not found, fetch the dataset and save it to cache
        print("Cache not found. Fetching dataset...")
        mnist = fetch_openml('mnist_784', cache=True, as_frame=False)
        joblib.dump(mnist, 'mnist_dataset.joblib')
        print("Dataset fetched and cached.")

    return mnist



mnist = load_mnist_cached()
data, target = mnist.data, mnist.target
x_train, x_test, y_train, y_test = train_test_split(
    data, target, random_state=42)


x_train = x_train.reshape(x_train.shape[0], 1, 28 * 28)
x_train = x_train.astype('float32')
x_train /= 255


y_train = np.eye(10)[y_train.astype(int)].reshape((len(y_train), 10))

x_test = x_test.reshape(x_test.shape[0], 1, 28 * 28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np.eye(10)[y_test.astype(int)].reshape((len(y_test), 10))

model_cache_file = 'model_cache.joblib'
model_cache_file = Path(__file__).parent / 'model_cache.joblib'






# Network
net = Network()

net.add(Dense(28 * 28, 100))
net.add(Activation(Tanh.tanh, Tanh.tanh_prime))

net.add(Dense(100, 16))
net.add(Activation(Tanh.tanh, Tanh.tanh_prime))

net.add(Dense(16, 10))
net.add(Activation(Tanh.tanh, Tanh.tanh_prime))


net.use(MSE.mse, MSE.mse_prime)
net.fit(x_train, y_train, epochs=35, learning_rate=0.1,
        batch_size=32, model_cache_file=model_cache_file)

net.evaluate_accuracy(x_test, y_test)

