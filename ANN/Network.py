import joblib
import os
import time
from sklearn.metrics import accuracy_score
import numpy as np



class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set which loss_function to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime
    
    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate, batch_size, model_cache_file=None):

        if model_cache_file and os.path.exists(model_cache_file):
            print(f"Loading model from {model_cache_file}...")
            self.load_model(model_cache_file)
            print("Model loaded successfully.")

        else:
            print("Training the model...")
            start_time = time.time()

            # sample dimension first
            samples = len(x_train)

            # training loop
            for i in range(epochs):
                err = 0
                indices = np.random.permutation(samples)

                for j in range(0, samples, batch_size):
                    mini_batch_indices = indices[j:j + batch_size]
                    mini_batch_x = x_train[mini_batch_indices]
                    mini_batch_y = y_train[mini_batch_indices]

                    # average error for the mini-batch
                    batch_err = 0

                    for k in range(len(mini_batch_x)):
                        # forward propagation
                        output = mini_batch_x[k]
                        for layer in self.layers:
                            output = layer.forward_propagation(output)

                        # compute loss
                        batch_err += self.loss(mini_batch_y[k], output)

                        # backward propagation
                        error = self.loss_prime(mini_batch_y[k], output)
                        for layer in reversed(self.layers):
                            error = layer.backward_propagation(
                                error, learning_rate)

                    # average error for the mini-batch
                    batch_err /= len(mini_batch_x)
                    err += batch_err

                # calculate average error on all mini-batches
                err /= len(range(0, samples, batch_size))
                print('epoch %d/%d   error=%f' % (i + 1, epochs, err))

            if model_cache_file:
                print(f"Saving the model to {model_cache_file}")
                self.save_model(model_cache_file)
                print("Model saved successfully.")
            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60
            print(
                f"Time taken to train the network: {elapsed_time:.2f} mins")
            
    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(np.argmax(output))

        return result

    # Calculate accuracy
    def evaluate_accuracy(self, x_data, y_data):
        y_true = []
        y_pred = self.predict(x_data)

        for i in range(len(x_data)):
            y_true.append(np.argmax(y_data[i]))

        accuracy = accuracy_score(y_true, y_pred)
        print("Accuracy: ", round(accuracy*100, 3), "%")

    # Save the model locally
    def save_model(self, filename):
        model_params = {'layers': self.layers}
        joblib.dump(model_params, filename)

    # Load the midel
    def load_model(self, filename):
        model_params = joblib.load(filename)
        self.layers = model_params['layers']

