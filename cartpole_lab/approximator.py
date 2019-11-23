from keras.models import Sequential
from keras.layers import Dense # Dropout, BatchNormalization
from keras.optimizers import Adam
import numpy as np

class Approximator:
    """A function approximator implemented with a deep neural net."""
    def __init__(self, num_inputs, num_outputs, learning_rate=0.01):
        self.model = Sequential()
        self.model.add(Dense(64, input_shape=(num_inputs,), activation='relu'))
        #self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(num_outputs, activation='linear'))

        adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer=adam, loss='mse', metrics=['accuracy','mse'])

    def predict_multi(self, X_batch):
        """Predict the outputs for multiple input values at once.
        X_batch: an np array of m number of X values to predict, of shape (num_inputs, m)
        where m is the number of items you'd like to predict.
        """
        return self.model.predict(X_batch)
    
    def train_multi(self, X, Y, batch_size=16, epochs=1, verbose=0):
        """
        Train the model with m samples.
        X: the input values, of shape (num_inputs, m)
        Y: the target values, of shape (num_outputs, m)
        """
        return self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    
    def predict(self, X):
        """Predict a single output given a single input.
        X: one set of X values to predict. X shape is (num_inputs,).
        """
        predictions = self.predict_multi(np.array([X])) # An array for m input values.
        return predictions[0] # We have a batch size of one here; return the first.
    
    def train(self, X, Y):
        """Train a single input/output pair.
        X: inputs of shape (num_inputs,)
        Y: target outputs of shape (num_outputs,)"""
        batch_X = np.array([X])
        batch_Y = np.array([Y])
        self.train_multi(batch_X, batch_Y, batch_size=1)