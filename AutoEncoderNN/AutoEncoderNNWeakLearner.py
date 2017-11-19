from Interface.IWeakLearner import IWeakLearner
from keras.models import Model, load_model, save_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from scipy import stats
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import json
import base64


class AutoEncoderNNWeakLearner(IWeakLearner):
    def __init__(self, cols_shape=10, training_epochs=100, batch_size=16, validation_size=0.2):
        super().__init__()
        self.cols_shape = cols_shape
        self.encoding_dim = int(cols_shape / 2)
        self.autoencoder = None
        self.epochs = training_epochs
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.norms = None

    def init_weak_learner(self):
        input_layer = Input(shape=(self.cols_shape,))

        encoder = Dense(self.encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_layer)
        encoder = Dense(int(self.encoding_dim / 2), activation='relu')(encoder)
        # encoder = Dense(int(self.encoding_dim / 4), activation='relu')(encoder)

        # decoder = Dense(int(self.encoding_dim / 2), activation='relu')(encoder)
        decoder = Dense(self.encoding_dim, activation='relu')(encoder)
        decoder = Dense(self.cols_shape, activation='sigmoid')(decoder)

        # encoder = Dense(self.encoding_dim, activation="relu",
        #                 activity_regularizer=regularizers.l1(10e-5))(input_layer)
        # encoder = Dense(int(self.encoding_dim / 2), activation="relu")(encoder)

        # decoder = Dense(int(self.encoding_dim / 2), activation='sigmoid')(encoder)
        # decoder = Dense(int(self.encoding_dim), activation='tanh')(decoder)
        # decoder = Dense(self.cols_shape, activation='relu')(decoder)

        self.autoencoder = Model(inputs=input_layer, outputs=decoder)
        self.autoencoder.compile(optimizer='sgd', loss='mean_absolute_error', metrics=['mae', 'accuracy'])

    def train(self, X):
        # normalize X
        X, self.norms = normalize(X, axis=0, norm='max', return_norm=True)
        X_train, X_test = train_test_split(X, test_size=self.validation_size, random_state=42)
        self.autoencoder.fit(X_train, X_train,
                             epochs=self.epochs,
                             batch_size=self.batch_size,
                             shuffle=True,
                             validation_data=(X_test, X_test),
                             verbose=0)

        # Test
        predictions = self.autoencoder.predict(X_test)
        print("Predicitions = " + str(predictions))
        mse = np.mean(np.power(X_test - predictions, 2), axis=1)
        print("MSE = " + str(mse))
        return (mse, predictions)

    def predict(self, x):
        x /= self.norms[np.newaxis, :]
        pred = self.autoencoder.predict(x)
        mse = np.mean(np.power(x - pred, 2), axis=1)
        return (mse, pred)

    def reset(self):
        self.init_weak_learner()

    def save_model(self, path):
        save_model(self.autoencoder, path)
        print(self.norms)
        return {'cols_shape': self.cols_shape, 'encoding_dim': self.encoding_dim,
                'epochs': self.epochs, 'validation_size': self.validation_size,
                'batch_size': self.batch_size,
                'norms': json.dumps(self.norms.tolist())}

    def load_model(self, path, metadata):
        self.autoencoder = load_model(path)
        self.autoencoder._make_predict_function()
        self.cols_shape = metadata['cols_shape']
        self.encoding_dim = metadata['encoding_dim']
        self.validation_size = metadata['validation_size']
        self.batch_size = metadata['batch_size']
        self.epochs = metadata['epochs']
        self.norms = np.array(json.loads(metadata['norms']))
