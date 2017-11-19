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
import pickle


class AutoEncoderNNWeakLearner(IWeakLearner):
    def __init__(self, cols_shape=10, training_epochs=100, batch_size=16, validation_size=0.2):
        super().__init__()
        self.cols_shape = cols_shape
        self.encoding_dim = int(cols_shape / 2)
        self.autoencoder = None
        self.epochs = training_epochs
        self.batch_size = batch_size
        self.validation_size = validation_size

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
        print("X = " +str(X))
        # normalize X
        X = normalize(X, axis=0, norm='max')
        X_train, X_test = train_test_split(X, test_size=self.validation_size, random_state=42)
        checkpointer = ModelCheckpoint(filepath="./user_model",
                                       verbose=0,
                                       save_best_only=True)
        print(X_train)
        self.autoencoder.fit(X_train, X_train,
                             epochs=self.epochs,
                             batch_size=self.batch_size,
                             shuffle=True,
                             validation_data=(X_test, X_test),
                             verbose=0,
                             callbacks=[checkpointer])

        # Test
        print("XTEST=" + str(X_test))
        predictions = self.autoencoder.predict(X_test)
        print(predictions)
        mse = np.mean(np.power(X_test - predictions, 2), axis=1)
        print("MSE = " + str(mse))
        print(len(mse))
        return (mse, predictions)

    def predict(self, x):
        print(x)
        pred = self.autoencoder.predict(x)
        mse = np.mean(np.power(x - pred, 2), axis=1)
        return (mse, pred)

    def reset(self):
        self.init_weak_learner()

    def save_model(self, path):
        save_model(self.autoencoder, path)

    def load_model(self, path):
        self.autoencoder = load_model(path)
