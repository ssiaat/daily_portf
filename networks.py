import os
import threading
import numpy as np

def set_session(sess): pass

print('----------------Keras Backend : ')
print(os.environ['KERAS_BACKEND'])

from keras.models import Model, Sequential
from keras.layers import Dense,BatchNormalization, concatenate
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.keras.backend import set_session
from keras import Input
import tensorflow as tf

graph = tf.compat.v1.get_default_graph()
sess = tf.compat.v1.Session()


class Network:
    lock = threading.Lock()

    def __init__(self, input_dim=0, output_dim=0, num_ticker=100, trainable=False, lr=0.001, activation=None, loss='mse'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_ticker = num_ticker
        self.trainable = trainable
        self.loss = loss
        self.model = None
        self.initializer = he_uniform()
        self.activation = 'relu'
        self.activation_last = activation
        self.optimizer = Adam(lr)

    def predict(self, sample):
        with self.lock:
            # with graph.as_default():
            if sess is not None:
                set_session(sess)
            return self.model.predict(sample).flatten()

    def train_on_batch(self, x, y):
        loss = 0.
        with self.lock:
            # with graph.as_default():
            if sess is not None:
                set_session(sess)
            loss = self.model.train_on_batch(x, y)
        return loss

    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True)

    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)

class DNN(Network):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        inp = [Input(shape=(self.input_dim,)) for i in range(self.num_ticker)]

        # with graph.as_default():
        if sess is not None:
            set_session(sess)
        output = self.get_network_head(inp, self.num_ticker, self.trainable).output
        output = Dense(
            self.output_dim, activation=self.activation_last,
            kernel_initializer=self.initializer)(output)
        self.model = Model(inp, output)
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss)


    def mini_dnn(self, inp):
        output = Dense(256, activation=self.activation, kernel_initializer=self.initializer)(inp)
        output = BatchNormalization(trainable=self.trainable)(output)
        output = Dense(128, activation=self.activation, kernel_initializer=self.initializer)(output)
        output = BatchNormalization(trainable=self.trainable)(output)
        return output


    def get_network_head(self, inp, num_ticker, trainable):
        output = concatenate([self.mini_dnn(tf.reshape(i, (-1, self.input_dim))) for i in inp])
        output_r = Dense(1024, activation=self.activation,
                       kernel_initializer=self.initializer)(output)
        output_r = BatchNormalization(trainable=trainable)(output_r)
        output = Dense(1024, activation=self.activation,
                       kernel_initializer=self.initializer)(output_r)
        output = BatchNormalization(trainable=trainable)(output)
        output = Dense(1024, activation=self.activation,
                       kernel_initializer=self.initializer)(output)
        output = BatchNormalization(trainable=trainable)(output)
        output_r = Dense(512, activation=self.activation,
                       kernel_initializer=self.initializer)(output + output_r)
        output_r = BatchNormalization(trainable=trainable)(output_r)
        output = Dense(512, activation=self.activation,
                         kernel_initializer=self.initializer)(output_r)
        output = BatchNormalization(trainable=trainable)(output)
        output = Dense(512, activation=self.activation,
                         kernel_initializer=self.initializer)(output + output_r)
        output = BatchNormalization(trainable=trainable)(output)
        output = Dense(128, activation=self.activation,
                       kernel_initializer=self.initializer)(output)
        output = BatchNormalization(trainable=trainable)(output)
        return Model(inp, output)

    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.input_dim))
        return super().train_on_batch(x, y)

    def predict(self, sample):
        return super().predict(sample)