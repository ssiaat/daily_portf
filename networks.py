import os
import threading
import numpy as np

def set_session(sess): pass

print('----------------Keras Backend : ')
print(os.environ['KERAS_BACKEND'])

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense,\
        BatchNormalization, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.keras.backend import set_session
from tensorflow.keras import Input
import tensorflow as tf

graph = tf.compat.v1.get_default_graph()
sess = tf.compat.v1.Session()


class Network:
    lock = threading.Lock()

    def __init__(self, input_dim=0, output_dim=0, trainable=False, lr=0.001,
                 shared_network=None, activation='sigmoid', loss='mse'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.trainable = trainable
        self.shared_network = shared_network
        self.activation = activation
        self.loss = loss
        self.model = None
        self.optimizer = Adam(lr)

    def predict(self, sample):
        with self.lock:
            with graph.as_default():
                if sess is not None:
                    set_session(sess)
                return self.model.predict(sample).flatten()

    def train_on_batch(self, x, y):
        loss = 0.
        with self.lock:
            with graph.as_default():
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

    @classmethod
    def get_shared_network(cls, net='dnn', num_steps=1, input_dim=0, trainable=False):
        with graph.as_default():
            if sess is not None:
                set_session(sess)
            if net == 'dnn':
                return DNN.get_network_head(Input((input_dim,)), trainable)


class DNN(Network):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with graph.as_default():
            if sess is not None:
                set_session(sess)
            inp = None
            output = None
            if self.shared_network is None:
                inp = Input((self.input_dim,))
                output = self.get_network_head(inp, self.trainable).output
            else:
                inp = self.shared_network.input
                output = self.shared_network.output
            output = Dense(
                self.output_dim, activation=self.activation,
                kernel_initializer='random_normal')(output)
            self.model = Model(inp, output)
            self.model.compile(
                optimizer=self.optimizer, loss=self.loss)

    @staticmethod
    def get_network_head(inp, trainable):
        output = Dense(256, activation='sigmoid',
                       kernel_initializer='random_normal')(inp)
        output = BatchNormalization(trainable=trainable)(output)
        output = Dropout(0.1)(output)
        output = Dense(128, activation='sigmoid',
                       kernel_initializer='random_normal')(output)
        output = BatchNormalization(trainable=trainable)(output)
        output = Dropout(0.1)(output)
        output = Dense(64, activation='sigmoid',
                       kernel_initializer='random_normal')(output)
        output = BatchNormalization(trainable=trainable)(output)
        output = Dropout(0.1)(output)
        output = Dense(32, activation='sigmoid',
                       kernel_initializer='random_normal')(output)
        output = BatchNormalization(trainable=trainable)(output)
        output = Dropout(0.1)(output)
        return Model(inp, output)

    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.input_dim))
        return super().train_on_batch(x, y)

    def predict(self, sample):
        sample = np.array(sample).reshape((1, self.input_dim))
        return super().predict(sample)