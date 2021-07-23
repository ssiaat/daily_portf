import os
import threading

print(f'Keras Backend : {os.environ["KERAS_BACKEND"]}')

from keras.models import Model, Sequential
from keras.layers import Dense,BatchNormalization, concatenate
from tensorflow.keras.initializers import he_uniform, he_normal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import mean_squared_error as mse
from keras import Input
import tensorflow as tf

class Network:
    lock = threading.Lock()

    def __init__(self, input_dim=0, output_dim=0, num_ticker=100, trainable=False, lr=0.001, activation=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_ticker = num_ticker
        self.trainable = trainable
        self.loss = mse
        self.model = None
        self.initializer = he_normal()
        self.activation = 'relu'
        self.activation_last = activation
        self.optimizer = Adam(lr)

    def predict(self, sample):
        with self.lock:
            return tf.squeeze(self.model(sample))

    def learn(self, x, y):
        with tf.GradientTape() as tape:
            # 가치 신경망 갱신
            output = self.model(x)
            loss = self.loss(y, output)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return tf.reduce_mean(loss)

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

        output = self.get_network_head(inp).output
        output = Dense(
            self.output_dim, activation=self.activation_last,
            kernel_initializer=self.initializer)(output)
        self.model = Model(inp, output)

    def residual_layer(self, inp, hidden_size):
        output_r = Dense(hidden_size, activation=self.activation, kernel_initializer=self.initializer)(inp)
        output_r = BatchNormalization(trainable=self.trainable)(output_r)
        output = Dense(hidden_size, activation=self.activation, kernel_initializer=self.initializer)(output_r)
        output = BatchNormalization(trainable=self.trainable)(output)
        output = Dense(hidden_size, activation=self.activation, kernel_initializer=self.initializer)(output)
        output = BatchNormalization(trainable=self.trainable)(output)
        return output + output_r

    def mini_dnn(self, inp):
        output = self.residual_layer(inp, 128)
        output = self.residual_layer(output, 32)
        return output

    def get_network_head(self, inp):
        output = concatenate([self.mini_dnn(tf.reshape(i, (-1, self.input_dim))) for i in inp])
        # output = self.residual_layer(output, 2048)
        output = self.residual_layer(output, 1024)
        output = self.residual_layer(output, 512)
        output = Dense(256, activation=self.activation, kernel_initializer=self.initializer)(output)
        output = BatchNormalization(trainable=self.trainable)(output)
        return Model(inp, output)