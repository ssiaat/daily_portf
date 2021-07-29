import os
import threading
import numpy as np

print(f'Keras Backend : {os.environ["KERAS_BACKEND"]}')

from keras.models import Model, Sequential
from keras.layers import Dense, BatchNormalization, LSTM, LayerNormalization, Concatenate, Dot
from tensorflow.keras.initializers import he_normal, glorot_normal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import mean_squared_error as mse
from keras import Input
import tensorflow as tf

class Network:
    lock = threading.Lock()

    def __init__(self, input_dim=0, output_dim=0, num_ticker=100, num_steps=5, trainable=False, lr=0.001, activation=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_ticker = num_ticker
        self.num_steps = num_steps
        self.trainable = trainable
        self.loss = mse
        self.model = None
        self.initializer = glorot_normal()
        self.activation = 'relu'
        self.activation_last = activation
        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(lr, 500, 0.96, True)
        self.optimizer = Adam(lr_scheduler, clipnorm=.01)

    def predict(self, sample):
        with self.lock:
            return tf.squeeze(self.model(sample))

    def learn(self, x, y, flag=True):
        with tf.GradientTape() as tape:
            # 가치 신경망 갱신
            output = self.model(x)
            loss = tf.sqrt(self.loss(y, output))
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
        self.sub_models = [self.mini_dnn() for _ in range(self.num_ticker)]
        inp = [Input(shape=(None, self.input_dim)) for _ in range(self.num_ticker)]
        self.model = self.get_network(inp)

    def residual_layer(self, inp, hidden_size):
        output_r = Dense(hidden_size, activation=self.activation, kernel_initializer=self.initializer)(inp)
        output_r = BatchNormalization(trainable=self.trainable)(output_r)
        output = Dense(hidden_size, activation=self.activation, kernel_initializer=self.initializer)(output_r)
        output = BatchNormalization(trainable=self.trainable)(output)
        output = Dense(hidden_size, activation=self.activation, kernel_initializer=self.initializer)(output)
        output = BatchNormalization(trainable=self.trainable)(output)
        return output + output_r

    def mini_dnn(self):
        model = Sequential()
        model.add(Dense(256, activation=self.activation, kernel_initializer=self.initializer))
        model.add(BatchNormalization(trainable=self.trainable))
        model.add(Dense(32, activation=self.activation, kernel_initializer=self.initializer))
        model.add(BatchNormalization(trainable=self.trainable))
        return model

    def get_network(self, inp):
        output = Concatenate()([m(tf.reshape(i, (-1, self.input_dim))) for i,m in zip(inp, self.sub_models)])
        output = self.residual_layer(output, 2048)
        output = self.residual_layer(output, 1024)
        output = Dense(256, activation=self.activation, kernel_initializer=self.initializer)(output)
        output = Dense(self.output_dim, activation=self.activation_last, kernel_initializer=self.initializer)(output)
        output = tf.reshape(output, (-1, self.output_dim))
        return Model(inp, output)

# refer to paper DTML(jaemin yoo)
# Accurate Multivariate Stock Movement Prediction via Data-Axis Transformer with Multi-Level Contexts
class AttentionLSTM(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_steps = num_steps
        self.hidden_size_lstm = 256
        inp = [Input((self.num_steps, self.input_dim)) for _ in range(self.num_ticker)]
        self.sub_models = [self.mini_model() for _ in range(self.num_ticker)]
        self.qkv_models = [self.qkv_model() for _ in range(3)]
        self.model = self.get_network(inp)

    # expand input at first time
    def shared_dnn(self, inp):
        return Dense(128, activation='tanh', kernel_initializer=self.initializer)(inp)

    # after expand input, calculate hidden state of input sequences
    def mini_model(self):
        model = Sequential()
        model.add(LSTM(self.hidden_size_lstm, dropout=0.1, return_sequences=True, stateful=False, kernel_initializer=self.initializer))
        model.add(LayerNormalization(trainable=self.trainable))
        model.add(LSTM(self.hidden_size_lstm * 2, dropout=0.1, return_sequences=True, stateful=False, kernel_initializer=self.initializer))
        model.add(LayerNormalization(trainable=self.trainable))
        model.add(LSTM(self.hidden_size_lstm, dropout=0.1, return_sequences=True, stateful=False, kernel_initializer=self.initializer))
        model.add(LayerNormalization(trainable=self.trainable))
        return model

    # calculate attention score from hidden states of each stocks
    def get_attention_score(self, hidden_states):
        last_hidden_state = hidden_states[-1]
        attention_score = tf.exp(last_hidden_state * hidden_states)
        attention_score = attention_score / tf.reduce_sum(attention_score)
        context_vector = tf.reduce_sum(attention_score * hidden_states, axis=1)
        return context_vector

    # layer for query, key, value
    # after concat context vectors, apply self attention
    def qkv_model(self):
        model = Sequential()
        model.add(Dense(self.hidden_size_lstm, activation='tanh', kernel_initializer=self.initializer))
        return model

    def get_network(self, inp):
        context_vectors = []
        for i, m in zip(inp, self.sub_models):
            h = m(self.shared_dnn(i))
            context_vectors.append(tf.convert_to_tensor([self.get_attention_score(h)]))
        h = Concatenate(axis=1)(context_vectors)

        # attention model
        qkv = [am(h) for am in self.qkv_models]

        s = tf.nn.softmax(Dot(axes=2)([qkv[0], qkv[1]])) / tf.math.sqrt(tf.cast(self.hidden_size_lstm, dtype=tf.float32))
        h_hat = Dot(axes=1)([s, qkv[2]])

        hidden_h = Dense(self.hidden_size_lstm * 4, activation=self.activation, kernel_initializer=self.initializer)(h + h_hat)
        hidden_h = Dense(self.hidden_size_lstm, activation=self.activation, kernel_initializer=self.initializer)(hidden_h)
        h_p = tf.math.tanh(h + h_hat + hidden_h)

        output = Dense(self.output_dim, activation=self.activation_last, kernel_initializer=self.initializer)(h_p)
        return Model(inp, tf.reshape(output, (-1, self.output_dim * self.num_ticker)))