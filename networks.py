import os
import threading
import numpy as np
print(f'Keras Backend : {os.environ["KERAS_BACKEND"]}')

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, LSTM, LayerNormalization, Concatenate, MultiHeadAttention
from tensorflow.keras.initializers import glorot_uniform, glorot_normal
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import mean_squared_error as mse
from keras import Input
import tensorflow as tf

optimizer = SGD

class Network:
    lock = threading.Lock()

    def __init__(self, input_dim=0, output_dim=0, num_ticker=100, num_index=5, num_steps=5, trainable=True,
                 batch_size=1, activation=None, value_flag='True'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_ticker = num_ticker
        self.num_index = num_index
        self.num_steps = num_steps
        self.trainable = trainable
        self.model = None
        self.initializer = glorot_normal()
        self.activation = 'relu'
        self.activation_last = activation
        self.value_flag = value_flag
        if self.value_flag:
            self.last_idx = -3
        else:
            self.last_idx = -2
        self.batch_size = batch_size

    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True)

    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)

class DNN(Network):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sub_models = [self.mini_dnn() for _ in range(self.num_ticker + 1)]
        # 마지막 input은 policy : index, value: action
        inp = [Input(shape=(None, self.input_dim)) for _ in range(self.num_ticker)]
        inp.append(Input(shape=(None, self.num_index)))
        inp.append(Input(shape=(self.num_ticker,)))
        # inp.append(Input(shape=(self.num_ticker,)))
        if self.value_flag:
            inp.append(Input(shape=(self.num_ticker,)))
        self.model = self.get_network(inp, sub_models)

    def residual_layer(self, inp, hidden_size):
        output_r = Dense(hidden_size, activation=self.activation, kernel_initializer=self.initializer)(inp)
        output_r = Dropout(0.1, trainable=self.trainable)(output_r)
        output = Dense(hidden_size, activation=self.activation, kernel_initializer=self.initializer)(output_r)
        output = Dropout(0.1, trainable=self.trainable)(output)
        output = Dense(hidden_size, activation=self.activation, kernel_initializer=self.initializer)(output)
        output = Dropout(0.1, trainable=self.trainable)(output)
        return output + output_r

    def mini_dnn(self):
        model = Sequential()
        model.add(Dense(256, activation=self.activation, kernel_initializer=self.initializer))
        model.add(Dropout(0.1, trainable=self.trainable))
        # model.add(Dense(128, activation=self.activation, kernel_initializer=self.initializer))
        # model.add(Dropout(0.1, trainable=self.trainable))
        model.add(Dense(32, activation=self.activation, kernel_initializer=self.initializer))
        # model.add(Dropout(0.1, trainable=self.trainable))
        return model

    def get_network(self, inp, sub_models):
        output = [tf.squeeze(m(i), axis=-2) for i, m in zip(inp[:self.last_idx], sub_models[:self.last_idx])]
        output.append(tf.squeeze(sub_models[self.last_idx](inp[self.last_idx]), axis=-2))
        output.append(inp[self.last_idx + 1])
        # output.append(inp[self.last_idx + 2])
        if self.value_flag:
            output.append(inp[-1])
        output = Concatenate()(output)
        output = Dense(2048, activation=self.activation, kernel_initializer=self.initializer)(output)
        output = Dropout(0.1, trainable=self.trainable)(output)
        output = self.residual_layer(output, 1024)
        output = self.residual_layer(output, 512)
        # output = Dense(512, activation=self.activation, kernel_initializer=self.initializer)(output)
        # output = Dropout(0.1, trainable=self.trainable)(output)
        # output = Dense(256, activation=self.activation, kernel_initializer=self.initializer)(output)
        output = Dense(256, activation=self.activation, kernel_initializer=self.initializer)(output)
        return Model(inp, output)


# refer to paper DTML(jaemin yoo)
# Accurate Multivariate Stock Movement Prediction via Data-Axis Transformer with Multi-Level Contexts
class AttentionLSTM(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_steps = num_steps
        self.hidden_size_lstm = 256

        # 마지막 input은 index
        inp = [Input((self.num_steps, self.input_dim)) for _ in range(self.num_ticker)]
        inp.append(Input((self.num_steps, self.num_index)))
        inp.append(Input((self.num_ticker,)))
        # inp.append(Input((self.num_ticker,)))
        if self.value_flag:
            inp.append(Input(shape=(self.num_ticker,)))
        sub_models = [self.mini_model() for _ in range(self.num_ticker + 1)]
        qkv_models = [self.qkv_model() for _ in range(3)]
        mha = MultiHeadAttention(4, 4)
        self.model = self.get_network(inp, sub_models, qkv_models, mha)

    # after expand input, calculate hidden state of input sequences
    def mini_model(self):
        model = Sequential()
        model.add(LSTM(self.hidden_size_lstm, dropout=0.1, return_sequences=True, stateful=False, kernel_initializer=self.initializer))
        model.add(LayerNormalization(trainable=self.trainable))
        return model

    # calculate attention score from hidden states of each stocks
    def get_attention_score(self, hidden_states):
        last_hidden_state = hidden_states[-1]
        attention_score = tf.exp(last_hidden_state * hidden_states)
        total_attention_score = tf.reduce_sum(attention_score)
        total_attention_score = tf.math.add(total_attention_score, 1e-5)
        attention_score = tf.math.divide(attention_score, total_attention_score)
        context_vector = tf.reduce_sum(attention_score * hidden_states, axis=1)
        return context_vector

    # layer for query, key, value
    # after concat context vectors, apply self attention
    def qkv_model(self):
        model = Sequential()
        model.add(Dense(self.hidden_size_lstm, activation='tanh', kernel_initializer=self.initializer))
        return model

    def get_network(self, inp, sub_models, qkv_models, mha):
        inp_action = None
        inp_ksportf = inp[self.last_idx+1]
        # inp_portf = inp[self.last_idx+2]
        inp_data = inp[:self.last_idx+1]
        if self.value_flag:
            inp_action = inp[-1]
        context_vectors = [tf.convert_to_tensor([self.get_attention_score(m(i))]) for i, m in zip(inp_data, sub_models)]
        context_vectors = [context_vectors[-1] + cv for cv in context_vectors[:-1]]
        context_vectors = [tf.transpose(cv, (1, 0, 2)) for cv in context_vectors]
        h = Concatenate(axis=1)(context_vectors)

        # attention model
        qkv = [am(h) for am in qkv_models]
        h_hat = mha(*qkv)
        h_p = Dense(32, activation=self.activation, kernel_initializer=self.initializer)(h + h_hat)

        output = tf.reshape(h_p, (-1, self.num_ticker * 32))
        if self.value_flag:
            output = [output, inp_ksportf, inp_action]
        else:
            output = [output, inp_ksportf]
        output = Concatenate(axis=-1)(output)
        output = Dense(2048, activation=self.activation, kernel_initializer=self.initializer)(output)
        output = Dropout(0.1, trainable=self.trainable)(output)
        output = Dense(1024, activation=self.activation, kernel_initializer=self.initializer)(output)
        output = Dropout(0.1, trainable=self.trainable)(output)
        output = Dense(512, activation=self.activation, kernel_initializer=self.initializer)(output)

        return Model(inp, output)


LOG_STD_MIN = -20
LOG_STD_MAX = 1
EPS = 1e-8
class pi_network:
    def __init__(self, net='dnn', lr=0.001, alpha=0.2, *args, **kargs):
        if net == 'dnn':
            self.network = DNN(*args, **kargs)
        else:
            self.network = AttentionLSTM(*args, **kargs)
        self.alpha = alpha
        self.discount_factor = 0.9
        self.optimizer = optimizer(lr, clipvalue=0.5)
        self.mu_layer = Dense(self.network.output_dim, activation=self.network.activation_last, kernel_initializer=self.network.initializer)
        self.log_std_layer = Dense(self.network.output_dim, activation=self.network.activation_last, kernel_initializer=self.network.initializer)

    def predict(self, s, deterministic=False, learn=True):
        output = self.network.model(s)
        mu = tf.reshape(self.mu_layer(output), (-1, self.network.num_ticker))
        log_std = tf.reshape(self.log_std_layer(output), (-1, self.network.num_ticker))
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        if not learn:
            mu = tf.squeeze(mu)
            log_std = tf.squeeze(log_std)
        std = tf.math.exp(log_std)
        pi = mu + tf.random.normal(tf.shape(mu)) * 0.01 * std
        # pi_distribution = tfp.distributions.Normal(mu, std)
        log_prob_pi = self.gaussian_likelihood(pi, mu, log_std)

        # apply squashing
        log_prob_pi -= 2 * (np.log(2) - pi - tf.nn.softplus(-2*pi))
        if deterministic:
            pi_action = mu
        else:
            pi_action = tf.clip_by_value(pi, -1.0, 1.0)

        return pi_action, log_prob_pi

    def gaussian_likelihood(self, x, mu, log_std):
        pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        return pre_sum

    def learn(self, s, value_network):
        with tf.GradientTape() as tape:
            pi, logp_pi = self.predict(s)
            # pi 업데이트라 q_net gradient off
            q1_pi, q2_pi = tf.stop_gradient(value_network.predict(s, pi))
            q_pi = tf.math.minimum(q1_pi, q2_pi)
            loss_pi = tf.reduce_mean(self.alpha * logp_pi - q_pi, axis=1)
        gradients = tape.gradient(loss_pi, self.network.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.network.model.trainable_variables))

        return tf.reduce_mean(loss_pi)

    def save_model(self, model_path):
        self.network.save_model(model_path)

    def load_model(self, model_path):
        self.network.load_model(model_path)


class q_network:
    def __init__(self, net='dnn', lr=0.001, *args, **kargs):
        if net == 'dnn':
            self.network1 = DNN(*args, **kargs)
            self.network2 = DNN(*args, **kargs)
        else:
            self.network1 = AttentionLSTM(*args, **kargs)
            self.network2 = AttentionLSTM(*args, **kargs)
        self.layer1 = Dense(self.network1.output_dim, activation=self.network1.activation_last,
                            kernel_initializer=self.network1.initializer)
        self.layer2 = Dense(self.network2.output_dim, activation=self.network2.activation_last,
                            kernel_initializer=self.network2.initializer)
        self.loss = mse
        self.optimizer1 = optimizer(lr, clipvalue=0.5)
        self.optimizer2 = optimizer(lr, clipvalue=0.5)

    def predict(self, s, a):
        s.append(a)
        output1 = self.network1.model(s)
        output1 = self.layer1(output1)
        output1 = tf.reshape(output1, shape=(self.network1.batch_size, self.network1.num_ticker))
        output2 = self.network2.model(s)
        output2 = self.layer2(output2)
        output2 = tf.reshape(output2, shape=(self.network1.batch_size, self.network1.num_ticker))
        return output1, output2

    def learn(self, s, a, backup):
        with tf.GradientTape() as tape_q, tf.GradientTape() as tape_pi:
            q1, q2 = self.predict(s, a)
            loss_q1 = tf.math.sqrt(self.loss(backup, q1))
            loss_q2 = tf.math.sqrt(self.loss(backup, q2))

        gradients1 = tape_q.gradient(loss_q1, self.network1.model.trainable_variables)
        self.optimizer1.apply_gradients(zip(gradients1, self.network1.model.trainable_variables))
        gradients2 = tape_pi.gradient(loss_q2, self.network2.model.trainable_variables)
        self.optimizer2.apply_gradients(zip(gradients2, self.network2.model.trainable_variables))

        return tf.reduce_mean(loss_q1 + loss_q2)

    def save_model(self, model_path):
        self.network1.save_model(model_path[0])
        self.network2.save_model(model_path[1])

    def load_model(self, model_path):
        self.network1.load_model(model_path[0])
        self.network2.load_model(model_path[1])




