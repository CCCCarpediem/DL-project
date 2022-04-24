import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from sklearn.metrics import *

class CAAE_SPARSE:
    def build_DCCA(self, X, Y):
        self.hidden_size = [512, 256]
        self.latent_size = 60

        with tf.variable_scope('DCCA'):
            W_1D = tf.get_variable(name='W_1D', shape=[self.feature_size, self.latent_size],
                                   initializer=tf.truncated_normal_initializer(mean=0.0, stddev=self.std,
                                                                               dtype=tf.float64))

            b_1D = tf.get_variable(name='b_1D', shape=[self.latent_size],
                                   initializer=tf.truncated_normal_initializer(mean=0.0, stddev=self.std,
                                                                               dtype=tf.float64))

            X_DCCA_latent = tf.nn.leaky_relu(tf.matmul(X, W_1D) + b_1D)
            X_DCCA_latent = tf.nn.dropout(X_DCCA_latent, keep_prob=self.keep_prob)

            self.hidden_size_1 = [800, 500, 200]
            W_1D_Y = tf.get_variable(name='W_1D_Y', shape=[self.label_size, self.hidden_size_1[1]],
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=self.std,
                                                                                 dtype=tf.float32))
            W_2D_Y = tf.get_variable(name='W_2D_Y', shape=[self.hidden_size_1[1], self.latent_size],
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=self.std,
                                                                                 dtype=tf.float32))

            b_1D_Y = tf.get_variable(name='b_1D_Y', shape=[self.hidden_size_1[1]],
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=self.std,
                                                                                 dtype=tf.float32))
            b_2D_Y = tf.get_variable(name='b_2D_Y', shape=[self.latent_size],
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=self.std,
                                                                                 dtype=tf.float32))
            Y1 = tf.matmul(Y, W_1D_Y) + b_1D_Y
            Y1 = tf.nn.dropout(Y1, keep_prob=self.keep_prob)
            Y2 = tf.matmul(Y1, W_2D_Y) + b_2D_Y
            Y2 = tf.nn.dropout(Y2, keep_prob=self.keep_prob)
            Y_DCCA_latent = tf.nn.leaky_relu(Y2)

            return X_DCCA_latent, Y_DCCA_latent, W_1D, b_1D, W_1D_Y, W_2D_Y

    def compute_correlation(self, X, Y):
        ''''r1,r2 regularization term'''
        r1 = 1e-4
        r2 = 1e-4
        H1 = tf.transpose(X)
        H2 = tf.transpose(Y)

        m = self.batch_size
        m = tf.cast(m, dtype=tf.float32)
        H1bar = H1 - 1.0 / (m - 1) * tf.matmul(H1, tf.ones([m, m], dtype=tf.float32))
        H2bar = H2 - 1.0 / (m - 1) * tf.matmul(H2, tf.ones([m, m], dtype=tf.float32))

        Sigmahat12 = 1.0 / (m - 1) * tf.matmul(H1bar, tf.transpose(H2bar))
        Sigmahat11 = 1.0 / (m - 1) * tf.matmul(H1bar, tf.transpose(H1bar)) + r1 * tf.eye(self.latent_size)
        Sigmahat22 = 1.0 / (m - 1) * tf.matmul(H2bar, tf.transpose(H2bar)) + r2 * tf.eye(self.latent_size)


        [D1, V1] = tf.self_adjoint_eig(Sigmahat11)
        [D2, V2] = tf.self_adjoint_eig(Sigmahat22)

        SigmaHat11RootInv = tf.matmul(tf.matmul(V1, tf.diag(D1) ** 0.5), tf.transpose(V1))
        SigmaHat22RootInv = tf.matmul(tf.matmul(V2, tf.diag(D2) ** 0.5), tf.transpose(V2))

        T = tf.matmul(tf.matmul(SigmaHat11RootInv, Sigmahat12), SigmaHat22RootInv)

        corr = tf.trace(tf.matmul(tf.transpose(T), T)) ** 0.5
        return -corr, Sigmahat11, Sigmahat22

    def auto_encoder(self, Y_latent, Y_initial):
        with tf.variable_scope('Autoencoder'):
            W_1A_Y = tf.get_variable(name='W_1A_Y', shape=[self.latent_size, self.hidden_size[1]],
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=self.std,
                                                                                 dtype=tf.float32))
            W_2A_Y = tf.get_variable(name='W_2A_Y', shape=[self.hidden_size[1], self.label_size],
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=self.std,
                                                                                 dtype=tf.float32))
            W_3A_Y = tf.get_variable(name='weight', shape=[self.label_size, self.label_size],
                                     initializer=tf.ones_initializer(dtype=tf.float32))
            b_1A_Y = tf.get_variable(name='b_1A_Y', shape=[self.hidden_size[1]],
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=self.std,
                                                                                 dtype=tf.float32))
            b_2A_Y = tf.get_variable(name='b_2A_Y', shape=[self.label_size],
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=self.std,
                                                                                 dtype=tf.float32))
            Y1_auto = tf.nn.dropout((tf.matmul(Y_latent, W_1A_Y) + b_1A_Y), keep_prob=self.keep_prob)
            Y2_auto = tf.nn.dropout((tf.matmul(Y1_auto, W_2A_Y) + b_2A_Y), keep_prob=self.keep_prob)
            Y3_auto = tf.nn.dropout(tf.matmul(Y2_auto, W_3A_Y), keep_prob=self.keep_prob)
            Y_decode = tf.layers.batch_normalization(Y3_auto)
            auto_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_initial, logits=Y_decode))
            return Y_decode, auto_loss
