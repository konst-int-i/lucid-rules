"""
Implementation of integrated gradients for tabular data based on
https://github.com/hcgitrepo/igtab with a few adjustments
"""

import numpy as np
import tensorflow as tf
from tqdm import tqdm


class IGExplainer():
    def __init__(self, model, base='zero', y_idx=1, m_steps=50, batch_size=100, type='sigmoid'):
        self.base = base
        self.y_idx = y_idx
        self.m_steps = m_steps
        self.batch_size = batch_size
        self.type = type
        self.result = None
        self.model = model

    def ig_values(self, X):
        # generate integrated gradients
        igs = []
        for i in tqdm(range(X.shape[0])):
            res = self.integrated_gradients(X[i, :])
            igs.append(res.numpy())
        self.result = np.array(igs)
        return self.result

    def interpolate_data(self, baseline, data, alphas):
        alphas_x = alphas[:, tf.newaxis]
        baseline_x = tf.expand_dims(baseline, axis=0)
        input_x = tf.expand_dims(tf.cast(data, tf.float32), axis=0)
        delta = input_x - baseline_x
        outputs = baseline_x + alphas_x * delta
        return outputs

    def get_gradients(self, data):
        with tf.GradientTape() as tape:
            tape.watch(data)
            output = self.model(data)
            if self.type == 'sigmoid':
                y = tf.nn.sigmoid(output)
            elif self.type == 'softmax':
                y = tf.nn.softmax(output, axis=-1)[:, self.y_idx]
            elif self.type == 'autoencoder_mse':
                x_diff = tf.math.subtract(data, output)
                y = tf.math.reduce_mean(tf.math.square(x_diff), axis=1)
            else:
                y = tf.keras.activations.linear(output)
        return tape.gradient(y, data)

    def integral_approximation(self, gradients):
        # riemann_trapezoidal
        grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
        integrated_gradients = tf.math.reduce_mean(grads, axis=0)
        return integrated_gradients

    @tf.function
    def integrated_gradients(self, data):
        with tf.device('/CPU:0'):
            data = tf.cast(data, tf.float32)
            if self.base == 'zero':
                baseline = tf.zeros(shape=(data.shape[-1]))
            elif self.base == 'uniform':
                baseline = tf.random.uniform(shape=(data.shape[-1],), minval=0.0, maxval=1.0)
            else:
                baseline = tf.zeros(shape=(data.shape[-1]))

            alphas = tf.linspace(start=0.0, stop=1.0, num=self.m_steps + 1)
            gradient_batches = tf.TensorArray(tf.float32, size=self.m_steps + 1)

            for alpha in tf.range(0, len(alphas), self.batch_size):
                start = alpha
                end = tf.minimum(start + self.batch_size, len(alphas))
                alpha_batch = alphas[start:end]
                outputs_batch = self.interpolate_data(baseline=baseline, data=data, alphas=alpha_batch)
                gradient_batch = self.get_gradients(data=outputs_batch)
                gradient_batches = gradient_batches.scatter(tf.range(start, end), gradient_batch)

            total_gradients = gradient_batches.stack()
            avg_gradients = self.integral_approximation(gradients=total_gradients)
            integrated_gradients = (data - baseline) * avg_gradients

        return integrated_gradients