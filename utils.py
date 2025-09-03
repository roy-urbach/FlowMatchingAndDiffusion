from tqdm import tqdm as counter
import tensorflow as tf
import numpy as np


def runs_on_batch(f):
    def call(*args, bs=None, **kwargs):
        if bs is None:
            return f(*args, **kwargs)
        else:
            n = len(args[0])
            batchit = lambda x, i: x[i:i + bs] if (isinstance(x, np.ndarray) and len(x) == n) else x
            return np.concatenate(
                [f(*[batchit(arg, i) for arg in args], **{k: batchit(v, i) for k, v in kwargs.items()}) for i in
                 range(0, n, bs)], axis=0)

    return call


def expand_t(t, shape):
    while len(t.shape) < len(shape):
        t = tf.expand_dims(t, axis=-1)
    return t


def one_hot(labels, num_labels, add_zeros=False):
    return tf.one_hot(labels, num_labels + add_zeros).numpy()
