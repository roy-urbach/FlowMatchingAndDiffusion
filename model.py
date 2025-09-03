import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np


def train_generative(prior, t_theta, optimizer, gen, loss, c_theta=None, epochs=10, steps_per_epoch=100, **kwargs):
    from tqdm.notebook import tqdm as counter
    alpha = 0.9
    losses = []
    for ep in range(epochs):
        pbar = counter(range(steps_per_epoch))
        mean_loss = None
        for step in pbar:
            x, c = next(gen)
            with tf.GradientTape() as tape:
                l = loss(prior, t_theta, x, c_theta=c_theta, c=c, **kwargs)
                if np.isnan(l.numpy()): break
                mean_loss = l if mean_loss is None else (alpha * mean_loss + (1 - alpha) * l)
            if not (step % 25):
                pbar.set_description(f"epoch {ep}: loss={mean_loss:.2e}")
            trainable_weights = prior.trainable_weights + t_theta.trainable_weights + (
                c_theta.trainable_weights if c_theta is not None else [])
            grads_model = tape.gradient(l, trainable_weights)
            optimizer.apply_gradients(zip(grads_model, trainable_weights))
        losses.append(mean_loss)
    return losses


def generator(x, y, bs=32, to_one_hot=True, label_dropout=0, classes=None):
    x = x.copy()
    y = y.copy()
    while True:
        inds = np.random.permutation(len(x))
        cur_x = x[inds]
        cur_y = y[inds]
        if label_dropout:
            cur_y = np.where(np.random.rand(len(cur_y)) > label_dropout, cur_y, 0)
        if to_one_hot:
            cur_y = tf.one_hot(cur_y,
                               depth=(len(np.unique(cur_y)) if classes is None else classes) + (label_dropout > 0))
        for i in range(0, len(x), bs):
            yield cur_x[i:i + bs], cur_y[i:i + bs]


def create_mlp_model(input_dim, output_dim, depth, width):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for i in range(depth):
        model.add(layers.Dense(width, activation='relu', name=f'linear{i}'))
    model.add(layers.Dense(output_dim, activation='linear', name='out'))
    return model


def create_mlp_model_t(output_shape, width=32, depth=3):
    model = models.Sequential()
    model.add(layers.Input(shape=(1,)))
    for i in range(depth):
        model.add(layers.Dense(width, name=f'linear{i}', activation='relu'))
    model.add(layers.Dense(width, name=f'linear_out', activation=None))  # (N, width)
    for i in range(len(output_shape)):
        model.add(layers.Lambda(lambda x: tf.expand_dims(x, axis=-2), name=f'expand{i}'))  # (N, 1,...,1, width)
    model.add(layers.Lambda(lambda x: tf.tile(x, [1] + list(output_shape) + [1],
                                              name='tile')))  # # (N, output_shape[0], output_shape[1], width)
    return model


def create_mlp_model_c(num_classes, output_shape, width=32, depth=3):
    model = models.Sequential()
    model.add(layers.Input(shape=(num_classes,)))
    for i in range(depth):
        model.add(layers.Dense(width, name=f'class_linear{i}', activation='relu'))
    model.add(layers.Dense(width, name=f'class_linear_out', activation=None))  # (N, width)
    for i in range(len(output_shape)):
        model.add(layers.Lambda(lambda x: tf.expand_dims(x, axis=-2), name=f'class_expand{i}'))  # (N, 1,...,1, width)
    model.add(layers.Lambda(lambda x: tf.tile(x, [1] + list(output_shape) + [1],
                                              name='tile')))  # # (N, output_shape[0], output_shape[1], width)
    return model


def create_unet_model(input_shape, depth=3, width=32, activation='relu'):
    inputs = layers.Input(input_shape)
    x = inputs

    # Downsampling
    downsampling_layers = []
    for i in range(depth):
        x = layers.Conv2D(width * (2 ** i), 3, padding='same', activation=activation)(x)
        x = layers.Conv2D(width * (2 ** i), 3, padding='same', activation=activation)(x)
        if i < depth - 1:
            downsampling_layers.append(x)
            x = layers.MaxPooling2D((2, 2), (2, 2))(x)

    # Bottleneck
    x = layers.Conv2D(width * (2 ** depth), 3, padding='same', activation=activation)(x)
    x = layers.Conv2D(width * (2 ** depth), 3, padding='same', activation=activation)(x)

    # Upsampling
    for i in range(depth - 2, -1, -1):
        x = layers.Conv2DTranspose(width * (2 ** i), 2, strides=(2, 2), padding='same')(x)
        x = layers.concatenate([x, downsampling_layers[i]])
        x = layers.Conv2D(width * (2 ** i), 3, padding='same', activation=activation)(x)
        x = layers.Conv2D(width * (2 ** i), 3, padding='same', activation=activation)(x)

    # Output layer
    outputs = layers.Conv2D(1, 1, activation=None)(x)

    model = models.Model(inputs, outputs)
    return model
