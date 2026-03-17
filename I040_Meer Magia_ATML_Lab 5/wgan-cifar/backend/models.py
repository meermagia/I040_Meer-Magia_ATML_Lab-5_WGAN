from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_generator(latent_dim: int = 100) -> keras.Model:
    z = keras.Input(shape=(latent_dim,), name="z")
    x = layers.Dense(4 * 4 * 256, use_bias=False)(z)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Reshape((4, 4, 256))(x)

    x = layers.Conv2DTranspose(128, 4, strides=2, padding="same", use_bias=False)(x)  # 8x8
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, 4, strides=2, padding="same", use_bias=False)(x)  # 16x16
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(32, 4, strides=2, padding="same", use_bias=False)(x)  # 32x32
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    img = layers.Conv2DTranspose(3, 3, strides=1, padding="same", activation="tanh")(x)
    return keras.Model(z, img, name="generator")


def build_critic() -> keras.Model:
    img = keras.Input(shape=(32, 32, 3), name="image")
    x = layers.Conv2D(64, 4, strides=2, padding="same")(img)  # 16x16
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(128, 4, strides=2, padding="same")(x)  # 8x8
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(256, 4, strides=2, padding="same")(x)  # 4x4
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Flatten()(x)
    score = layers.Dense(1)(x)  # real-valued critic score (no sigmoid)
    return keras.Model(img, score, name="critic")

