import tensorflow as tf
from tensorflow.keras import layers, models
from configs import IMG_H, IMG_W, CHANNELS, N_FRAMES, N_ACTIONS

def build_q_network():
    inputs = layers.Input(shape=(N_FRAMES, IMG_H, IMG_W, CHANNELS))

    x = layers.TimeDistributed(layers.Conv2D(128, 8, strides=4, activation='relu'))(inputs)
    x = layers.TimeDistributed(layers.Conv2D(64, 4, strides=2, activation='relu'))(x)
    x = layers.TimeDistributed(layers.Conv2D(64, 3, strides=1, activation='relu'))(x)
    x = layers.TimeDistributed(layers.Flatten())(x)


    x = layers.LSTM(512)(x)
    x = layers.Dense(256, activation='relu')(x)
    q_values = layers.Dense(N_ACTIONS)(x)


    model = models.Model(inputs, q_values)
    return model