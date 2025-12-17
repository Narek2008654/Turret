import tensorflow as tf
from tensorflow.keras import layers, models
from configs import IMG_H, IMG_W, CHANNELS, N_FRAMES, N_ACTIONS


def build_q_network():
    inputs = layers.Input(shape=(N_FRAMES, IMG_H, IMG_W, CHANNELS))

    # Larger ConvLSTM encoder for stronger spatio-temporal modeling
    x = layers.ConvLSTM2D(96, (3, 3), padding='same', return_sequences=True, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(192, (3, 3), padding='same', return_sequences=False, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Spatial refinement
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Dueling DQN head
    advantage = layers.Dense(256, activation='relu')(x)
    advantage = layers.Dense(N_ACTIONS)(advantage)

    value = layers.Dense(256, activation='relu')(x)
    value = layers.Dense(1)(value)

    q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))

    model = models.Model(inputs, q_values)
    return model