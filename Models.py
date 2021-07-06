import gym
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow import keras
import numpy as np
import VizdoomWrapper
import tensorflow.keras.utils as utils


def create_model(name):
    if name == "VizDoom":
        return create_vizdoom_model()
    if name == "CartPole-v0":
        return create_cartpole_model()
    raise Exception("Unresolved model name in create_model.")


def create_baseline_model(name):
    if name == "VizDoom":
        return create_vizdoom_baseline_model()
    if name == "CartPole-v0":
        return create_cartpole_baseline_model()
    raise Exception("Unresolved model name in create_model.")


def policy_gradient_loss(reward, action_prob):
    # add a very small number to the action probability
    # so practically there won't be a log(0) error
    loss = K.log(action_prob + 1e-7) * reward
    loss = K.sum(loss)
    return - loss


def create_cartpole_model():
    model = Sequential()
    model.add(Input(shape=(4,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss=policy_gradient_loss)
    return model


def create_cartpole_baseline_model():
    model = Sequential()
    model.add(Input(shape=(4,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss="MSE")
    return model


def create_vizdoom_model(input_width=160, input_height=120, action_space=3):
    # screen-input and conv_layers
    screen_input = Input((input_height, input_width, 1))
    conv_layers = Conv2D(32, kernel_size=(3, 3), activation="relu")(screen_input)
    conv_layers = MaxPooling2D(pool_size=(2, 2))(conv_layers)
    conv_layers = Conv2D(32, kernel_size=(3, 3), activation="relu")(conv_layers)
    conv_layers = MaxPooling2D(pool_size=(2, 2))(conv_layers)
    conv_layers = Conv2D(32, kernel_size=(3, 3), activation="relu")(conv_layers)
    conv_layers = MaxPooling2D(pool_size=(2, 2))(conv_layers)
    conv_layers = Conv2D(32, kernel_size=(3, 3), activation="relu")(conv_layers)
    conv_layers = MaxPooling2D(pool_size=(2, 2))(conv_layers)
    conv_layers = Flatten()(conv_layers)

    # concat and add some dense layers for good measure
    dense_layers = Dense(16, activation='relu')(conv_layers)
    dense_layers = Dense(16, activation='relu')(dense_layers)
    dense_layers = Dense(16, activation='relu')(dense_layers)

    # finalize
    output = Dense(action_space, activation='softmax')(dense_layers)
    model = keras.Model(inputs=screen_input, outputs=output)
    model.compile(optimizer='adam', loss=policy_gradient_loss)
    return model


def create_vizdoom_baseline_model(input_width=160, input_height=120):
    # screen-input and conv_layers
    screen_input = Input((input_height, input_width, 1))
    conv_layers = Conv2D(32, kernel_size=(3, 3), activation="relu")(screen_input)
    conv_layers = MaxPooling2D(pool_size=(2, 2))(conv_layers)
    conv_layers = Conv2D(32, kernel_size=(3, 3), activation="relu")(conv_layers)
    conv_layers = MaxPooling2D(pool_size=(2, 2))(conv_layers)
    conv_layers = Conv2D(32, kernel_size=(3, 3), activation="relu")(conv_layers)
    conv_layers = MaxPooling2D(pool_size=(2, 2))(conv_layers)
    conv_layers = Conv2D(32, kernel_size=(3, 3), activation="relu")(conv_layers)
    conv_layers = MaxPooling2D(pool_size=(2, 2))(conv_layers)
    conv_layers = Flatten()(conv_layers)

    # concat and add some dense layers for good measure
    dense_layers = Dense(16, activation='relu')(conv_layers)
    dense_layers = Dense(16, activation='relu')(dense_layers)
    dense_layers = Dense(16, activation='relu')(dense_layers)

    # finalize
    output = Dense(1)(dense_layers)
    model = keras.Model(inputs=screen_input, outputs=output)
    model.compile(optimizer='adam', loss="MSE")
    return model


def create_vizdoom_model_old(input_width=160, input_height=120, action_space=3):
    # screen-input and conv_layers
    screen_input = Input((input_height, input_width, 1))
    conv_layers = Conv2D(32, kernel_size=(3, 3), activation="relu")(screen_input)
    conv_layers = MaxPooling2D(pool_size=(2, 2))(conv_layers)
    conv_layers = Conv2D(32, kernel_size=(3, 3), activation="relu")(conv_layers)
    conv_layers = MaxPooling2D(pool_size=(2, 2))(conv_layers)
    conv_layers = Conv2D(32, kernel_size=(3, 3), activation="relu")(conv_layers)
    conv_layers = MaxPooling2D(pool_size=(2, 2))(conv_layers)
    conv_layers = Conv2D(32, kernel_size=(3, 3), activation="relu")(conv_layers)
    conv_layers = MaxPooling2D(pool_size=(2, 2))(conv_layers)
    conv_layers = Flatten()(conv_layers)

    # health input
    health_input = Input((1,))

    # concat and add some dense layers for good measure
    dense_layers = layers.concatenate([conv_layers, health_input])
    dense_layers = Dense(16, activation='relu')(dense_layers)
    dense_layers = Dense(16, activation='relu')(dense_layers)
    dense_layers = Dense(16, activation='relu')(dense_layers)

    # finalize
    output = Dense(action_space, activation='softmax')(dense_layers)
    model = keras.Model(inputs=[screen_input, health_input], outputs=output)
    model.compile(optimizer='adam', loss=policy_gradient_loss)
    return model

