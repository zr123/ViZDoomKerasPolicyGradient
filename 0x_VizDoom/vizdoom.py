import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import layers
import tensorflow.keras.utils as utils
from vizdoom import *
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

SCREEN_BUFFER_SHAPE = (3, 240, 320)
ACTION_SPACE = 3
PROCESSED_SCREEN_HEIGTH = 60
PROCESSED_SCREEN_WIDTH = 80
FRAMES_PER_ACTION = 6


def create_game(display=True):
    game = DoomGame()
    game.load_config("../scenarios/health_gathering.cfg")
    game.set_doom_scenario_path("../scenarios/health_gathering.wad")
    # game.set_render_hud(True)
    game.set_window_visible(display)
    game.init()

    # actions: left, right, forward
    actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    return game, actions


def play_randomly(game, actions):
    game.set_render_hud(True)
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()
        img = state.screen_buffer
        misc = state.game_variables
        reward = game.make_action(random.choice(actions), FRAMES_PER_ACTION)
    game.get_total_reward()


def preprocess_screen(screen, width=PROCESSED_SCREEN_WIDTH, height=PROCESSED_SCREEN_HEIGTH):
    # vizdoom used channel-first, cv2 uses channel-last
    screen = screen.transpose((1, 2, 0))
    # grayscale
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    # resize
    screen = cv2.resize(screen, (width, height))
    # normalize
    screen = screen / 255.0
    return screen


def create_model(input_width=PROCESSED_SCREEN_WIDTH, input_height=PROCESSED_SCREEN_HEIGTH, action_space=ACTION_SPACE):
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
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


def format_state(state):
    screen = game.get_state().screen_buffer
    health = game.get_state().game_variables
    processed_screen = preprocess_screen(screen)
    return [processed_screen, health]


def take_probabilistic_action(model, formated_state):
    probabilities = model.predict(formated_state)
    action = np.random.choice(ACTION_SPACE, p=probabilities[0])
    return action


def play_and_display(game, model):
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()
        screen = preprocess_screen(game.get_state().screen_buffer)
        health = game.get_state().game_variables
        action = take_probabilistic_action(model, [np.array([screen]), np.array([health])])
        reward = game.make_action(actions[action], FRAMES_PER_ACTION)
    return game.get_total_reward()


def compute_discounted_reward(reward_history, discount_rate=0.99):
    discounted_rewards = []
    discounted_sum = 0
    for r in reward_history[::-1]:
        discounted_sum = r + discount_rate * discounted_sum
        discounted_rewards.insert(0, discounted_sum)

    # use simple Baseline
    discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / np.std(discounted_rewards)

    return discounted_rewards


def train(gym_env, model):
    game.new_episode()
    action_history = []
    reward_history = []
    # state history
    screen_history = []
    health_history = []

    while not game.is_episode_finished():
        state = game.get_state()

        screen = preprocess_screen(game.get_state().screen_buffer)
        health = game.get_state().game_variables
        screen_history.append(screen)
        health_history.append(health)

        action = take_probabilistic_action(model, [np.array([screen]), np.array([health])])
        action_history.append(action)
        reward = game.make_action(actions[action], FRAMES_PER_ACTION)
        reward_history.append(reward)

    # update weights
    loss = model.train_on_batch(
        x=[np.array(screen_history), np.array(health_history)],
        y=utils.to_categorical(action_history),
        sample_weight=compute_discounted_reward(reward_history))

    return loss, game.get_total_reward()


if __name__ == '__main__':
