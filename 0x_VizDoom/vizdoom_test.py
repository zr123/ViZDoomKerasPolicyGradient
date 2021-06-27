# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
import concurrent.futures
import time


SCREEN_BUFFER_SHAPE = (3, 240, 320)
ACTION_SPACE = 3
PROCESSED_SCREEN_HEIGHT, PROCESSED_SCREEN_WIDTH = (120, 160)
FRAMES_PER_ACTION = 6
ACTIONS = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
# Number of parallel executors (actually fully separate processed, not threads)
THREAD_COUNT = 8
# number of training-simulations per epoch
BATCH_SIZE = 8


def create_game(display=False):
    game = DoomGame()
    game.load_config("../scenarios/health_gathering.cfg")
    game.set_doom_scenario_path("../scenarios/health_gathering.wad")
    # default rewards are way too high
    game.set_living_reward(0.1)
    game.set_death_penalty(10.0)
    # game.set_render_hud(True)
    game.set_window_visible(display)
    game.init()

    return game


# def custom_loss_function(y_true, y_pred):
def custom_loss_function(reward, action_prob):
    loss = K.log(action_prob) * reward
    loss = K.sum(loss)
    loss = loss / BATCH_SIZE
    return - loss


def play_randomly(game, actions):
    game.set_render_hud(True)
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()
        img = state.screen_buffer
        misc = state.game_variables
        reward = game.make_action(random.choice(actions), FRAMES_PER_ACTION)
    game.get_total_reward()


def preprocess_screen(screen, width=PROCESSED_SCREEN_WIDTH, height=PROCESSED_SCREEN_HEIGHT):
    # vizdoom used channel-first, cv2 uses channel-last
    screen = screen.transpose((1, 2, 0))
    # grayscale
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    # resize
    screen = cv2.resize(screen, (width, height))
    # normalize
    screen = screen / 255.0
    return screen


def create_model(input_width=PROCESSED_SCREEN_WIDTH, input_height=PROCESSED_SCREEN_HEIGHT, action_space=ACTION_SPACE):
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
    model.compile(optimizer='adam', loss=custom_loss_function)
    return model



def take_probabilistic_action(model, formatted_state):
    probabilities = model.predict(formatted_state)
    action = np.random.choice(ACTION_SPACE, p=probabilities[0])
    return action


def play_and_display(game, model):
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()
        screen = preprocess_screen(game.get_state().screen_buffer)
        health = game.get_state().game_variables
        action = take_probabilistic_action(model, [np.array([screen]), np.array([health])])
        reward = game.make_action(ACTIONS[action], FRAMES_PER_ACTION)
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


def format_rewards(action_history, reward_history, action_space=3):
    formatted_ah = utils.to_categorical(action_history, num_classes=action_space)
    formatted_rw = np.full((action_space, reward_history.shape[0]), reward_history).T
    return formatted_ah * formatted_rw


def train(game, model):
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
        reward = game.make_action(ACTIONS[action], FRAMES_PER_ACTION)
        reward_history.append(reward)

    # update weights
    loss = model.train_on_batch(
        x=[np.array(screen_history), np.array(health_history)],
        y=format_rewards(action_history, compute_discounted_reward(reward_history)))

    return loss, game.get_total_reward()


# simple struct-class to keep track of a simulation's history
class History:
    def __init__(self):
        self.screen = []
        self.health = []
        self.action = []
        self.reward = []

    def append(self, screen, health, action, reward):
        self.screen.append(screen)
        self.health.append(health)
        self.action.append(action)
        self.reward.append(reward)


def create_batch(model_weights):
    game = create_game()
    model = create_model()
    model.set_weights(model_weights)
    history = History()

    while not game.is_episode_finished():
        state = game.get_state()
        screen = preprocess_screen(game.get_state().screen_buffer)
        health = game.get_state().game_variables
        action = take_probabilistic_action(model, [np.array([screen]), np.array([health])])
        reward = game.make_action(ACTIONS[action], FRAMES_PER_ACTION)
        history.append(screen, health, action, reward)

    game.close()
    return history


def create_multibatch(model, pool_size=THREAD_COUNT, batch_size=BATCH_SIZE):
    args = [model.get_weights() for i in range(batch_size)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=THREAD_COUNT) as executor:
        results = executor.map(create_batch, args, timeout=60)
    return results


def update_weights(model, histories):
    #
    histories = [h for h in filter(lambda h: isinstance(h, History), histories)]
    x1 = np.vstack([
        histories[i].screen for i in range(len(histories))
    ])
    x2 = np.vstack([
        histories[i].health for i in range(len(histories))
    ])
    # formatted action & reward data
    y = np.vstack([
        format_rewards(histories[i].action, compute_discounted_reward(histories[i].reward)) for i in range(len(histories))
    ])

    loss = model.train_on_batch([x1, x2], y)
    average_reward = np.mean([np.sum(histories[i].reward) for i in range(len(histories))])
    return loss, average_reward


def plot_training(reward_history):
    plt.plot(reward_history, color="blue", label="Batch")
    plt.plot(pd.DataFrame(reward_history).rolling(window=10).mean(), color="red", label="Moving average (10 batches)")
    plt.xlabel("Epochs")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    #model = create_model()
    #total_reward_history = []
    model = keras.models.load_model('model', compile=False)
    model.compile(optimizer='adam', loss=custom_loss_function)
    total_reward_history = pd.read_csv("reward_history.csv").values[:, 0].tolist()

    start_time = time.time()

    for i in range(820):
        batch = create_multibatch(model)
        loss, total_reward = update_weights(model, batch)
        total_reward_history.append(total_reward)
        print("Iteration: ", i, " --- Loss: ", loss, "Reward: ", total_reward)
        if i%20 == 0:
            print("Elapsed time:", time.time() - start_time)
            plot_training(total_reward_history)
            model.save('model')
            pd.DataFrame(total_reward_history).to_csv("reward_history.csv", index=False)

    print("Total training time: ", time.time() - start_time)
