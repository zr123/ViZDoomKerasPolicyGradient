import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
import tensorflow.keras.utils as utils
import numpy as np
import pandas as pd
import math
import gym
import matplotlib.pyplot as plt
import ipywidgets as widgets
from multiprocessing import Process
from threading import Thread
from multiprocessing import Process, Pool
import time


# def custom_loss_function(y_true, y_pred):
def custom_loss_function(reward, action_prob):
    loss = K.log(action_prob) * reward
    # loss = action_prob * reward
    loss = K.sum(loss)
    # negate to turn the maximization into a minimization
    return -loss


def create_model():
    model = Sequential()
    model.add(Input(shape = (4,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss=custom_loss_function)
    return model


def take_probabilistic_action(model, state):
    probabilities = model.predict(state.reshape(1, -1))
    action = np.random.choice(2, p=probabilities[0])
    return action, probabilities


def play_and_display(gym_env, model):
    state = gym_env.reset()
    done = False
    while not done:
        gym_env.render()
        action, _ = take_probabilistic_action(model, state)
        state, reward, done, info = gym_env.step(action)
    gym_env.close()


def compute_discounted_reward(reward_history, discount_rate=0.99):
    discounted_rewards = []
    discounted_sum = 0
    for r in reward_history[::-1]:
        discounted_sum = r + discount_rate * discounted_sum
        discounted_rewards.insert(0, discounted_sum)

    # use simple Baseline
    discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / np.std(discounted_rewards)

    return discounted_rewards


#def format_rewards(rewards, action_space=2):
#    return np.full((action_space, rewards.shape[0]), rewards).T
def format_rewards(action_history, reward_history, action_space=2):
    formated_ah = utils.to_categorical(action_history, num_classes=action_space)
    formated_rw = np.full((action_space, reward_history.shape[0]), reward_history).T
    return formated_ah * formated_rw


def train(gym_env, model):
    state = gym_env.reset()
    state_history = []
    action_history = []
    reward_history = []

    done = False
    # simulation
    while not done:
        action, _ = take_probabilistic_action(model, state)
        action_history.append(action)
        state_history.append(state)
        state, reward, done, info = gym_env.step(action)
        reward_history.append(reward)
    gym_env.close()

    # update weights
    loss = model.train_on_batch(
        x=np.array(state_history),
        y=format_rewards(action_history, compute_discounted_reward(reward_history)))

    return (loss, np.sum(reward_history))


# simple struct-class to keep track of a simulation's history
class History:
    def __init__(self):
        self.state = []
        self.action = []
        self.reward = []

    def append(self, state, action, reward):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)


def create_batch(model_weights):
    # def create_batch(model_weights, history_ref=[{}], i=0):
    gym_env = gym.make('CartPole-v0')
    model = create_model()
    model.set_weights(model_weights)
    state = gym_env.reset()
    history = History()

    done = False
    # simulation
    while not done:
        action, _ = take_probabilistic_action(model, state)
        new_state, reward, done, info = gym_env.step(action)
        history.append(state, action, reward)
        state = new_state

    return history


def create_multibatch(model, n):
    with Pool(n) as p:
        results = p.map(create_batch, [model.get_weights() for i in range(n)])

    return results


def weasel_histories(model, histories):
    # states
    x = np.vstack(
        [histories[i].state for i in range(len(histories))]
    )
    # formated action & reward data
    y = np.vstack(
        [format_rewards(histories[i].action, compute_discounted_reward(histories[i].reward)) for i in
         range(len(histories))]
    )

    loss = model.train_on_batch(x, y)
    average_reward = np.mean([np.sum(histories[i].reward) for i in range(len(histories))])
    return loss, average_reward


if __name__ == '__main__':
    # model = keras.models.load_model('model')
    model = create_model()
    THREADS = 4
    total_reward_history = []

    results = create_multibatch(model, THREADS)
    weasel_histories(model, results)

    start_time = time.time()

    for i in range(500):
        batch = create_multibatch(model, THREADS)
        loss, total_reward = weasel_histories(model, batch)
        total_reward_history.append(total_reward)
        print("Iteration: ", i, " --- Loss: ", loss, "Reward: ", total_reward)
        if i%10 == 0:
            print("Elapsed time:", time.time() - start_time)

    print("Total training time: ", time.time() - start_time)
    plt.plot(total_reward_history, color="blue", label="Total")
    plt.plot(pd.DataFrame(total_reward_history).rolling(window=10).mean(), color="red", label="Moving average")
    plt.set_xlabel("Simulations")
    plt.set_ylabel("Reward")
    plt.legend()
    plt.show()

    model.save('model')
