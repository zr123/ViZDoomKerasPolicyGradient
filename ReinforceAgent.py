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


class ReinforceAgent:
    def __init__(self, game='CartPole-v0'):
        if game == "VizDoom":
            self.model = ReinforceAgent.create_vizdoom_model()
            game_env = VizdoomWrapper.VizdoomWrapper()
        else:
            self.model = ReinforceAgent.create_cartpole_model()
            game_env = gym.make(game)
        self.game = game
        self.action_space = game_env.action_space.n

    class History:
        def __init__(self):
            self.states = []
            self.actions = []
            self.rewards = []

        def append(self, gamestate, action, reward):
            self.states.append(gamestate)
            self.actions.append(action)
            self.rewards.append(reward)

    @staticmethod
    def policy_gradient_loss(reward, action_prob):
        loss = K.log(action_prob) * reward
        loss = K.sum(loss)
        return - loss

    @staticmethod
    def create_cartpole_model():
        model = Sequential()
        model.add(Input(shape=(4,)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer='adam', loss=ReinforceAgent.policy_gradient_loss)
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
        model.compile(optimizer='adam', loss=ReinforceAgent.policy_gradient_loss)
        return model

    def take_action(self, state):
        if self.game != "VizDoom":
            state = np.array([state])
        probabilities = self.model.predict(state)
        action = np.random.choice(self.action_space, p=probabilities[0])
        return action

    def run_simulation(self, render=False):
        if self.game == "VizDoom":
            game_env = VizdoomWrapper.VizdoomWrapper(render=render)
        else:
            game_env = gym.make(self.game)
        state = game_env.reset()
        history = self.History()
        done = False
        while not done:
            if render:
                game_env.render()
            action = self.take_action(state)  # , _ = take_probabilistic_action(actor, state)
            new_state, reward, done, _ = game_env.step(action)
            history.append(state, action, reward)
            state = new_state
        game_env.close()
        return history

    def play_and_display(self):
        history = self.run_simulation(render=True)
        return np.sum(history.rewards)

    @staticmethod
    def compute_discounted_reward(reward_history, discount_factor=0.99):
        dsr = tf.scan(lambda agg, x: discount_factor * agg + x, reward_history, reverse=True)
        # normalize
        return (dsr - tf.math.reduce_mean(dsr)) / tf.math.reduce_std(dsr)

    def format_rewards(self, action_history, reward_history):
        formatted_ah = utils.to_categorical(action_history, num_classes=self.action_space)
        formatted_rw = np.full((self.action_space, reward_history.shape[0]), reward_history).T
        return formatted_ah * formatted_rw

    def train(self):
        history = self.run_simulation()

        if self.game != "VizDoom":
            history.states = np.array(history.states)

        loss = self.model.train_on_batch(
            history.states,
            self.format_rewards(history.actions, self.compute_discounted_reward(np.array(history.rewards)))
        )

        return loss
