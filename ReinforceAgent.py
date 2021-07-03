import gym
import tensorflow as tf
import numpy as np
import VizdoomWrapper
import tensorflow.keras.utils as utils
import Models


class ReinforceAgent:
    def __init__(self, game='CartPole-v0'):
        self.model = Models.create_model(game)
        self.game = game
        self.training_reward_history = []
        # get actions space
        game_env = self.create_game_env()
        self.action_space = game_env.action_space.n
        game_env.close()

    class History:
        def __init__(self):
            self.states = []
            self.actions = []
            self.rewards = []

        def append(self, gamestate, action, reward):
            self.states.append(gamestate)
            self.actions.append(action)
            self.rewards.append(reward)

    def take_action(self, state):
        probabilities = self.model.predict(np.array([state]))
        action = np.random.choice(self.action_space, p=probabilities[0])
        return action

    def create_game_env(self, render=False):
        if self.game == "VizDoom":
            game_env = VizdoomWrapper.VizdoomWrapper(render=render)
        else:
            game_env = gym.make(self.game)
        return game_env

    def run_simulation(self, render=False):
        game_env = self.create_game_env(render)
        state = game_env.reset()
        history = self.History()
        done = False
        while not done:
            if render:
                game_env.render()
            action = self.take_action(state)
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

    def get_batch(self):
        while True:
            history = self.run_simulation()
            #if self.game != "VizDoom":
            #    history.states = np.array(history.states)
            self.training_reward_history.append(np.sum(history.rewards))
            x = np.array(history.states)
            y = self.format_rewards(history.actions, self.compute_discounted_reward(np.array(history.rewards)))
            yield x, y

    def train(self, epochs=1, batch_size=1, verbose=1, callbacks=None):
        training_history = self.model.fit(
            self.get_batch(),
            epochs=epochs,
            verbose=verbose,
            steps_per_epoch=1,
            callbacks=callbacks
        )

        return training_history, self.training_reward_history
