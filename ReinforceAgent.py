import concurrent

import gym
import tensorflow as tf
import numpy as np
import VizdoomWrapper
import tensorflow.keras.utils as utils
import Models
import matplotlib.pyplot as plt
import pandas as pd


class ReinforceAgent:
    def __init__(self, game='CartPole-v0', load_model_path=None):
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

    def get_batch(self, batch_size=1):
        while True:
            results = self.get_multibatch(batch_size)
            histories = [h for h in results]
            x = np.vstack([h.states for h in histories])
            y = np.vstack([
                self.format_rewards(h.actions, self.compute_discounted_reward(np.array(h.rewards)))
                for h in histories
            ])
            self.training_reward_history.append(np.mean([np.sum(h.rewards) for h in histories]))

            #history = self.run_simulation()
            #self.training_reward_history.append(np.sum(history.rewards))
            #x = np.array(history.states)
            #y = self.format_rewards(history.actions, self.compute_discounted_reward(np.array(history.rewards)))
            yield x, y

    def get_multibatch(self, batch_size):
        args = [False for i in range(batch_size)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            results = executor.map(self.run_simulation, args, timeout=120)
        return results

    def train(self, epochs=1, batch_size=1, verbose=1, callbacks=None):
        training_history = self.model.fit(
            self.get_batch(batch_size),
            epochs=epochs,
            verbose=verbose,
            steps_per_epoch=1,
            callbacks=callbacks
        )

        return training_history, self.training_reward_history

    def plot_training(self):
        plt.plot(self.training_reward_history, color="blue", label="Batch")
        plt.plot(
            pd.DataFrame(self.training_reward_history).rolling(window=10).mean(),
            color="red",
            label="Moving average (10 batches)")
        plt.xlabel("Epochs")
        plt.ylabel("Reward")
        plt.legend()
        plt.show()

    def save_model(self, path="models/VizdoomReinforceMultibatch", i=""):
        self.model.save(path + "/" + str(i))
        pd.DataFrame(self.training_reward_history).to_csv(path + "/VizdoomReinforceMultibatchRewards.csv", index=False)


if __name__ == '__main__':
    agent = ReinforceAgent("VizDoom")

    for i in range(25):
        history, reward_history = agent.train(epochs=20, batch_size=6)
        agent.plot_training()
        agent.save_model(i=i)
