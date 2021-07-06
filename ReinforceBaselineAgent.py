import concurrent

import gym
import tensorflow as tf
import numpy as np
import VizdoomWrapper
import tensorflow.keras.utils as utils
import Models
import matplotlib.pyplot as plt
import pandas as pd


class ReinforceBaselineAgent:
    def __init__(self, game='CartPole-v0', load_model_path=None):
        self.model = Models.create_model(game)
        self.baseline = Models.create_baseline_model(game)
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

    def create_game_env(self, render=False, frames_per_action=6):
        if self.game == "VizDoom":
            game_env = VizdoomWrapper.VizdoomWrapper(render=render, frames_per_action=frames_per_action)
        else:
            game_env = gym.make(self.game)
        return game_env

    def run_simulation(self, render=False, frames_per_action=6):
        game_env = self.create_game_env(render, frames_per_action=frames_per_action)
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

    def play_and_display(self, frames_per_action=6):
        history = self.run_simulation(render=True, frames_per_action=frames_per_action)
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
            # process histories
            histories = [h for h in results]
            x = np.vstack([h.states for h in histories])
            actions = np.hstack([h.actions for h in histories])
            #y = np.hstack([
            discounted_rewards = np.hstack([
                self.compute_discounted_reward(np.array(h.rewards))
                for h in histories
            ])
            self.training_reward_history.append(np.mean([np.sum(h.rewards) for h in histories]))
            # calculate advantage and update baseline
            advantage = discounted_rewards - self.baseline.predict(x).flatten()
            self.baseline.train_on_batch(x, discounted_rewards)
            # devide by the batch_size to get the mean of the batch
            advantage = self.format_rewards(actions, advantage)
            advantage = advantage / batch_size
            yield x, advantage

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

    def save_model(self, path="models/VizdoomReinforceBaseline"):
        self.model.save(path + "/m_" + str(len(self.training_reward_history)))
        self.baseline.save(path + "/b_" + str(len(self.training_reward_history)))
        pd.DataFrame(self.training_reward_history).to_csv(path + "/rewards.csv", index=False)

    @staticmethod
    def load_model(path, game):
        agent = ReinforceBaselineAgent(game)
        agent.training_reward_history = pd.read_csv(path + "/rewards.csv").iloc[:, 0].tolist()
        agent.model = tf.keras.models.load_model(path + "/m_" + str(len(agent.training_reward_history)), compile=False)
        agent.model.compile(optimizer='adam', loss=Models.policy_gradient_loss)
        agent.baseline = tf.keras.models.load_model(path + "/b_" + str(len(agent.training_reward_history)))
        return agent


if __name__ == '__main__':
    #agent = ReinforceBaselineAgent("VizDoom")
    agent = ReinforceBaselineAgent.load_model("models/VizdoomReinforceBaseline", "VizDoom")

    for i in range(25):
        history, reward_history = agent.train(epochs=20, batch_size=6)
        agent.plot_training()
        agent.save_model()
