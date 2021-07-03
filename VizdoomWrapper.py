import numpy as np
from vizdoom import DoomGame
from gym.spaces import Discrete
import cv2


class VizdoomWrapper:
    def __init__(self, render=False, frames_per_action=6):
        self.action_space = Discrete(3)
        self.actions = np.identity(self.action_space.n).tolist()
        self.frames_per_action = frames_per_action
        self.processed_screen_width = 160
        self.processed_screen_heigth = 120
        self.game = DoomGame()
        self.game.load_config("scenarios/health_gathering.cfg")
        self.game.set_doom_scenario_path("scenarios/health_gathering.wad")
        # default rewards are way too high
        self.game.set_living_reward(0.01)
        self.game.set_death_penalty(3.84)
        #self.game.set_render_hud(True)
        self.game.set_window_visible(render)
        self.game.init()

    def reset(self):
        self.game.new_episode()
        return self.convert_state(self.game.get_state())

    def step(self, action):
        reward = self.game.make_action(self.actions[action], self.frames_per_action)

        return (
            self.convert_state(self.game.get_state()),
            reward,
            self.game.is_episode_finished(),
            {}
        )

    def render(self):
        return

    def close(self):
        self.game.close()

    def convert_state(self, state):
        if state is None:
            return None
        return self.preprocess_screen(state.screen_buffer)

    def preprocess_screen(self, screen):
        # vizdoom used channel-first, cv2 uses channel-last
        screen = screen.transpose((1, 2, 0))
        # grayscale
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        # resize
        screen = cv2.resize(screen, (self.processed_screen_width, self.processed_screen_heigth))
        # normalize
        screen = screen / 255.0
        return screen
