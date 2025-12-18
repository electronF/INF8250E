import gymnasium as gym
import numpy as np

from env.models import BaseEnv


class ASEnv(BaseEnv):
    name="AS"

    def declare_action_space(self):
        self.action_space = gym.spaces.Box(
            low=0.0, high=1000000, shape=(2,), dtype=np.float32
        )

    def get_bid_ask_prices(self, action):
        bid_price, ask_price = action
        return bid_price, ask_price, False