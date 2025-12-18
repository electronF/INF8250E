import numpy as np


class ASAgent:
    def __init__(self, env, gamma = 0.1, sigma = 0, kappa = 1.5, T = 0):
        self.gamma = gamma
        self.kappa = kappa

        self.T = T
    def compute_spread(self, inventory):
        ret = (2 / self.gamma) * np.log(1 + self.gamma * inventory / self.kappa)
        return ret
    
    def take_action(self, observation):
        midpoint = observation[0]
        inventory = observation[2]
        spread = self.compute_spread(inventory)
        bid_price = midpoint - spread / 2
        ask_price = midpoint + spread / 2
        return bid_price, ask_price