import pandas as pd


import gymnasium as gym
import numpy as np

from env.models import BaseEnv, TypeOfReward


class PPOEnv(BaseEnv):
    name="Discrete"
    n = 100
    
    def declare_action_space(self):
        self.action_space = gym.spaces.MultiDiscrete([self.n+1, self.n+1]) 
        
    def get_bid_ask_prices(self, action):
        bid_action, ask_action = action
        
        current_row = self.lob_data.iloc[self.t]
        
        if self.reward_type == TypeOfReward.REWARD_1:
            upper_spread = (current_row['midpoint'] + current_row['spread']) - self.rolling_avg
            lower_spread = self.rolling_avg - (current_row['midpoint'] - current_row['spread'])
            
            invalid = False
            if lower_spread < 0 or upper_spread < 0:
                invalid = True
                
            bid_price = self.rolling_avg - lower_spread * (ask_action/self.n)
            ask_price = self.rolling_avg + upper_spread * (ask_action/self.n)
            
            return bid_price, ask_price, invalid
        
        else:
            bid_price = current_row['midpoint'] - current_row['spread']*(bid_action/self.n)
            ask_price = current_row['midpoint'] + current_row['spread']*(bid_action/self.n)

            return bid_price, ask_price, False
    
