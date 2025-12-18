import pandas
import pandas as pd
import enum

import gymnasium as gym
import numpy as np



class TypeOfReward(enum.Enum):
    REWARD_1 = 0
    REWARD_2 = 1



class BaseEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, lob_data, initial_cash=500_000, initial_inventory=0, trade_volume=1, inventory_penalty=0.001, reward_type:TypeOfReward=TypeOfReward.REWARD_1):
        super(BaseEnv, self).__init__()

        # lob_data: pd.DataFrame with flexible structure
        self.lob_data = lob_data if isinstance(lob_data, pd.DataFrame) else pd.DataFrame(lob_data)
        self.initial_cash = initial_cash
        self.initial_inventory = initial_inventory
        self.trade_volume = trade_volume
        self.inventory_penalty = inventory_penalty
        self.reward_type = reward_type

        self._reset_state()

        # Observation space inferred from lob_data columns
        self.observation_columns = [col for col in lob_data.columns if col not in ['system_time']]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.observation_columns) + 1,), dtype=np.float32
        )
        
        self.rolling_avg = self.lob_data.iloc[0]['midpoint']
        
        self.declare_action_space()


    def _reset_state(self):
        self.t = 0
        self.cash = float(self.initial_cash)
        self.inventory = float(self.initial_inventory)
        self.prev_valuation = self._current_mark_to_market()

    def _current_mark_to_market(self):
        midpoint = self.lob_data.iloc[self.t]['midpoint']
        return self.cash + self.inventory * midpoint

    def reset(self, seed=0):
        self._reset_state()
        return self._get_obs(), {}

    def _get_obs(self):
        # Dynamically extract observation based on columns
        row = self.lob_data.iloc[self.t]
        return np.array([row[col] for col in self.observation_columns] + [self.rolling_avg], dtype=np.float32)

    def step(self, action):
        done = False

        if self.t >= len(self.lob_data) - 1:
            done = True
            reward = 0.0
            return self._get_obs(), reward, done, {}
        
        bid_price, ask_price, invalid = self.get_bid_ask_prices(action)

        self.t += 1
        next_row = self.lob_data.iloc[self.t]
        next_best_bid = next_row['midpoint'] - next_row['spread'] / 2
        next_best_ask = next_row['midpoint'] + next_row['spread'] / 2
        
        if self.t < 5:
            self.rolling_avg = next_row['midpoint'] 
        bought = 0
        sold = 0

        # Determine if buy happened
        can_buy = (bid_price > next_best_bid) and (self.cash >= bid_price * self.trade_volume) and not invalid
        if can_buy:
            self.cash -= bid_price * self.trade_volume
            self.inventory += self.trade_volume
            bought = (bid_price - self.rolling_avg) * self.trade_volume

        # Determine if sell happened
        can_sell = (ask_price < next_best_ask) and (self.inventory >= self.trade_volume) and not invalid
        if can_sell:
            self.cash += ask_price * self.trade_volume
            self.inventory -= self.trade_volume
            sold = (ask_price - self.rolling_avg) * self.trade_volume
            
        self.rolling_avg += 0.001 * (next_row['midpoint'] - self.rolling_avg)

        if self.reward_type == TypeOfReward.REWARD_1:
            pnl_change = sold + bought
            trading_rew = 5 * int(sold != 0) + 5 * int(bought != 0)
            reward = pnl_change + trading_rew
        else:
            new_valuation = self.cash + self.inventory * next_row['midpoint']
            pnl_change = new_valuation - self.prev_valuation
            inv_penalty = 0
            if self.inventory > 150:
                inv_penalty = -self.inventory_penalty * ((self.inventory-150)**2)
            reward = pnl_change + inv_penalty
            self.prev_valuation = new_valuation

        if self.t == len(self.lob_data) - 1:
            done = True

        return self._get_obs(), reward, done, False, {}

    def render(self, mode='human'):
        print("Timestep:", self.t)
        print("Cash:", self.cash)
        print("Inventory:", self.inventory)
        print("Current Valuation:", self._current_mark_to_market())
        if mode == 'human':
            row = self.lob_data.iloc[self.t]
            print("LOB Data at Timestep {}: {}".format(self.t, row.to_dict()))
            
    def get_bid_ask_prices(self, action):
        raise NotImplementedError("get_bid_ask_prices(self, action) has to be implemented")
    
    def declare_action_space(self):
        raise NotImplementedError("declare_action_space(self) has to be implemented")