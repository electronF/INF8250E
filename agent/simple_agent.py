class SimpleAgent:
    def __init__(self, env, bid_spread_fraction=0.5, ask_spread_fraction=0.5):
        self.bid_spread_fraction = bid_spread_fraction
        self.ask_spread_fraction = ask_spread_fraction

    def take_action(self, observation):
        midpoint = observation[0]
        spread = observation[1]
        bid_price = midpoint - (spread * self.bid_spread_fraction)
        ask_price = midpoint + (spread * self.ask_spread_fraction)
        return bid_price, ask_price