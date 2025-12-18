from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class PPOAgent:
    def __init__(self, env):
        train_env = DummyVecEnv([lambda: env])
        self.model = PPO("MlpPolicy", train_env, learning_rate=0.01)
        self.model.learn(total_timesteps=100_000)
        env.reset()
    
    def take_action(self, observation):
        action, _ = self.model.predict(observation)
        return action
