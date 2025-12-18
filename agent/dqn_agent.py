from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env


class DQNAgent:
    def __init__(self, env):
        train_env = make_vec_env(lambda: env, n_envs=1)
        
        self.model = DQN(
            policy="MlpPolicy",           # Use a multi-layer perceptron policy
            env=train_env,                # Your custom environment
            learning_rate=1e-3,           # Learning rate
            buffer_size=10000,            # Replay buffer size
            learning_starts=1000,         # Steps before training begins
            batch_size=64,                # Batch size for training
            tau=0.1,                      # Target network update rate
            gamma=0.99,                   # Discount factor
            train_freq=4,                 # Train every 4 steps
            target_update_interval=500,   # Update target network every 500 steps
        )
        self.model.learn(total_timesteps=100_000)
        env.reset()
        
    def take_action(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return action