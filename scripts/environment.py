import gymnasium
from gymnasium import spaces
import numpy as np
import pandas as pd

class energyEnv(gymnasium.Env):
    def __init__(self, data, energy_idle, energy_frame, delta_time):
        # irradiation data
        self.data = data
        
        # battery and system specs
        self.battery_level = 0.0
        self.e_idle = energy_idle
        self.e_f = energy_frame
        self.interval = delta_time
        
        self.current_idx = 0
        self.action_space = spaces.Discrete(2)
                
        self.obs_space = spaces.Box(
            low = np.inf,
            high = np.inf,
            shape = (len(self.data.values[0]), ),
            dtype = np.int16
        )
        
    def reset(self):
        self.current_idx = 0
        return self.data.values[self.current_idx]
    
    def step(self, action):
        self.current_idx += 1
        reward = self.calculate_reward(action)
        done = self.current_idx >= len(self.data - 1)
        
        obs = None
        if(not done):
            obs = self.data.values[self.current_idx]
        else:
            obs = np.zeros(self.observation_space.shape)
            
        return obs, reward, done, {}
    
    def calculate_reward(self, action):
        pass
    
# dataset = np.random.rand(100, 4)
# env = energyEnv(dataset)
# obs, reward, done, _ = env.step(0)
# print(f"obs: {obs},\n reward: {reward},\n done: {done}")

df = pd.read_csv('../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M.csv')
df_numeric = df.select_dtypes(include=[np.number])  # Solo colonne numeriche

# env = energyEnv(df_numeric)
# obs = env.reset()
# print(f"Initial obs: {obs}")
# print(env.step(0))