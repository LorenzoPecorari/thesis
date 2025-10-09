import gymnasium
from gymnasium import spaces
import numpy as np
import interpol as ip

import random

class energyEnv(gymnasium.Env):
    def __init__(self, data, battery_capacity, power_idle, power_frame, delta_time, proc_interval):
        # irradiation data coming from dataset
        self.data = data
        
        # battery and system specs
        self.battery_level = 0.0
        self.battery_capacity = battery_capacity
        
        self.e_idle = power_idle * delta_time
        self.e_frame = power_frame * delta_time
        
        self.irrad = 0
        self.interval = proc_interval
        self.delta_time = delta_time
        
        # interal vars
        self.current_idx = 0
        self.inner_idx = 1
        self.action_space = spaces.Discrete(2)
        
        #   AZIONI:
        # 0 -> drop frame
        # 1 -> process frame
                
        self.obs_space = spaces.Box(
            low = -np.inf,
            high = np.inf,
            shape = (5, ),
            dtype = np.float32
        )
        
    def reset(self):
        self.current_idx = 0
        self.irrad = self.data.values[self.current_idx][1]
        return self.battery_level, self.irrad, self.e_idle, self.e_frame,  0
    
    def step(self, action):
        if(self.inner_idx % (self.delta_time / self.interval) == 0):
            self.current_idx += 1
            self.inner_idx = 1
        else:
            self.inner_idx +=1
        
        reward = self.calculate_reward(action)
        self.irrad = self.data.values[self.current_idx][1]
            
        return self.battery_level, self.irrad, self.e_idle, self.e_frame,  reward
    
    def update_battery_level(self, value):
        normalized_value = round(value / self.battery_capacity, 1)
        if((self.battery_level + normalized_value) > 1.0):
            self.battery_level = 1.0
        else:
            self.battery_level += normalized_value
            # print(self.battery_level, normalized_value)
    
    def calculate_reward(self, action):
        if(self.battery_level <= (self.e_frame + self.e_idle)):
            if(action == 1):
                self.battery_level = 0.0
                return -2
            else:
                self.update_battery_level(self.irrad * self.interval)
                return 1
        else:
            if(action == 1):
                self.update_batery_level((self.irrad * self.interval) - (self.e_frame + self.e_idle))
                return 1
            else:
                self.update_batery_level(self.irrad * self.interval)
                return -4
 
def main():

    df = ip.interpolate('../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M.csv', 15, 1)
    df_numeric = df[['dni', 'ghi']]
    
    battery = 0
    irradiation = 0
    idle = 0
    p = 0.0
    reward = 0
    partial_reward = 0

    env = energyEnv(df_numeric, 6000, 2.5, 7, 900, 10)
    
    for i in range(1000000):
        # battery, irradiation, idle, p, partial_reward = env.step(0)
        battery, irradiation, idle, p, partial_reward = env.step(0)
        reward += partial_reward
        print(f"battery: {battery}, irradiation: {irradiation}, idle: {idle}, frame: {p}, reward: {reward}")

main()