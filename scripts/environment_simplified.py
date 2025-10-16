# gym api
import gymnasium
from gymnasium import spaces
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
import os

# numpy numbers management
import numpy as np

# others
import random
import datetime

# custom script
import interpol as ip

'''
 === Customized environment for energy-aware system with PV ===

Each status is represented as:
    S = {s_1, ..., s_n}
    if s_t belongs to S, so
        s_t = {battery_level(t-1), irradiance(t), time(t)}

where each variable is normalized over the range [0, 1].

It is needed to provide to the environment :
 - the path of a dataset of data related to irradiance measurements (coming from Solcast),
 - the nominal capacity of the battery,
 - the power at idle and the avg needed for computing a frame,
 - the frequency of measurements and the desired computational frequency in seconds,
 - an estimation of the maximum irradiance reachable,
 - the efficiency and the area of the photovoltaic panel.
 
 For now, only two actions are availables: DROP and PROCESS; both identified as 0 and 1.
 
 The reward function defined is the following:
    let A = {DROP, PROCESS}
    if action belongs to A, so:
        if (action == DROP)
            if (not enough energy for processing)
                reward = +1
            else
                reward = 0
        else if (action == PROCESS)
            if (enough energy for processing batch of frames)
                reward = +1
            else
                reward = 0
    
'''


class EnergyPVEnv(gymnasium.Env):
    def __init__(self, 
                 datapath,
                 battery_capacity,
                 power_idle,
                 power_frame,
                 delta_time,
                 proc_interval,
                 max_irradiation,
                 pv_efficiency,
                 pv_area,
                 fps):
        
        super().__init__()
        
        # irradiation data coming from dataset
        self.datapath = datapath
        self.data = ip.interpolate(datapath, delta_time, proc_interval)

        # battery and system specs
        self.battery_level = 0.0
        self.battery_capacity = battery_capacity                # [Wh]
        
        self.pv_area = pv_area                                  # [m^2]
        self.pv_efficiency = pv_efficiency                      # [Wh/m^2]
        
        # energy params
        self.e_idle = (power_idle * proc_interval) / 3600       # [Wh]
        self.e_frame = (power_frame) / 3600                     # [Wh] per singolo frame
        
        self.irrad = 0                                          # [W/m^2]
        self.max_irrad = max_irradiation                        # [W/m^2]
        self.interval = proc_interval                           # [s]
        self.delta_time = delta_time                            # [s]
        self.fps = fps                                          # [1/s]

        # value of images to process in batch
        self.frames_per_interval = int(fps * proc_interval)

        # interal vars
        self.current_step = 0
        self.inner_step = 0
        self.steps_per_interval = int(delta_time / proc_interval)
        self.steps_per_day_data = int((24 * 60 * 60) / proc_interval)
        self.steps_per_day_logic = int((24 * 60 * 60) / delta_time)
        self.max_steps = self.steps_per_day_data
        
        # frames metrics
        self.total_frames_processed = 0
        self.total_frames_dropped = 0
        
        # temporal metrics
        self.time = 0.0
        self.year = self.data.index[0].year
        self.equinox = datetime.datetime(self.year, 3, 20)
        self.day = 0


        # ACTIONS :
        # 0 -> drop frame
        # 1 -> process frame
        self.action_space = spaces.Discrete(2)
        
        # obs = [battery_level, time_day]        
        self.observation_space = spaces.Box(
            low = np.array([0.0, 0.0]),
            high = np.array([1.0, 1.0]),
            dtype = np.float64
        )
    
    def reset(self, seed=None, **kwargs):
        if(seed == None):
            super().reset(seed = seed)
            self.day = random.randint((datetime.datetime(self.year, 1, 1) - self.equinox).days, (datetime.datetime(self.year, 12, 31) - self.equinox).days)
        
        elif(seed == "linear"):
            super().reset()
            self.day += 1
        
        self.battery_level = 0.5
        self.inner_step = 0
        self.current_step = 0
        self.irrad = self.get_irradiance()

        self.total_frames_processed = 0
        self.total_frames_dropped = 0
        
        obs = self.get_observation()
        info = self.get_info()

        return obs, info
    

    def step(self, action):
        self.inner_step += 1
        
        if(self.inner_step % self.steps_per_interval == 0):
            self.current_step += 1
            
            if(self.current_step % self.steps_per_day_logic == 0):
                self.day += 1
        
        self.irrad = self.get_irradiance()
        e_pv = self.get_pv_energy(self.irrad * self.max_irrad)
        
        reward = self.calculate_reward(action, e_pv)
        
        terminated = (self.inner_step >= self.max_steps)
        truncated = False
        
        obs = self.get_observation()
        info = self.get_info()
        
        return obs, reward, terminated, truncated, info
    
    def get_irradiance(self):
        if(self.inner_step >= self.max_steps):
            return 0.0
        
        equinox_day_of_year = (self.equinox - datetime.datetime(self.year, 1, 1)).days
        day_of_year = (equinox_day_of_year + self.day) % 365
        
        idx = day_of_year * self.steps_per_day_data + self.inner_step
        
        if idx < 0 or idx >= len(self.data):
            return 0.0
        
        return round(self.data.iloc[idx]['ghi'] / self.max_irrad, 2)
    
    def get_pv_energy(self, irradiance):
        if(irradiance <= 0.0):
            return 0.0
        
        power_pv = irradiance * self.pv_area * self.pv_efficiency
        energy_pv = (power_pv * self.interval) / 3600
        return energy_pv
    
    def update_battery_level(self, value):
        normalized_value = value / self.battery_capacity
        
        if((self.battery_level + normalized_value) > 1.0):
            self.battery_level = 1.0
        elif((self.battery_level + normalized_value) < 0.0):
            self.battery_level = 0.0
        else:
            self.battery_level = self.battery_level + normalized_value

    def calculate_reward(self, action, panel_energy):
        
        actual_battery_energy = self.battery_level * self.battery_capacity

        # energy for all the images of the batch        
        energy_for_frames = self.e_frame * self.frames_per_interval
        needed_energy = energy_for_frames + self.e_idle
        
        available_energy = actual_battery_energy + panel_energy
        
        reward = 0
        
        # action is "drop the frames batch"
        if(action == 0):
            self.total_frames_dropped += self.frames_per_interval

            if(available_energy < needed_energy):
                self.update_battery_level(panel_energy - self.e_idle)
                reward = 1
            else:
                self.update_battery_level(panel_energy - self.e_idle)
                reward = 0     
        
        # action is "process the frames batch"
        elif(action == 1):
            if(available_energy < needed_energy):
                self.update_battery_level(panel_energy - needed_energy)
                self.total_frames_dropped += self.frames_per_interval
                reward = 0
            else:
                self.update_battery_level(panel_energy - needed_energy)
                self.total_frames_processed += self.frames_per_interval
                reward = 1
        
        return reward

    def get_info(self):
        return {
            'step': self.current_step,
            'irradiance': self.irrad,
            'battery_level': self.battery_level,
            'battery_wh': self.battery_level * self.battery_capacity,
            'day': self.day,
            'frames_processed': self.total_frames_processed,
            'frames_dropped': self.total_frames_dropped,
            'frames_per_interval': self.frames_per_interval,
        }

    def get_observation(self):
        self.irrad = self.get_irradiance()
        self.time = round(self.inner_step / self.steps_per_day_data, 2)
        
        obs = np.array([
            round(self.battery_level, 2),
            round(self.time, 2)
        ], dtype=np.float64)
        
        return obs
