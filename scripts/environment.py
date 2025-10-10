import gymnasium
from gymnasium import spaces
import numpy as np
import interpol as ip
import random

'''
 === Customized environment for energy-aware system with PV ===

Each status is represented as:
    S = {s_1, ..., s_n}
    if s_t belongs to S, so
        s_t = {battery_level(t-1), irradiance(t), time(t)}

where each variable is normalized over the range [0, 1].

It is needed to provide to the environment :
 - the path of a dataset of data related to irradiance measurements (coming from a source like Solcast),
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
                reward = -3
        else if (action == PROCESS)
            if (enough energy for processing frame)
                reward = +1
            else
                reward = -5
    
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
                 pv_area):
        
        super().__init__()
        
        # irradiation data coming from dataset
        self.data = ip.interpolate(datapath, delta_time/60, proc_interval/60)
        self.max_steps = len(self.data)
        
        # battery and system specs
        self.battery_level = 0.0
        self.battery_capacity = battery_capacity                 # [Wh]
        
        self.pv_area = pv_area                                  # [m^2]
        self.pv_efficiency = pv_efficiency                      # [Wh/m^2]
        
        # energy params
        self.e_idle = (power_idle * proc_interval) / 3600       # [Wh]
        self.e_frame = (power_frame * proc_interval) / 3600     # [Wh]
        
        self.irrad = 0                                          # [W/m^2]
        self.max_irrad = max_irradiation                        # [W/m^2]
        self.interval = proc_interval                           # [s]
        self.delta_time = delta_time                            # [min]
        
        # interal vars
        self.current_step = 0
        self.inner_step = 1
        self.steps_per_interval = int(delta_time / proc_interval)
        
        # frames metrics
        self.total_frames_processed = 0
        self.total_frames_dropped = 0
        
        self.time = 0.0
        
        # ACTIONS :
        # 0 -> drop frame
        # 1 -> process frame
        self.action_space = spaces.Discrete(2)
        
        # obs = [battery_level, irradiation, e_idle, e_frame, time_day]        
        self.observation_space = spaces.Box(
            low = np.array([0.0, 0.0, 0.0]),
            high = np.array([1.0, 1.0, 1.0]),
            dtype = np.float64
        )
    
    def reset(self, seed):
        if(seed == None):
            super().reset()
        else:
            super.reset(seed)
        
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
        if(self.inner_step % self.steps_per_interval == 0):
            self.current_step += 1
            self.inner_step = 1
        else:
            self.inner_step += 1
        
        self.irrad = self.get_irradiance()
        e_pv = self.get_pv_energy(self.irrad * self.max_irrad)
        
        reward = self.calculate_reward(action, e_pv)
        
        terminated = (self.current_step >= self.max_steps)
        truncated = False
        
        obs = self.get_observation()
        info = self.get_info()
        
        return obs, reward, terminated, truncated, info
    
    def get_irradiance(self):
        if(self.current_step >= self.max_steps):
            return 0.0
        
        return round(self.data.iloc[self.current_step]['ghi'] / self.max_irrad, 2)
    
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
        needed_energy = self.e_frame + self.e_idle
        available_energy = actual_battery_energy + panel_energy
        
        reward = 0
        
        # action is "drop the frame"
        if(action == 0):
            self.total_frames_dropped += 1

            if(available_energy < needed_energy):
                self.update_battery_level(panel_energy - self.e_idle)
                reward = 1
            else:
                # it should have processed the frame
                # thanks to enough energy in the battery, weak penalty
                self.update_battery_level(panel_energy - self.e_idle)
                reward = -3     
        
        # action is "process the frame"
        elif(action == 1):
            if(available_energy < needed_energy):
                # it should have dropped the frame
                # due to unsufficient energy in the battery, strong penalty
                self.update_battery_level(panel_energy - needed_energy)
                reward = -5
            else:
                self.update_battery_level(panel_energy - needed_energy)
                self.total_frames_processed += 1
                reward = 1
        
        return reward

    def get_info(self):
        return {
            'step': self.current_step,
            'irradiance': self.irrad,
            'battery_level': self.battery_level,
            'battery_wh': self.battery_level * self.battery_capacity,
            'frames_processed': self.total_frames_processed,
            'frames_dropped': self.total_frames_dropped,
        }

    def get_observation(self):
        self.irrad = self.get_irradiance()
        # time expressed as fraction of day completed wrt delta_time of measurements in dataset
        self.time = round((self.current_step % self.delta_time) / self.delta_time, 2)
        
        obs = np.array([
            round(self.battery_level, 2),
            round(self.irrad, 2),
            round(self.time, 2)            
        ], dtype=np.float64)
        
        return obs
        
 
def main():

    datapath = '../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M.csv'
    battery_capacity = 100                  # [Wh]
    power_idle = 2.5                        # [W]
    power_frame = 7.0                       # [W]
    delta_time = 15 * 60            # [sec]
    proc_interval = 1 * 60      # [sec]
    max_irrad = 1200                        # [W/m^2]
    pv_efficiency = 0.2
    pv_area = 1.0

    env = EnergyPVEnv(
        datapath,
        battery_capacity,
        power_idle,
        power_frame,
        delta_time,
        proc_interval,
        max_irrad,
        pv_efficiency,
        pv_area
    )
    
    obs, info = env.reset(None)
    total_reward = 0
    
    for i in range(100000):
        # action = 1
        action = random.randint(0, 1)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"action: {action}, battery: {obs[0]}, irrad: {obs[1]}, time: {obs[2]}")
        