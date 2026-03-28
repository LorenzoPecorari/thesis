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
import random

# custom script
import interpol as ip

'''
 === Customized environment for energy-aware system with PV ===

Each status is represented as:
    S = {s_1, ..., s_n}
    if s_t belongs to S, so
        s_t = {battery_level(t-1), time(t)}

where each variable is normalized over the range [0, 1].

It is needed to provide to the environment :
 - the path of a dataset of data related to irradiance measurements (coming from Solcast),
 - the nominal capacity of the battery,
 - the power at idle and the peak power required by the hardware,
 - the frequency of measurements and the desired computational frequency in seconds,
 - an estimation of the maximum irradiance reachable,
 - the efficiency and the area of the photovoltaic panel.
 
There are (fps + 1) action that can be chosen and each of them tell which is the framerate
to adopt in such step on the environment in order to compute the frames that arrives in the backlog at each time.

If the action chosen allow to not drain all the energy stored in the battery until that moment,
the reward will be the number of frames taken from the backlog and computed, so (fps * interval between each step).
Otherwise, the reawrd will be 0 and the number of frames extracted from the backlog and computed will be based on 
the energy of the battery.   
 
    
'''


class EnergyPVEnv(gymnasium.Env):
    def __init__(self, 
                 datapath,
                 battery_capacity,
                 backlog_capacity,
                 power_idle,
                 power_max,
                 delta_time,
                 proc_interval,
                 max_irradiation,
                 pv_efficiency,
                 pv_area,
                 fps,
                 arrival_rate):
        
        super().__init__()
        
        # irradiation data coming from dataset
        self.datapath = datapath
        self.data = ip.interpolate(datapath, delta_time, proc_interval)

        # battery and system specs
        self.battery_level = 0.0
        self.battery  = 0
        self.battery_capacity = battery_capacity * 3600             # [Wh -> Ws = J]
        
        self.backlog = 0
        self.backlog_capacity = backlog_capacity * 100000000000        # UNBOUNDED!
        self.backlog_level = 0
        
        self.pv_area = pv_area                                      # [m^2]
        self.pv_efficiency = pv_efficiency                          # [%]
        
        # energy params
        self.p_idle = power_idle
        self.e_idle = power_idle * proc_interval                    # [Ws = J over defined interval]
        self.e_frame = (((power_max - power_idle)) / fps)                           # [Ws = J] per singolo frame in un secondo, da rivedere
        # self.e_frame = ((power_max - power_idle) / 30)
        
        self.energy_consumption = 0.0
        
        self.irrad = 0                                              # [W/m^2]
        self.max_irrad = max_irradiation                            # [W/m^2]
        self.interval = proc_interval                               # [s]
        self.delta_time = delta_time                                # [s]
        self.fps = fps                                              # [1/s]
        self.arrival_rate = arrival_rate
        
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
        
        # temporal metrics
        self.time = 0.0
        self.year = self.data.index[0].year
        self.equinox = datetime.datetime(self.year, 3, 20)
        self.day = 0

        # ACTIONS :
        # 0, 1, ..., fps -> tells fps to process

        self.action_space = spaces.Discrete(self.fps+1)
        
        self.observation_space = spaces.Box(
            low = np.array([0.0, 0.0, 0.0]),
            high = np.array([1.0, 3.0, 1.0]),
            dtype = np.float64
        )
    
    def reset(self, seed=None, **kwargs):
        if(seed == None):
            super().reset(seed = seed)
            self.day = random.randint((datetime.datetime(self.year, 1, 1) - self.equinox).days, (datetime.datetime(self.year, 12, 31) - self.equinox).days)
        
        elif(seed == "linear"):
            super().reset()
            self.day = (self.day + 1) % 365
        
        elif seed == "fixed_summer":
            super().reset()
            self.day = 172
        
        elif seed == "fixed_winter":
            super().reset()
            self.day = 355
            
        elif isinstance(seed, int):
            super().reset()
            self.day = seed % 365
        
        self.battery_level = 0.5
        self.battery = 0.5 * self.battery_capacity
        
        self.backlog = 0
        self.backlog_level = self.calculate_backlog()
        
        self.inner_step = 0
        self.current_step = 0
        self.irrad = self.get_irradiance()
        
        self.energy_consumption = 0.0

        self.total_frames_processed = 0
        self.total_frames_dropped = 0
        
        obs = self.get_observation()
        info = self.get_info()

        return obs, info
    
    def calculate_backlog(self):
        qty = self.backlog
        max_backlog = self.interval * self.arrival_rate * 10
        
        if(qty == 0):
            return 0
        elif(qty > 0 and qty < int(max_backlog / 3)):
            return 1
        elif(qty >= int(max_backlog / 3) and qty < int((2/3) * max_backlog)):
            return 2
        else:
            return 3
        

    def step(self, action):
        self.inner_step += 1
        
        if(self.inner_step % self.steps_per_interval == 0):
            self.current_step += 1
            
            if(self.current_step % self.steps_per_day_logic == 0):
                self.day += 1
        
        self.irrad = self.get_irradiance()
        e_pv = self.get_pv_energy(self.irrad * self.max_irrad)
        self.backlog += (self.arrival_rate * self.interval)
        
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
        
        return self.data.iloc[idx]['ghi'] / self.max_irrad
    
    def get_pv_energy(self, irradiance):
        if(irradiance <= 0.0):
            return 0.0
        
        
        power_pv = irradiance * self.pv_area * self.pv_efficiency
        # power_pv = min(irradiance * self.pv_area * self.pv_efficiency, 10)
        # print(f"pv power: {irradiance * self.pv_area * self.pv_efficiency} VS {power_pv} => energy: {power_pv * self.interval}")
        energy_pv = power_pv * self.interval

        # print(self.battery_level, energy_pv, power_pv, irradiance)
        # input()

        return energy_pv
    
    def update_battery_level(self, value):
        normalized_value = value / self.battery_capacity
        
        if((normalized_value) > 1.0):
            # print(f"battery: {self.battery} - battery_level: {self.battery_level} - normalized_value: {normalized_value}")
            self.battery = self.battery_capacity
            self.battery_level = 1.0
        elif((normalized_value) < 0.0):
            self.battery_level = 0.0
            self.battery = 0.0
        else:
            self.battery = value
            self.battery_level = normalized_value

    
    def calculate_reward(self, action, panel_energy):
        battery = self.battery
        actual = battery + panel_energy
        needed = (action * self.interval * self.e_frame) + self.e_idle
        
        processable = max(min(int((actual - self.e_idle) / self.e_frame), self.fps * self.interval, self.backlog), 0)
        processed = 0
        backlog = self.backlog
        
        reward = 0
        
        if(actual > needed and processable > 0):
            # print(actual, needed, panel_energy)
            # input()
            # input(f"{backlog} - old")
            actual = max(actual - needed, 0)
            # print(battery)
            
            processed = min(processable, action * self.interval)
            new_backlog = self.backlog - processed
            # input(f"{backlog} - new")
            
            # self.update_battery_level(panel_energy - ((processed * self.e_frame) + self.e_idle))
            # self.backlog = max(self.backlog - processed, 0)
            # self.total_frames_processed += processed

            try:
                reward = (processed / processable) + (actual / self.battery_capacity) + (processed / backlog)
                # return processed / processable
            except:
                reward = (processed / processable) + (actual / self.battery_capacity)
            finally:
                backlog = new_backlog
        else:
            actual = panel_energy - self.e_idle
            # self.update_battery_level(panel_energy - self.e_idle)
            
            if(processable == 0 and action == 0):
                reward = (actual / self.battery_capacity)
            else:
                reward = 0
                
        self.update_state(actual, backlog, processed)
        return reward
    
    def update_state(self, battery, backlog, processed):
        self.update_battery_level(battery)
        self.backlog = max(backlog, 0)
        self.total_frames_processed += processed
        
        # input()

        return
        
        # processed = min(processable, action * self.interval)
        
        # needed = processed * self.e_frame
        
        # self.update_battery_level(panel_energy - ((processed * self.e_frame) + self.e_idle))
        
        # if(actual > needed):
        #     if(processable > 0):
        #         reward = (processed / processable) * self.battery_level * 100
        #         self.backlog -= processed
        #         self.total_frames_processed += processed
        #         return reward
        #     else:
        #         return -100
        # else:
        #     self.backlog -= processed
        #     self.total_frames_processed += processed
        #     return -100
            
            
        if(actual > needed and self.battery_level > 0.0):
            try:
                reward = (processed / processable) * self.battery_level * 100
                self.backlog -= processed
                self.total_frames_processed += processed
                return reward
                # return processed / processable
            except:
                return 0
        else:
            return -100
            # return -100

        # if(processable > 0):
        #     try:
        #         return processed
        #     except:
        #         return 0
        
        return 0    
    
    # (G)old 
    def calculate_reward_old(self, action, panel_energy):
                
        battery = self.battery_level * self.battery_capacity
        actual = battery + panel_energy
        
        # if self.irrad < 0.1 and self.battery_level < 0.35:
        # # Override dell'azione
        #     processed = 0
            
        #     # Consuma solo idle
        #     self.update_battery_level(panel_energy - self.e_idle)
        #     self.energy_consumption = self.e_idle
        #     self.backlog += (self.fps * self.interval)
        #     self.total_frames_processed += 0
            
        #     # Reward minimo (non zero per non confondere)
        #     return 0.1
        
        needed = action * self.interval * self.e_frame + self.e_idle
        
        processable = min(int((actual - self.e_idle) / self.e_frame), self.fps * self.interval)
        processed = min(processable, action * self.interval)
        
        self.update_battery_level(panel_energy - ((processed * self.e_frame) + self.e_idle))
        
        self.energy_consumption = (processed * self.e_frame) + self.e_idle
        
        self.backlog -= processed
        self.total_frames_processed += processed
        
        reward = 0

        # if(processable == 0 or self.battery_level <= 0.2):
        #     reward = 0
        # elif(processable > 0):
        #     reward = round((self.battery_level ** 2) * (processed/processable) * processed, 2)
        
        if(processable == 0):
          return 0  
        
        if(self.battery_level <= 0.2):
            try:
                return round((self.battery_level ** 2) * (processed/processable), 2)
            except:
                return 0
        
        try:
            return round((self.battery_level ** 1) * (processed/processable), 2)
        except:
            return 0
            
        # return reward

        # if(processable > 0):
        #     efficiency = processed / processable
        #     # reward = processed * efficiency * self.battery_level
        #     reward = (processed / processable) * processed * (self.battery_level ** 2)

        # return reward
        
        # if(actual > needed):
        #     try:
        #         return (processed / processable) * 100
        #     except:
        #         return 0
        # else:
        #     return 0
            
        
        # if(actual > needed):
        #     processable = min(int((actual - self.e_idle) / self.e_frame), self.fps * self.interval)
        #     processed = min(processable, action * self.interval)
            
        #     # return int((processed)/ processable)
        #     try:
        #         return processed / processable
        #     except:
        #         return -1
        #     # return int(processed / self.interval)
        # else:
        #     return 0
        
        # battery = self.battery_level * self.battery_capacity
        # requested = action * self.interval
        # processable = min(self.fps * self.interval, int((battery - self.e_idle + panel_energy) / self.e_frame))
        
        # if(requested >= processable):
        #     self.update_battery_level((battery - self.e_idle + panel_energy - (processable * self.e_frame)))
        #     self.backlog -= processable
        #     self.total_frames_processed += processable
        #     return 0
        # else:
        #     self.update_battery_level((battery - self.e_idle + panel_energy - (requested * self.e_frame)))
        #     self.backlog -= requested
        #     self.total_frames_processed += requested
        #     return (requested / processable) * 100
        
        '''
        actual_battery_energy = self.battery_level * self.battery_capacity

        frames_per_interval = action * self.interval
        # print(f"fps: {self.fps} - action: {action} - fpi: {frames_per_interval}")
        energy_needed = (frames_per_interval * self.e_frame) + self.e_idle
        
        # print(f"actual_battery_energy: {actual_battery_energy} - E_pv: {panel_energy} - needed: {energy_needed}")
        # input(f"Presse enter...")
        
        # VERSIONE 2
        max_processable = max(0, int(((actual_battery_energy + panel_energy) - self.e_idle) / self.e_frame))
        frames_processed = min(frames_per_interval, max_processable)
        energy_used = (frames_processed * self.e_frame) + self.e_idle
        
        
        input(f"HALT! requested: {frames_per_interval} - processed: {max_processable}")
        
        self.update_battery_level(panel_energy - energy_used)
        self.backlog -= frames_processed
        self.total_frames_processed += frames_processed
        
        # print(f"requested: {frames_per_interval} - processed: {frames_processed}")
        # input("...")
        
        if(frames_per_interval > max_processable):
            input(f"HALT! requested: {frames_per_interval} - processed: {max_processable}")
            return 0
        else:
            return 1
        
        # # REWARD: semplicemente i frame processati
        # return frames_processed
        
        # VERSIONE 1     
        if((actual_battery_energy + panel_energy) >= energy_needed):
            self.update_battery_level(panel_energy - energy_needed)
            self.backlog -= frames_per_interval
            self.total_frames_processed += frames_per_interval
            return frames_per_interval
            
            # # compute if other frames might be processed
            # computable = min(int((actual_battery_energy + panel_energy) / self.e_frame), self.fps * self.interval)
            # if(computable > frames_per_interval):
            #     return (frames_per_interval - computable)
            #     # return -1
            # else:
            #     # return 1
            #     return frames_per_interval
        
        else:
            # self.update_battery_level(panel_energy - self.e_idle)
            processable = int((actual_battery_energy + panel_energy) / self.e_frame)
            # print(f"estimated: {frames_per_interval} - processed: {processable} - ")
            # input("Press ENTER to continue...")
            # print(f"negative delta: {processable - frames_per_interval}")
            self.backlog -= processable
            self.update_battery_level(panel_energy - (self.e_idle + (processable * self.e_frame)))
            # return -1
            return (processable - frames_per_interval)
            
        '''

    def get_info(self):
        return {
            'step': self.current_step,
            'irradiance': self.irrad,
            'battery_level': self.battery_level,
            'battery_wh': self.battery_level * self.battery_capacity,
            'backlog_level': self.backlog,
            'energy_consumption': self.energy_consumption,
            'day': self.day,
            'frames_processed': self.total_frames_processed,
            'frames_dropped': self.total_frames_dropped,
            'frames_per_interval': self.frames_per_interval,
        }

    def get_observation(self):
        self.irrad = self.get_irradiance()
        self.backlog_level = self.calculate_backlog()
        self.time = round(self.inner_step / self.steps_per_day_data, 2)
        
        obs = np.array([
            round(self.battery_level, 2),
            round(self.backlog_level, 1),
            round(self.time, 2)
        ], dtype=np.float64)
        
        return obs