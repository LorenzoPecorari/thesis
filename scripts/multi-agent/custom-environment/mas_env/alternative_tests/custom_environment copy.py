from pettingzoo import ParallelEnv
from gymnasium import spaces
import matplotlib.pyplot as plt
from copy import copy

import functools
import numpy as np
import random

import interpol as ip
import jit_reward_function as jrf

class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, num_agents, irradiance_datapaths, delta_time, proc_interval, proc_rate, arr_rate, batteries, panel_surfaces, power_idle, power_max, w):
        super().__init__()
        
        self.agents = []
        self.possible_agents = [i for i in range(0, num_agents)]
        
        self._num_agents = num_agents
        self._processing_rate = proc_rate
        self._arrival_rate = arr_rate
        self._proc_interval = proc_interval
        
        self.p_idle = power_idle
        self.p_max = power_max
        
        self.max_irrad = 1000.0
        self.panel_efficiency = 0.2
        
        self.irradiance_data = []
        self.irradiance_arrays = []
        
        for filepath in irradiance_datapaths:
            df = ip.interpolate(filepath, delta_time, proc_interval)
            self.irradiance_data.append(df)
            
            self.irradiance_arrays.append(df['ghi'].values)
        
        for elem in self.irradiance_data:
            print(len(elem))
        
        self.irradiance_level = [0.0 for i in range(0, self._num_agents)]
        
        self.battery_capacities = [(battery*3600) for battery in batteries]
        self.battery_energies = [0.0 for i in range(0, self._num_agents)]
        self.panel_surfaces = panel_surfaces
        
        self.e_idle = power_idle * self._proc_interval
        self.e_frame = (0.8 * (power_max - power_idle) * 1) / proc_rate
        self.e_tx_rx = (0.2 * (power_max - power_idle) * 1) / proc_rate
        
        self.backlogs = [0 for i in range(0, self._num_agents)]
        
        # state_i = (battery_level_i, daily_completion_i)
        self.states = [[0.0, 0, 0.0] for i in range(0, self._num_agents)]
        self.actions = [[0.0, 0, 0.0, 0.0] for i in range(0, self._num_agents)]
        self.rewards = [0 for i in range(0, self._num_agents)]
        
        # internal counters for episode compeltion 
        self.timestep = 0
        self.max_steps = (3600 / proc_interval) * 5
        self.episode = 355
        self._w = w
        
        try:
            self.max_steps = int(24 * 60 * 60 / proc_interval)
        except:
            self.max_steps = 1

        # observation_space: [b_1, s_i, t_1, b_2, s_2, t_2, ..., b_n, s_n, t_n]
        # for each agent there are three variables in the range [0.0, 1.0] where
            # b_i -> "battery level"
            # s_i -> "storage level"
            # t_i -> "episode completion"
        self._action_spaces = {
            agent: spaces.MultiDiscrete([self._processing_rate + 1, 3, self._num_agents, self._processing_rate + 1]) for agent in self.possible_agents
        }
        
        # action_space: [f_i, x_i, g_i, h_i] where
            # f_i -> "local framerate"
            # x_i -> "offloading mode"
            # g_i -> "target node"
            # h_i -> "offloading framerate"
        self._observation_spaces = {
            agent: spaces.Box(
                low=np.array([0.0, 0.0, 0.0] * self._num_agents, dtype=np.float32),
                high=np.array([1.0, 3.0, 1.0] * self._num_agents, dtype=np.float32),
                dtype=np.float32
            ) 
            for agent in self.possible_agents
        
        }
        
        self.fs = [0 for i in range(0, self._num_agents)]
        self.hs = [0 for i in range(0, self._num_agents)]
        self.hs_counter = [0 for i in range(0, self.num_agents)]
        
        # self.r_battery = [0 for i in range(0, self._num_agents)]
        # self.r_frames = [0 for i in range(0, self._num_agents)]
        # self.r_cooperation = [0 for i in range(0, self._num_agents)]
        # self.r_backlog = [0 for i in range(0, self._num_agents)]
    
    # function for retrieving level of backlog
    def calculate_backlog_level(self, agent_id):
        qty = self.backlogs[agent_id]
        # max_storage = self._processing_rate * self._proc_interval
        
        # DA TESTARE VERSIONE CON CAPACITÀ PARI A 10 INTERVALLI
        max_storage = self._processing_rate * self._proc_interval * 10
        # max_storage = self._processing_rate * self._proc_interval * (3600 / self._proc_interval)
        
        # return 1 - qty/max_storage
        
        # return (1 - qty / (self._processing_rate * 3600))
        
        
        if(qty == 0):
            return 0
        elif(qty > 0 and qty < int(max_storage / 3)):
            return 1
        elif(qty >= int(max_storage / 3) and qty < int((2/3) * max_storage)):
            return 2
        else:
            return 3

    # function for evaluating reward over frames management
    def calculate_reward_frames(self, agent_id, fti, xti, gti, hti, xt_gti, gt_gti, ht_gti):
        # load = (fti + hti)
        # if(load <= self._processing_rate):
        #     return load / self._processing_rate        
        # # return min(1, (fti + hti)/self._processing_rate)
        
        # return -(load - self._processing_rate) / self._processing_rate
        
        # ATTENZIONE: cambiato da proc_rate a arrival_rate dentro al rapporto framerate a quanto processare
        
        if((fti + hti) <= self._processing_rate):
            if(xti == 0):
                return fti / self._processing_rate
                # return 1
                # return (fti / self._arrival_rate)

            elif(xti == 1 and gti != 0 and gti != agent_id and xt_gti == 2 and gt_gti == agent_id):
                # hti = max(hti, ht_gti)
                # return (fti + hti)/self._processing_rate
                
                return (fti + min(hti, ht_gti)/ self._processing_rate)

                # return (fti + min(hti, ht_gti)/ self._arrival_rate)
                # return (fti / self._arrival_rate) + (hti / (self._processing_rate + self._arrival_rate))
                
                # return (fti / self._arrival_rate) + (min(hti, ht_gti) / self._arrival_rate)
                
                # return 1
                
                
            elif(xti == 2 and gti != 0 and gti != agent_id and xt_gti == 1 and gt_gti == agent_id):
                # return (fti / self._arrival_rate) + (min(hti, ht_gti) / self._arrival_rate)
                # if(hti != 0):
                #     return (fti/self._arrival_rate) + ((self._processing_rate - fti)/hti)
                # else:
                #     return (fti/self._arrival_rate)
                # return (fti + min(hti, ht_gti)) / self._processing_rate               
                # return (fti / self._arrival_rate) + (min(hti, ht_gti) / (self._processing_rate - self._arrival_rate))

                hti = min(hti, ht_gti)
                return (fti+hti) / self._processing_rate
    
                # return 1

            else:
                return fti / self._processing_rate

        else:
            return 0 
            # excess_penalty = ((fti + hti) - self._processing_rate) / (fti + hti)
            # return max(0, 1 - excess_penalty)        
        # if((fti + hti) < self._processing_rate):
        #     return 1
        # else:
        #     return 0
        
    # function for evaluating reward over battery management
    def calculate_reward_battery(self, agent_id, fti, xti, gti, hti, ft_gti, xt_gti, gt_gti, ht_gti, eb_t, ef_loc, e_tx_rx, pv):

        consumption = self.e_idle
        reward = 0
        
        if(xti == 0):
            consumption += (fti * ef_loc * self._proc_interval)
            
        elif(xti == 1):
            if(gti != 0 and gti != agent_id and xt_gti == 2 and gt_gti == agent_id):
                consumption += ((fti * ef_loc * self._proc_interval) + (min(hti, ht_gti) * e_tx_rx * self._proc_interval))
            else:
                consumption += (fti * ef_loc * self._proc_interval)
                
        elif(xti == 2):
            if(gti != 0 and gti != agent_id and xt_gti == 1 and gt_gti == agent_id):
                consumption += ((fti * ef_loc * self._proc_interval) + (min(hti, ht_gti) * (ef_loc + e_tx_rx) * self._proc_interval))
            else:
                consumption += (fti * ef_loc * self._proc_interval)
            
        # TODO: test the new assumption for battery going to zero 
        # if(consumption > (eb_t + pv) or (eb_t + pv) == 0):
            # return 0
        
        available_energy = (eb_t + pv)
        
        if(available_energy == 0):
            return 0
        
        if(consumption > available_energy):
            return 0

        # return (consumption/(available_energy))

        return consumption/available_energy
                
        # return 1 - (consumption/available_energy)

        # if(consumption == 0):
        #     return 0

        # return 1 + (((eb_t + pv) - consumption) / ((eb_t + pv) + consumption))        
        # return (((consumption + (eb_t + pv)) / ((eb_t + pv) - consumption))) / 100
        # return ((eb_t + pv)/consumption)
        # return 1 - (consumption/(eb_t + pv))
        
        
        # if(xti == 0 and ((fti * ef_loc * self._proc_interval) < (eb_t + pv))):
        #     consumption += (fti * ef_loc * self._proc_interval)
        #     # reward = min((fti * ef_loc * self._proc_interval)/(eb_t + pv), 1)
            
        # elif(xti == 1 and (((fti * ef_loc * self._proc_interval) + (hti * e_tx_rx * self._proc_interval)) < (eb_t + pv))):
        #     consumption += ((fti * ef_loc * self._proc_interval) + (hti * e_tx_rx * self._proc_interval))
        #     # reward =  min(1, ((fti * ef_loc * self._proc_interval) + (hti * e_tx_rx * self._proc_interval))/(eb_t + pv))
        
        # elif(xti == 2 and (((fti * ef_loc * self._proc_interval) + ((hti * (ef_loc + e_tx_rx) * self._proc_interval))) < (eb_t + pv))):
        #     consumption += ((fti * ef_loc * self._proc_interval) + (hti * (ef_loc + e_tx_rx) * self._proc_interval))
        #     # reward = min(1, (((fti + hti) * ef_loc * self._proc_interval) + ((hti * (ef_loc + e_tx_rx) * self._proc_interval)))/(eb_t + pv))
                
        # return reward
        
        
        # if(xti == 0 and ((fti * ef_loc * self._proc_interval) < (eb_t + pv))):
        #     return 1
        # elif(xti == 1 and (((fti * ef_loc * self._proc_interval) + (hti * e_tx_rx * self._proc_interval)) < (eb_t + pv))):
        #     return 1
        # elif(xti == 2 and (((fti * ef_loc * self._proc_interval) + ((hti * (ef_loc + e_tx_rx) * self._proc_interval))) < (eb_t + pv))):
        #     return 1
        
        # return 0
        
    # function for evaluating correctness of cooperations between nodes
    def calculate_reward_cooperation(self, agent_id, xti, gti, hti, xt_gti, gt_gti, ht_gti):
        # if(xti == 0 and gti == 0 and hti == 0):
        #     return 1
        
        if(xti == 0 and gti == 0 and hti == 0):
            return 1
        
        if(xti == 1 and xt_gti == 2 and gt_gti == agent_id and hti > 0 and ht_gti > 0):
            if(hti == 0 or ht_gti == 0):
                return 0
            return min(hti / ht_gti, ht_gti / hti)
        elif(xti == 2 and xt_gti == 1 and gt_gti == agent_id and hti > 0 and ht_gti > 0):
            if hti == 0 or ht_gti == 0:
                return 0
            return min(hti / ht_gti, ht_gti / hti)

        return 0
        
        '''
        elif(xti == 1 and gti != 0 and gti != agent_id and xt_gti == 2 and gt_gti == agent_id):
            if hti == 0 or ht_gti == 0:
                return 0
            
            match = min(hti / ht_gti, ht_gti / hti)
            off_ratio = min(hti, ht_gti) / self._processing_rate
            # return min(hti / ht_gti, ht_gti / hti)
            
            # return match * off_ratio
        
            return match + off_ratio
        
        elif(xti == 2 and gti != 0 and gti != agent_id and xt_gti == 1 and gt_gti == agent_id):
            if hti == 0 or ht_gti == 0:
                return 0
        
            match = min(hti / ht_gti, ht_gti / hti)
            off_ratio = min(hti, ht_gti) / self._processing_rate
            # # return min(hti / ht_gti, ht_gti / hti)
            
            return match + off_ratio
            # return 1
        
        
        return 0
        '''
    
    # function for evaluating the management of backlog
    def calculate_reward_backlog(self, agent_id, level, fti, hti):
        qty = self.backlogs[agent_id]
        max_storage = self._processing_rate * self._proc_interval * 10

        # max_storage = self._processing_rate * self._proc_interval * (3600 * 2 / self._proc_interval)


        # max_storage = self._processing_rate * 3600 * 12
        
        # return abs((1 - qty/max_storage))
        
        # max_storage = self._processing_rate * self._proc_interval * (3600 / self._proc_interval)
        
        # return 1 - qty/max_storage
        # if(self.backlogs[agent_id] == 0):
        #     return 1
        # else:
        #     return ((fti + hti) * self._proc_interval) / self.backlogs[agent_id]

        if(qty < max_storage):
            return (1 - (qty / max_storage))
        
        return 0
        
        # return (1 - qty/max_storage)
        
        # if(qty == 0):
        #     return 1
        # elif (qty <= max_storage):
        #     return 1 - (qty / max_storage)
        # else:
        #     return 0
        
        # return 1 - (self.backlogs[agent_id]/((self._processing_rate * 3600)))
            
    def calculate_reward(self, agent_id):
        fti = self.actions[agent_id][0]
        xti = self.actions[agent_id][1]
        gti = self.actions[agent_id][2]
        hti = self.actions[agent_id][3]
        
        # print(f"agent_id: {agent_id} - fti: {fti} - xti: {xti} - gti: {gti} - hti: {hti}")
        
        ft_gti = self.actions[gti][0]
        xt_gti = self.actions[gti][1]
        gt_gti = self.actions[gti][2]
        ht_gti = self.actions[gti][3]
        
        idx = (self.episode * self.max_steps) + self.timestep
        # print(idx)
        irradiance = self.irradiance_arrays[agent_id][idx]
        self.irradiance_level[agent_id] = self.irradiance_arrays[agent_id][idx] / self.max_irrad
        
        # if(fti + hti > self._processing_rate):
        #     k = self._processing_rate / (fti + hti)
        #     fti *= k
        #     hti *= k
        
        # # hti = processing_rate - fti
        
        # # if(hti > fti):
        # #     fti = processing_rate - hti
        # # else:
        # #     hti = processing_rate - fti
        
        # if(ft_gti + ht_gti > self._processing_rate):
        #     k = self._processing_rate / (ft_gti + ht_gti)
        #     ft_gti *= k
        #     ht_gti *= k
        #     # ht_gti = processing_rate - ft_gti

        #     # if(ht_gti > ft_gti):
        #     #     ft_gti = processing_rate - ht_gti
        #     # else:
        #     #     ht_gti = processing_rate - ft_gti
            

        # backlog = self.backlogs[agent_id]
        # panel_energy = irradiance * self.panel_surfaces[agent_id] * self.panel_efficiency * self._proc_interval
        # actual_battery = self.battery_capacities[agent_id] + panel_energy
        
        # processable = max(min(backlog, int((actual_battery - self.e_idle) / self.e_frame), self._processing_rate * self._proc_interval), 0)    
        # # processed = min(processable, fti * proc_interval)
        
        # # needed_energy = processed * e_frame + e_idle
        # # needed_energy = (fti * proc_interval * e_frame) + e_idle
        
        # local_reward = 0
        # offloading_reward = 0

        # if(processable > 0):
        #     processed = min(fti * self._proc_interval, processable)
            
        #     needed_energy = (processed * self.e_frame) + self.e_idle
            
        #     if(actual_battery > needed_energy and processed > 0):
        #         backlog = max(backlog - processed, 0)
        #         actual_battery = max(actual_battery - needed_energy, 0)
        #         if(backlog > 0):
                    
        #             if(fti == 0):
        #                 input(f"fti: {fti} -> {fti * self._proc_interval} - hti: {hti} -> {hti * self._proc_interval} - processable: {processable} - procesed: {processed}") 
        #             offloading_reward = (processed/(fti * self._proc_interval)) * (actual_battery/self.battery_capacities[agent_id]) * (processed / backlog)
        #         else:
        #             local_reward = (processed / (fti * self._proc_interval)) * (actual_battery / self.battery_capacities[agent_id])
                
        #     else:
        #         actual_battery -= self.e_idle
        #         local_reward = 0
        
        return jrf.jit_calculate_reward(fti,
                                   xti,
                                   gti,
                                   hti,
                                   ft_gti,
                                   xt_gti,
                                   gt_gti,
                                   ht_gti,
                                   irradiance,
                                   self.panel_surfaces[agent_id],
                                   self.panel_efficiency,
                                   self.backlogs[agent_id],
                                   self.e_idle,
                                   self.e_frame,
                                   self.e_tx_rx,
                                   self.battery_energies[agent_id],
                                   self.battery_capacities[agent_id],
                                   self._processing_rate,
                                   self._proc_interval,
                                   agent_id,
                                   self._w
                                   )
        
        
            
    def calculate_reward_non_opt(self, agent_id):
        # print(self.actions[agent_id])
        
        fti = self.actions[agent_id][0]
        xti = self.actions[agent_id][1]
        gti = self.actions[agent_id][2]
        hti = self.actions[agent_id][3]
        
        # print(f"agent_id: {agent_id} - fti: {fti} - xti: {xti} - gti: {gti} - hti: {hti}")
        
        ft_gti = self.actions[gti][0]
        xt_gti = self.actions[gti][1]
        gt_gti = self.actions[gti][2]
        ht_gti = self.actions[gti][3]
        
        if((fti + hti) > self._processing_rate):
            hti = 0
            
            # return 0
            # return self._processing_rate - (fti + hti)
            # fti = self._processing_rate
            # hti = 0
                
        if((ft_gti + ht_gti) > self._processing_rate):
            # ft_gti = self._processing_rate
            ht_gti = 0
        
        idx = (self.episode * self.max_steps) + self.timestep
        # print(idx)
        self.irradiance_level[agent_id] = self.irradiance_data[agent_id].iloc[idx]['ghi'] / self.max_irrad
        
        # --- NEW REWARD FUNCTION ---
        panel_energy = self.irradiance_level[agent_id] * self.max_irrad * self.panel_surfaces[agent_id] * self._proc_interval * self.panel_efficiency
        actual_battery = self.battery_energies[agent_id] + panel_energy
        needed_energy = (fti * self._proc_interval * self.e_frame) + self.e_idle
        backlog = self.backlogs[agent_id]
        
        processable = min(backlog, int((actual_battery - self.e_idle) / self.e_frame), self._processing_rate * self._proc_interval)
        
        if(actual_battery <= needed_energy):
            # backlog -= processable
            # actual_battery = 0.0
            return -1
        
        processed = min(processable, fti * self._proc_interval)
        local_reward = 0
        
        actual_battery = max(actual_battery - needed_energy, 0)
        
        # state update
        # self.update_state(agent_id, panel_energy, needed_energy, processed)
        
        if(processable > 0):
            # if(backlog > 0):
            local_reward = (processed / processable) * (actual_battery / self.battery_capacities[agent_id]) * (processed / backlog)
            backlog = max(backlog - processed, 0)
        else:
            return -1
        
        # else:
        #     print(f"agent {agent_id} -> processed/processable: {processed} / {processable}")
        # return local_reward
        
        # print(f"agent {agent_id} LOCAL -> processed/processable: {processed} / {processable}")
        
        # end local computation
        
        # ---------------------------------------------------
        
        # begin of offloading computation
        
        if(xti == 0):
            return local_reward
        
        remaining_framerate = self._processing_rate - fti
        if(remaining_framerate < 0):
            return local_reward
        
        offloading_reward = 0
        
        if(xti == 1 and gti != agent_id and hti > 0 and xt_gti == 2 and gt_gti == agent_id and ht_gti > 0):
            ht = min(hti, ht_gti)
            needed_energy = ht * self._proc_interval * self.e_tx_rx
            # processable = min(backlog, int(actual_battery / self.e_frame))
            processable = min(backlog, int(actual_battery / self.e_tx_rx), remaining_framerate * self._proc_interval)
            processed = min(processable, ht * self._proc_interval)
            
            if(actual_battery > needed_energy):
                if(processable > 0):
                    # offloading_reward = float(processed/processable) * (actual_battery / self.battery_capacities[agent_id])
                    offloading_reward = float(processed/processable) * (actual_battery / self.battery_capacities[agent_id]) * (processed / backlog)
                else:
                    offloading_reward = 0
            else:
                # return -1
                # offloading_reward = 0
                offloading_reward = -1
                    
        elif(xti == 2 and gti != agent_id and hti > 0 and xt_gti == 1 and gt_gti == agent_id and ht_gti > 0):
            ht = min(hti, ht_gti)
            needed_energy = ht * self._proc_interval * (self.e_tx_rx + self.e_frame)
            # processable = min(backlog, int(actual_battery / self.e_frame))
            processable = min(backlog, int(actual_battery / (self.e_frame + self.e_tx_rx)))
            processed = min(processable, ht * self._proc_interval)
            
            if(actual_battery > needed_energy):
                if(processable > 0):
                    # offloading_reward = float(processed/processable) * (actual_battery / self.battery_capacities[agent_id])
                    offloading_reward = float(processed/processable) * (actual_battery / self.battery_capacities[agent_id]) * (processed / backlog)
                else:
                    offloading_reward = 0
            else:
                # return -1
                # offloading_reward = 0
                offloading_reward = -1

        
        # print(f"agent {agent_id} OFF {xti} -> processed/processable: {processed} / {processable}")
        # input()
        
        return (local_reward + offloading_reward)

        # --- OLD REWARD FUNCTION ---
        
        # reward_frames = self.calculate_reward_frames(agent_id, fti, xti, gti, hti, xt_gti, gt_gti, ht_gti)
        # reward_battery = self.calculate_reward_battery(
                            # agent_id,
                            # fti,
                            # xti,
                            # gti,
                            # hti,
                            # ft_gti,
                            # xt_gti,
                            # gt_gti,
                            # ht_gti,
                            # self.battery_energies[agent_id],
                            # self.e_frame,
                            # self.e_tx_rx,
                            # self.irradiance_level[agent_id] * self.max_irrad * self.panel_surfaces[agent_id] * self._proc_interval
                            # )
        
        # reward_backlog = self.calculate_reward_backlog(agent_id, self.states[agent_id][1], fti, hti)
        # reward_cooperation = self.calculate_reward_cooperation(agent_id, xti, gti, hti, xt_gti, gt_gti, ht_gti)
        
        # self.r_battery[agent_id] += reward_battery
        # self.r_frames[agent_id] += reward_frames
        # self.r_cooperation[agent_id] += reward_cooperation
        # self.r_backlog[agent_id] += reward_backlog

        # return reward_battery + reward_frames + reward_backlog + reward_cooperation        
        # return reward_frames * reward_battery * reward_backlog * reward_cooperation
        
    def update_state(self, agent_id, panel_energy, needed_energy, processed):
        '''
        state: [0 -> battery, 1-> backlog, 2-> timestep]
        '''
        
        # battery update
        self.battery_energies[agent_id] += (panel_energy - needed_energy)
        if(self.battery_energies[agent_id] < 0.0):
            self.battery_energies[agent_id] = 0.0
        
        # backlog upate
        self.backlogs[agent_id] -= processed
        if(self.backlogs[agent_id] < 0):
            self.backlogs[agent_id] = 0
        
        self.states[agent_id][0] = round(float(self.battery_energies[agent_id] / self.battery_capacities[agent_id]), 2)
        self.states[agent_id][1] = self.calculate_backlog_level(agent_id)
        self.states[agent_id][2] = round(float(self.timestep / self.max_steps), 4)
        
    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        
        # setting to 0 all training variables
        self.timestep = 0
        self.states = [[0.5, 0, 0.0] for i in range(0, self._num_agents)]
        self.actions = [[0.0, 0, 0.0, 0.0] for i in range(0, self._num_agents)]
        self.battery_energies = [(self.battery_capacities[i] * self.states[i][0]) for i in range(0, self._num_agents)]
        self.backlogs = [0 for i in range(0, self._num_agents)]

        self.fs = [0 for i in range(0, self._num_agents)]
        self.hs = [0 for i in range(0, self._num_agents)]
        self.hs_counter = [0 for i in range(0, self._num_agents)]

        observations = {
            a: (
                np.array([val for sublist in self.states for val in sublist], dtype=np.float32)
            )
            for a in self.agents
        }

        infos = {a: {} for a in self.agents}
        
        self.episode = 355

        return observations, infos
        
    def update_states(self):
        # for each agent, update its state on the basis of the actions it executes
        for agent_id in range(0, self._num_agents):

            fti = self.actions[agent_id][0]
            xti = self.actions[agent_id][1]
            gti = self.actions[agent_id][2]
            hti = self.actions[agent_id][3]
            
            ft_gti = self.actions[gti][0]
            xt_gti = self.actions[gti][1]
            gt_gti = self.actions[gti][2]
            ht_gti = self.actions[gti][3]

            idx = (self.episode * self.max_steps) + self.timestep
            # print(idx)
            self.irradiance_level[agent_id] = self.irradiance_arrays[agent_id][idx] / self.max_irrad
            panel_energy = self.irradiance_level[agent_id] * self.max_irrad * self.panel_surfaces[agent_id] * self._proc_interval * self.panel_efficiency

            if((fti + hti) > self._processing_rate):
                k = float(self._processing_rate / (fti + hti))
                fti *= k
                hti *= k
                
                # hti = self._processing_rate - fti

                # if(hti > fti):
                #     fti = self._processing_rate - hti
                # else:
                #     hti = self._processing_rate - fti
                
            if((ft_gti + ht_gti) > self._processing_rate):
                k = float(self._processing_rate / (ft_gti + ht_gti))
                ft_gti *= k
                ht_gti *= k

                # ht_gti = self._processing_rate - ft_gti

                # if(ht_gti > ft_gti):
                #     ft_gti = self._processing_rate - ht_gti
                # else:
                #     ht_gti = self._processing_rate - ft_gti
            
            # --- NEW REWARD FUNCTION ---
            
            local_processing = 0
            offloading_processing = 0
            
            actual_battery = self.battery_energies[agent_id] + panel_energy

            backlog = self.backlogs[agent_id]
            
            processable = max(min(backlog, int((actual_battery - self.e_idle) / self.e_frame), self._processing_rate * self._proc_interval), 0)
            if(processable > 0):
                processed = min(processable, fti * self._proc_interval)
                needed_energy = (processed * self.e_frame) + self.e_idle
               
                if(actual_battery > needed_energy):
                   self.backlogs[agent_id] = max(backlog - processed, 0)
                   actual_battery = max((actual_battery - needed_energy), 0)
                   local_processing = processed / self._proc_interval
                
                else:
                    actual_battery = max(actual_battery - self.e_idle, 0)
            else:
                actual_battery = max(actual_battery - self.e_idle, 0)
            
            # self.backlogs[agent_id] = backlog
            remaining_framerate = self._processing_rate - fti
            
            if(remaining_framerate > 0 and xti != 0):
                if(xti == 1 and gti != agent_id and hti > 0 and xt_gti == 2 and gt_gti == agent_id and ht_gti > 0):
                    ht = min(hti, ht_gti)
                    backlog = self.backlogs[agent_id]
                    
                    # needed_energy = ht * self._proc_interval * self.e_tx_rx
                    processable = max(min(backlog, int((actual_battery - self.e_idle) / self.e_tx_rx), remaining_framerate * self._proc_interval), 0)
                    if(processable > 0):
                        processed = min(ht * self._proc_interval, processable)
                        needed_energy = processed * self.e_tx_rx
                        
                        if(needed_energy <= actual_battery):
                            self.backlogs[agent_id] = max(backlog  - processed, 0)
                            actual_battery = max(actual_battery - needed_energy, 0)
                            offloading_processing = processed / self._proc_interval
                            self.hs_counter[agent_id] += 1
                
                elif(xti == 2 and gti != agent_id and hti > 0 and xt_gti == 1 and gt_gti == agent_id and ht_gti > 0):
                    ht = min(hti, ht_gti)
                    # needed_energy = ht * self._proc_interval * self.e_tx_rx
                    processable = max(min(backlog, int((actual_battery - self.e_idle) / (self.e_tx_rx + self.e_frame)), remaining_framerate * self._proc_interval), 0)
                    if(processable > 0):
                        processed = min(processable, ht * self._proc_interval)
                        needed_energy = processed * (self.e_tx_rx + self.e_frame)
                        
                        if(needed_energy <= actual_battery):
                            # backlog -= processed
                            actual_battery = max(actual_battery - needed_energy, 0)
                            offloading_processing = processed / self._proc_interval
                            self.hs_counter[agent_id] += 1

                    
            # processed = min(processable, fti * self._proc_interval)
            
            # needed_energy = (processed * self.e_frame) + self.e_idle 
            
            # input(f"agent: {agent_id} -> processable: {processable} - processed: {processed} - panel: {panel_energy} - actual: {actual_battery} - needed: {needed_energy}")
            # input(f"agent: {agent_id} -> panel: {panel_energy} - actual: {actual_battery} - needed: {needed_energy}")
            
            # local_processing = 0
            # offloading_processing = 0
            
            # if(actual_battery < needed_energy):
            #     # actual_battery = max(actual_battery - needed_energy, 0)

            #     if(actual_battery > self.e_idle):
            #         actual_battery -= self.e_idle
            #     else:
            #         actual_battery = 0
                
            # else:
                
            #     if(processable > 0):
            #         local_processing = processed / self._proc_interval
            #         self.backlogs[agent_id] = max(self.backlogs[agent_id] - processed, 0)
            #         actual_battery = actual_battery - needed_energy
            #         # self.battery_energies[agent_id] = min(max(actual_battery - needed_energy, 0), self.battery_capacities[agent_id])
                
            #         processed = 0
                        
            #         remaining_framerate = self._processing_rate - fti
                    
            #         if(remaining_framerate > 0):
            #             # actual_battery = max(actual_battery - needed_energy, 0)
                        
            #             if(xti == 1 and gti != agent_id and hti > 0 and xt_gti == 2 and gt_gti == agent_id and ht_gti > 0):
            #                 ht = min(hti, ht_gti)
            #                 # needed_energy = ht * self._proc_interval * self.e_tx_rx
            #                 processable = max(min(backlog, int(actual_battery / self.e_tx_rx), remaining_framerate * self._proc_interval), 0)
            #                 processed = min(processable, ht * self._proc_interval)

            #                 needed_energy = processed * self.e_tx_rx
                                    
            #                 if(actual_battery >= needed_energy):
            #                     if(processable > 0):
            #                         offloading_processing = processed / self._proc_interval
            #                         self.backlogs[agent_id] -= processed
            #                         actual_battery -= needed_energy
                            
            #             elif(xti == 2 and gti != agent_id and hti > 0 and xt_gti == 1 and gt_gti == agent_id and ht_gti > 0):
            #                 ht = min(hti, ht_gti)
            #                 # needed_energy = ht * self._proc_interval * (self.e_tx_rx + self.e_frame)
            #                 processable = max(min(backlog, int(actual_battery / (self.e_tx_rx + self.e_frame)), remaining_framerate * self._proc_interval), 0)
            #                 processed = min(processable, ht * self._proc_interval)
                            
            #                 needed_energy = processed * (self.e_tx_rx + self.e_frame)

            #                 if(actual_battery >= needed_energy):
            #                     if(processable > 0):
            #                         offloading_processing = processed / self._proc_interval
            #                         self.backlogs[gti] -= processed
            #                         actual_battery -= needed_energy

            self.fs[agent_id] += local_processing
            self.hs[agent_id] += offloading_processing
            
            self.battery_energies[agent_id] = min(actual_battery, self.battery_capacities[agent_id])
            
            self.states[agent_id][0] = round(self.battery_energies[agent_id] / self.battery_capacities[agent_id], 2)
            self.states[agent_id][1] = self.calculate_backlog_level(agent_id)
            self.states[agent_id][2] = round(self.timestep / self.max_steps, 4)

        
    # def update_states(self):
    #     # for each agent, update its state on the basis of the actions it executes
    #     for agent_id in range(0, self._num_agents):

    #         fti = self.actions[agent_id][0]
    #         xti = self.actions[agent_id][1]
    #         gti = self.actions[agent_id][2]
    #         hti = self.actions[agent_id][3]
            
    #         ft_gti = self.actions[gti][0]
    #         xt_gti = self.actions[gti][1]
    #         gt_gti = self.actions[gti][2]
    #         ht_gti = self.actions[gti][3]

    #         idx = (self.episode * self.max_steps) + self.timestep
    #         # print(idx)
    #         self.irradiance_level[agent_id] = self.irradiance_arrays[agent_id][idx] / self.max_irrad
    #         panel_energy = self.irradiance_level[agent_id] * self.max_irrad * self.panel_surfaces[agent_id] * self._proc_interval * self.panel_efficiency

    #         if(fti + hti > self._processing_rate):
    #             if(hti > fti):
    #                 fti = self._processing_rate - hti
    #             else:
    #                 hti = self._processing_rate - fti
                
    #         if(ft_gti + ht_gti > self._processing_rate):
    #             if(ht_gti > ft_gti):
    #                 ft_gti = self._processing_rate - ht_gti
    #             else:
    #                 ht_gti = self._processing_rate - ft_gti
            
    #         # print(f"\nagent {agent_id} BEFORE -> fti: {fti} - hti: {hti}")
            
    #             # self.fs[agent_id] += 0
    #             # self.hs[agent_id] += 0
                
    #             # self.battery_energies[agent_id] = min(max((self.battery_energies[agent_id] - self.e_idle + panel_energy), 0) , self.battery_capacities[agent_id])
    #             # self.states[agent_id][0] = round(self.battery_energies[agent_id] / self.battery_capacities[agent_id], 2)
    #             # self.states[agent_id][1] = self.calculate_backlog_level(agent_id)
    #             # self.states[agent_id][2] = round(self.timestep / self.max_steps, 4)
                
    #             # continue
                
    #         # if((fti + hti) > self._processing_rate):
    #         #     hti = self._processing_rate - fti
    #         # if((ft_gti + ht_gti) > self._processing_rate):
    #         #     ht_gti = self._processing_rate - ft_gti

    #         # input(f"agent {agent_id} AFTER -> fti: {fti} - hti: {hti}")

    #         # self.fs[agent_id] += fti
    #         # self.hs[agent_id] += hti
            
    #         # --- NEW REWARD FUNCTION ---
    #         # actual_battery = min(self.battery_energies[agent_id] + panel_energy, self.battery_capacities[agent_id])
    #         actual_battery = self.battery_energies[agent_id] + panel_energy

    #         backlog = self.backlogs[agent_id]
            
    #         processable = min(backlog, int((actual_battery - self.e_idle) / self.e_frame), self._processing_rate * self._proc_interval)
    #         processed = min(processable, fti * self._proc_interval)
            
    #         needed_energy = (processed * self.e_frame) + self.e_idle 
    #         # needed_energy = (fti * self._proc_interval * self.e_frame) + self.e_idle
            
    #         # if(actual_battery > self.battery_capacities[agent_id]):
    #         #     print(f"battery overflow: {actual_battery} / {self.battery_capacities[agent_id]}")
                
    #         # input(f"agent {agent_id} -> battery: {self.battery_energies[agent_id]} / {self.battery_capacities[agent_id]}")
    #         # self.fs[agent_id] += processed / self._proc_interval
    #         local_processing = 0
    #         offloading_processing = 0
            
    #         if(actual_battery < needed_energy):
    #             # actual_battery = max(actual_battery - needed_energy, 0)

    #             if(actual_battery > self.e_idle):
    #                 actual_battery -= self.e_idle
    #             else:
    #                 actual_battery = 0
                
    #             # self.backlogs[agent_id] = max(self.backlogs[agent_id] - processed, 0)
    #             # actual_battery = 0 # it consumes all the available energy!!!
    #             # self.battery_energies[agent_id] = 0.0
    #             # self.fs[agent_id] += 0
                
    #         else:
    #             # self.fs[agent_id] += processed_temp / self._proc_interval
    #             # print(f"fs: {processed_temp / self._proc_interval}")
    #             # processed = min(processable, fti * self._proc_interval)
                
    #             if(processable > 0):
    #                 local_processing = processed / self._proc_interval
    #                 self.backlogs[agent_id] = max(self.backlogs[agent_id] - processed, 0)
    #                 actual_battery = actual_battery - needed_energy
    #                 # self.battery_energies[agent_id] = min(max(actual_battery - needed_energy, 0), self.battery_capacities[agent_id])
                
    #             processed = 0
    #             # if(xti == 0 or (fti + hti) > self._processing_rate):
    #             #     actual_battery = max(actual_battery - needed_energy, 0) # redundant
    #                 # self.battery_energies[agent_id] = min(max(actual_battery - needed_energy, 0), self.battery_capacities[agent_id])
    #             # if(not(xti == 0 or (fti + hti) > self._processing_rate)):
    #             #     remaining_framerate = self._processing_rate - fti
    #             #     # if(remaining_framerate < 0):
    #             #     #     continue
                    
    #             remaining_framerate = self._processing_rate - fti
                
    #             if(remaining_framerate > 0):
    #                 # actual_battery = max(actual_battery - needed_energy, 0)
                    
    #                 if(xti == 1 and gti != agent_id and hti > 0 and xt_gti == 2 and gt_gti == agent_id and ht_gti > 0):
    #                     ht = min(hti, ht_gti)
    #                     # needed_energy = ht * self._proc_interval * self.e_tx_rx
    #                     processable = min(backlog, int(actual_battery / self.e_tx_rx), remaining_framerate * self._proc_interval)
    #                     processed = min(processable, ht * self._proc_interval)

    #                     needed_energy = processed * self.e_tx_rx
                                
    #                     # self.backlogs[agent_id] -= processed
    #                     # actual_battery = max(actual_battery - needed_energy, 0)                            
                                                                    
    #                     if(actual_battery >= needed_energy):
    #                         if(processable > 0):
    #                             offloading_processing = processed / self._proc_interval
    #                             self.backlogs[agent_id] -= processed
    #                             actual_battery -= needed_energy
    #                             # actual_battery = max(actual_battery - needed_energy, 0)
    #                             # actual_battery -= ht * self.e_tx_rx
    #                     # else:
    #                     #     # actual_battery = max(actual_battery - needed_energy, 0)
    #                     #     if(actual_battery > self.e_idle):
    #                     #         actual_battery -= self.e_idle
    #                     #     else:
    #                     #         actual_battery = 0
                            
    #                     # self.battery_energies[agent_id] = actual_battery
                        
    #                 elif(xti == 2 and gti != agent_id and hti > 0 and xt_gti == 1 and gt_gti == agent_id and ht_gti > 0):
    #                     ht = min(hti, ht_gti)
    #                     # needed_energy = ht * self._proc_interval * (self.e_tx_rx + self.e_frame)
    #                     processable = min(backlog, int(actual_battery / (self.e_tx_rx + self.e_frame)), remaining_framerate * self._proc_interval)
    #                     processed = min(processable, ht * self._proc_interval)
                        
    #                     needed_energy = processed * (self.e_tx_rx + self.e_frame)

    #                     # self.backlogs[gti] -= processed
    #                     # actual_battery = max(actual_battery - needed_energy, 0)

    #                     if(actual_battery >= needed_energy):
    #                         if(processable > 0):
    #                             offloading_processing = processed / self._proc_interval
    #                             self.backlogs[gti] -= processed
    #                             actual_battery -= needed_energy
    #                             # actual_battery = max(actual_battery - needed_energy, 0)
    #                             # actual_battery -= ht * self.e_tx_rx
    #                     # else:
    #                     #     # actual_battery = max(actual_battery - needed_energy, 0)
    #                     #     if(actual_battery > self.e_idle):
    #                     #         actual_battery -= self.e_idle
    #                     #     else:
    #                     #         actual_battery = 0
                        
    #                     # if(actual_battery >= needed_energy):
    #                     #     if(processable > 0):
    #                     #         self.backlogs[gti] -= processed
    #                     #         actual_battery = max(actual_battery - needed_energy, 0)
    #                     # else:
    #                     #     actual_battery = 0
    #                     # self.battery_energies[agent_id] = actual_battery

    #                 # else:

    #             # else:
    #             #     self.battery_energies[agent_id] = min(actual_battery, self.battery_capacities[agent_id])

    #             # self.hs[agent_id] += processed / self._proc_interval
    #             # print(f"hs: {processed / self._proc_interval}")


    #         self.fs[agent_id] += local_processing
    #         self.hs[agent_id] += offloading_processing
            
    #         self.battery_energies[agent_id] = min(actual_battery, self.battery_capacities[agent_id])
            
    #         self.states[agent_id][0] = round(self.battery_energies[agent_id] / self.battery_capacities[agent_id], 2)
    #         self.states[agent_id][1] = self.calculate_backlog_level(agent_id)
    #         self.states[agent_id][2] = round(self.timestep / self.max_steps, 4)

        # return
        # # frames_arrived = self._arrival_rate * self._proc_interval
        # # self.backlogs[agent_id] += frames_arrived
        
        # idx = self.episode * self.max_steps + self.timestep
        # self.irradiance_level[agent_id] = round(self.irradiance_data[agent_id].iloc[idx]['ghi'] / self.max_irrad, 5)
        # panel_energy = self.irradiance_level[agent_id] * self.max_irrad * self.panel_efficiency * self.panel_surfaces[agent_id] * self._proc_interval        
        
        # # energy of local computation
        # comp_energy = fti * self._proc_interval * self.e_frame + self.e_idle
                    
        # # if it accepts the load of another node, compute the auxiliary effort
        # if(xti == 2 and xt_gti == 1 and gti != 0 and gti != agent_id and gt_gti == agent_id):
        #     offload_frames = min(ht_gti, hti) * self._proc_interval
        #     comp_energy += (offload_frames * self.e_frame + offload_frames * self.e_tx_rx)
        # # if it needs to send the load to another node, add energy needed for data exchange
        # elif(xti == 1 and xt_gti == 2 and gti != 0 and gti != agent_id and gt_gti == agent_id):
        #     comp_energy += min(ht_gti, hti) * self.e_tx_rx
        
        # available_energy = self.battery_energies[agent_id] + panel_energy
        
        # # if computational energy is higher than the available one, drop to zero otherwise update battery level directly on state structure
        # if(comp_energy > available_energy):
        #     # find remaining energy excluding the idle needed
        #     # !!! usable_energy = max(0, available_energy - self.e_idle)
        #     usable_energy = available_energy
            
        #     # computing the processable frames and the processed ones
        #     frames_processable = int(usable_energy / self.e_frame)
        #     # frames_processable = int(usable_energy / self.e_frame)
            
        #     frames_processed = min(frames_processable, self.backlogs[agent_id])
            
        #     self.battery_energies[agent_id] = 0.0
        #     self.states[agent_id][0] = 0.0
        #     self.backlogs[agent_id] -= frames_processed

        # else:
        #     self.battery_energies[agent_id] = available_energy - comp_energy
        #     self.battery_energies[agent_id] = min(self.battery_energies[agent_id], self.battery_capacities[agent_id])
            
        #     self.states[agent_id][0] = round(self.battery_energies[agent_id] / self.battery_capacities[agent_id], 2)
        #     # self.backlogs[agent_id] -= fti * self._proc_interval
            
        #     frames_to_process = fti * self._proc_interval
        #     frames_processed = min(frames_to_process, self.backlogs[agent_id])
        #     self.backlogs[agent_id] -= frames_processed

        # self.backlogs[agent_id] = max(0, self.backlogs[agent_id])                
        
        # self.states[agent_id][1] = self.calculate_backlog_level(agent_id)
        # self.states[agent_id][2] = round(self.timestep / self.max_steps, 4) 
        
        # # print(f"agent: {agent_id} -> battery: [{self.battery_energies[agent_id]} - {self.states[agent_id][0]}], panel_power: {self.panel_efficiency * self.panel_surfaces[agent_id] * self.irradiance_level[agent_id] * self.max_irrad}, irradiance: {self.irradiance_level[agent_id] * self.max_irrad}")

    def step(self, actions):
        # manual copy of actions inside internal actions variable
        for i in range(0, self._num_agents):
            for j in range(0, len(actions[i])):
                self.actions[i][j] = actions[i][j]

        # updating backlogs with arriving frames for each agent
        for agent_id in range(0, self._num_agents):
            frames_arrived = self._arrival_rate * self._proc_interval
            self.backlogs[agent_id] += frames_arrived

        # for each agent is returned the reward according the reward function defined a priori        
        rewards = {a: self.calculate_reward(a) for a in self.agents}
               
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        
        if(self.timestep == (self.max_steps - 1)):
            self.episode = 355
            truncations = {a: True for a in self.agents}
        
        self.timestep += 1
        
        # update of states after receiving all actions
        self.update_states()
        
        # observations structure is a dictionary with keys the indeces of agents
        observations = {
            a: (
                np.array([val for sublist in self.states for val in sublist], dtype=np.float32)
            )
            for a in self.agents
        }
        
        infos = {a: {} for a in self.agents}
        
        if any(terminations.values()) or all(truncations.values()):
            self.agents = []
        
        return observations, rewards, terminations, truncations, infos

    def render(self):
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]
    
    def observe(self, agent):
        return np.array(self.observations[agent])
