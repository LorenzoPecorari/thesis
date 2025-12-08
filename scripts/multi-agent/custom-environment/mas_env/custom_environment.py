from pettingzoo import ParallelEnv
from gymnasium import spaces
import matplotlib.pyplot as plt
from copy import copy

import functools
import numpy as np
import random
import interpol as ip


class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, num_agents, irradiance_datapaths, delta_time, proc_interval, proc_rate, arr_rate, batteries, panel_surfaces, power_idle, power_max):
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
        for filepath in irradiance_datapaths:
            self.irradiance_data.append(ip.interpolate(filepath, delta_time, proc_interval))
        
        for elem in self.irradiance_data:
            print(len(elem))
        
        self.irradiance_level = [0.0 for i in range(0, self._num_agents)]
        
        self.battery_capacities = [(battery*3600) for battery in batteries]
        self.battery_energies = [0.0 for i in range(0, self._num_agents)]
        self.panel_surfaces = panel_surfaces
        
        self.e_idle = power_idle * self._proc_interval
        self.e_frame = (0.8 * (power_max - power_idle) * self._proc_interval) / proc_rate
        self.e_tx_rx = (0.2 * (power_max - power_idle) * self._proc_interval) / proc_rate
        
        self.backlogs = [0 for i in range(0, self._num_agents)]
        
        # state_i = (battery_level_i, daily_completion_i)
        self.states = [[0.0, 0, 0.0] for i in range(0, self._num_agents)]
        self.actions = [[0.0, 0, 0.0, 0.0] for i in range(0, self._num_agents)]
        self.rewards = [0 for i in range(0, self._num_agents)]
        
        # internal counters for episode compeltion 
        self.timestep = 0
        self.max_steps = 1440
        self.episode = 172
        
        try:
            self.max_steps = int(24 * 60 * 60 / proc_interval)
        except:
            self.max_steps = 1

        # observation_space: [b_1, t_1, b_2, t_2, ..., b_n, t_n]
        # for each agent there are two variables in the range [0.0, 1.0] where
            # b_i -> "battery level"
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
            agent: spaces.Box(low = 0.0, high = 1.0, shape = (self._num_agents * 3, ), dtype = np.float64) for agent in self.possible_agents
        }
        
        self.fs = [0 for i in range(0, self._num_agents)]
        self.hs = [0 for i in range(0, self._num_agents)]
    
    # function for retrieving level of backlog
    def calculate_backlog_level(self, agent_id):
        qty = self.backlogs[agent_id]
        max_storage = self._processing_rate * self._proc_interval * (3600 / self._proc_interval)
        
        if(qty == 0):
            return 0
        elif(qty > 0 and qty < int(max_storage / 3)):
            return 1
        elif(qty >= int(max_storage / 3) and qty < int((2/3) * max_storage)):
            return 2
        else:
            return 3

    # function for evaluating reward over frames management
    def calculate_reward_frames(self, fti, hti):
        # return min(1, (fti + hti)/self._processing_rate)
        
        if((fti + hti) <= self._processing_rate):
            return min(1, (fti + hti)/self._processing_rate)
        else:
            return 0
        
        # if((fti + hti) < self._processing_rate):
        #     return 1
        # else:
        #     return 0
        
    # function for evaluating reward over battery management
    def calculate_reward_battery(self, agent_id, fti, xti, hti, eb_t, ef_loc, e_tx_rx, pv):

        consumption = 0.0
        
        if(xti == 0 and ((fti * ef_loc * self._proc_interval) < (eb_t + pv))):
            consumption = (fti * ef_loc * self._proc_interval)
            reward = min((fti * ef_loc * self._proc_interval)/(eb_t + pv), 1)
            
        elif(xti == 1 and (((fti * ef_loc * self._proc_interval) + (hti * e_tx_rx * self._proc_interval)) < (eb_t + pv))):
            consumption = ((fti * ef_loc * self._proc_interval) + (hti * e_tx_rx * self._proc_interval))
            reward =  min(1, ((fti * ef_loc * self._proc_interval) + (hti * e_tx_rx * self._proc_interval))/(eb_t + pv))
        
        elif(xti == 2 and (((fti * ef_loc * self._proc_interval) + ((hti * (ef_loc + e_tx_rx) * self._proc_interval))) < (eb_t + pv))):
            consumption = ((fti * ef_loc * self._proc_interval) + (hti * (ef_loc + e_tx_rx) * self._proc_interval))
            reward = min(1, (((fti + hti) * ef_loc * self._proc_interval) + ((hti * (ef_loc + e_tx_rx) * self._proc_interval)))/(eb_t + pv))
        
        return reward
        
        # TODO: test the new assumption for battery going to zero 
        # if(((eb_t + pv) - consumption) < 0):
        #     return -1
        # return 0
        
        
        # if(xti == 0 and ((fti * ef_loc * self._proc_interval) < (eb_t + pv))):
        #     return 1
        # elif(xti == 1 and (((fti * ef_loc * self._proc_interval) + (hti * e_tx_rx * self._proc_interval)) < (eb_t + pv))):
        #     return 1
        # elif(xti == 2 and (((fti * ef_loc * self._proc_interval) + ((hti * (ef_loc + e_tx_rx) * self._proc_interval))) < (eb_t + pv))):
        #     return 1
        
        # return 0
        
    # function for evaluating correctness of cooperations between nodes
    def calculate_reward_cooperation(self, agent_id, xti, gti, xt_gti, gt_gti):
        if(xti == 0 and gti == 0):
            return 1
        elif(xti == 1 and gti != 0 and gti != agent_id and xt_gti == 2 and gt_gti == agent_id):
            return 1
        elif(xti == 2 and gti != 0 and gti != agent_id and xt_gti == 1 and gt_gti == agent_id):
            return 1
        
        return 0
    
    # function for evaluating the management of backlog
    def calculate_reward_backlog(self, level):
        return 1 - round(level/3, 3)
        
    def calculate_reward(self, agent_id):
        # print(self.actions[agent_id])
        
        fti = self.actions[agent_id][0]
        xti = self.actions[agent_id][1]
        gti = self.actions[agent_id][2]
        hti = self.actions[agent_id][3]
        
        # print(f"agent_id: {agent_id} - fti: {fti} - xti: {xti} - gti: {gti} - hti: {hti}")
        
        xt_gti = self.actions[gti][1]
        gt_gti = self.actions[gti][2]
        
        idx = (self.episode * self.max_steps) + self.timestep
        # print(idx)
        self.irradiance_level[agent_id] = round(self.irradiance_data[agent_id].iloc[idx]['ghi'] / self.max_irrad, 2)
        
        reward_frames = self.calculate_reward_frames(fti, hti)
        reward_battery = self.calculate_reward_battery(
                            agent_id,
                            fti,
                            self.battery_energies[agent_id],
                            hti,
                            self.battery_capacities[agent_id],
                            self.e_frame,
                            self.e_tx_rx,
                            self.irradiance_level[agent_id] * self.max_irrad * self.panel_surfaces[agent_id] * self._proc_interval
                            )
        
        reward_backlog = self.calculate_reward_backlog(self.states[agent_id][1])
        reward_cooperation = self.calculate_reward_cooperation(agent_id, xti, gti, xt_gti, gt_gti)

        return reward_frames + reward_battery + reward_backlog + reward_cooperation        
        # return reward_frames * reward_battery * reward_backlog * reward_cooperation
        
    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        
        # setting to 0 all training variables
        self.timestep = 0
        self.states = [[0.5, 0, 0.0] for i in range(0, self._num_agents)]
        self.actions = [[0.0, 0, 0.0, 0.0] for i in range(0, self._num_agents)]
        self.battery_energies = [(self.battery_capacities[i] / 2) for i in range(0, self._num_agents)]
        self.backlogs = [0 for i in range(0, self._num_agents)]

        observations = {
            a: (
                np.array([val for sublist in self.states for val in sublist], dtype=np.float64)
            )
            for a in self.agents
        }

        infos = {a: {} for a in self.agents}
        
        self.episode = 172

        return observations, infos
        
    def update_states(self):
        # for each agent, update its state on the basis of the actions it executes
        for agent_id in range(0, self._num_agents):

            fti = self.actions[agent_id][0]
            xti = self.actions[agent_id][1]
            gti = self.actions[agent_id][2]
            hti = self.actions[agent_id][3]
            
            xt_gti = self.actions[gti][1]
            gt_gti = self.actions[gti][2]
            ht_gti = self.actions[gti][3]

            self.fs[agent_id] += fti
            self.hs[agent_id] += hti

            frames_arrived = self._arrival_rate * self._proc_interval
            self.backlogs[agent_id] += frames_arrived
            
            idx = self.episode * self.max_steps + self.timestep
            self.irradiance_level[agent_id] = round(self.irradiance_data[agent_id].iloc[idx]['ghi'] / self.max_irrad, 5)
            panel_energy = self.irradiance_level[agent_id] * self.max_irrad * self.panel_efficiency * self.panel_surfaces[agent_id] * self._proc_interval        
            
            # energy of local computation
            comp_energy = fti * self._proc_interval * self.e_frame + self.e_idle
                      
            # if it accepts the load of another node, compute the auxiliary effort
            if(xti == 2 and xt_gti == 1 and gti != 0 and gti != agent_id and gt_gti == agent_id):
                offload_frames = min(ht_gti, hti) * self._proc_interval
                comp_energy += (offload_frames * self.e_frame + offload_frames * self.e_tx_rx)
            # if it needs to send the load to another node, add energy needed for data exchange
            elif(xti == 1 and xt_gti == 2 and gti != 0 and gti != agent_id and gt_gti == agent_id):
                comp_energy += min(ht_gti, hti) * self.e_tx_rx
            
            available_energy = self.battery_energies[agent_id] + panel_energy
            
            # if computational energy is higher than the available one, drop to zero otherwise update battery level directly on state structure
            if(comp_energy > available_energy):
                # find remaining energy excluding the idle needed
                usable_energy = max(0, available_energy - self.e_idle)
                
                # computing the processable frames and the processed ones
                frames_processable = int(usable_energy / self.e_frame)
                frames_processed = min(frames_processable, self.backlogs[agent_id])
                
                self.battery_energies[agent_id] = 0.0
                self.states[agent_id][0] = 0.0
  
            else:
                self.battery_energies[agent_id] = available_energy - comp_energy
                self.battery_energies[agent_id] = min(self.battery_energies[agent_id], self.battery_capacities[agent_id])
                
                self.states[agent_id][0] = round(self.battery_energies[agent_id] / self.battery_capacities[agent_id], 2)
                self.backlogs[agent_id] -= fti * self._proc_interval
                
                frames_to_process = fti * self._proc_interval
                frames_processed = min(frames_to_process, self.backlogs[agent_id])
                self.backlogs[agent_id] -= frames_processed

            self.backlogs[agent_id] = max(0, self.backlogs[agent_id])                
            
            self.states[agent_id][1] = self.calculate_backlog_level(agent_id)
            self.states[agent_id][2] = round(self.timestep / self.max_steps, 4) 
            
            # print(f"agent: {agent_id} -> battery: [{self.battery_energies[agent_id]} - {self.states[agent_id][0]}], panel_power: {self.panel_efficiency * self.panel_surfaces[agent_id] * self.irradiance_level[agent_id] * self.max_irrad}, irradiance: {self.irradiance_level[agent_id] * self.max_irrad}")

    def step(self, actions):
        # manual copy of actions inside internal actions variable
        for i in range(0, self._num_agents):
            for j in range(0, len(actions[i])):
                self.actions[i][j] = actions[i][j]

        # for each agent is returned the reward according the reward function defined a priori        
        rewards = {a: self.calculate_reward(a) for a in self.agents}
               
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        
        if(self.timestep == (self.max_steps - 1)):
            self.episode = 172
            truncations = {a: True for a in self.agents}
        
        self.timestep += 1
        
        # update of states after receiving all actions
        self.update_states()
        
        # observations structure is a dictionary with keys the indeces of agents
        observations = {
            a: (
                np.array([val for sublist in self.states for val in sublist], dtype=np.float64)
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
     
    