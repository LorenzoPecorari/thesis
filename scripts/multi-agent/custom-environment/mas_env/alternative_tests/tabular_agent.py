import numpy as np
import matplotlib.pyplot as plt
import random

class TabularAgent:
    def __init__(self,
                 agent_id,
                 battery_capacity,
                 storage_capacity,
                 power_idle,
                 power_max,
                 delta_time,
                 proc_interval,
                 pv_area,
                 max_fps,
                 arrival_rate,
                 num_agents,
                 battery_bins,
                 time_bins,
                 alpha,
                 gamma,
                 eps_min,
                 eps_dec,
                 eps_init,
                 episodes
                 ):
        
        self.agent_id = agent_id
        
        self.battery_capacity = battery_capacity
        self.storage_capacity = storage_capacity
        
        self.power_idle = power_idle
        self.power_max = power_max
        
        self.max_fps = max_fps
        self.arrival_rate = arrival_rate
        
        self.episodes = episodes
        self.num_agents = num_agents
        
        self.battery_bins = battery_bins
        self.time_bins = time_bins
        self.alpha = alpha
        self.gamma = gamma
        self.eps_min = eps_min
        self.eps = eps_init
        self.eps_dec = eps_dec

        action_dims = [self.max_fps + 1, 3, self.num_agents, self.max_fps + 1]
        state_dims = []
        
        for i in range(self.num_agents):
            state_dims.append(battery_bins)
            state_dims.append(4)
            state_dims.append(time_bins)

        total_dims = tuple(state_dims + action_dims)
        
        # |Q| = X_i->n(|possible_battery_levels|) x X_i->n(|possible_episode_completion_levels|) x |max fps + 1| x |3 offloading levels| x |max fps + 1| x |max fps + 1|
        self.table = np.zeros(total_dims)

    def state_discretization(self, state):
        indeces = []
        
        # print(f"state: {state}")
        # print("state [discretization]:", state)
        
        for agent in range(0, self.num_agents):
            # print(state[2*agent], state[3*agent+1], state[2*agent+2])
            battery_idx = int(min(state[3*agent] * self.battery_bins, self.battery_bins - 1))
            backlog_idx = min(int(state[3*agent + 1]), 3)
            time_idx = int(min(state[3*agent + 2] * self.time_bins, self.time_bins - 1))            
            
            indeces.append(battery_idx)
            indeces.append(backlog_idx)
            indeces.append(time_idx)
            # print(f"agent {agent} -> battery: {battery_idx}, backlog: {backlog_idx}, time_idx: {time_idx}")

            # input()

        # print(indeces)

        return tuple(indeces)

    def choice_action(self, state):
        best_action = [0, 0, 0, 0]

        if(np.random.random() < self.eps):
            best_action = [random.randint(0, self.max_fps),
                    random.randint(0, 2),
                    random.randint(0, self.num_agents - 1),
                    random.randint(0, self.max_fps)
                    ]
        else:
            state_idx = self.state_discretization(state)
            # q_values = self.table
            # for elem in state_idx:
            #     q_values = q_values[elem]

            q_values = self.table[*state_idx]

            best_value = -1
            best_action = [0, 0, 0, 0]
            
            for f in range(0, self.max_fps+1):
                for x in range(0, 3):
                    for g in range(0, self.num_agents):
                        for h in range(0, self.max_fps + 1):
                            if(q_values[f][x][g][h] > best_value):
                                best_value = q_values[f][x][g][h]
                                best_action = [f, x, g, h]
            
        fti = best_action[0]
        hti = best_action[3]
        
        # if((fti + hti) > self.max_fps):
        #     k = self.max_fps / (fti + hti)
        #     best_action[0] = int(best_action[0] * k)
        #     best_action[3] = int(best_action[3] * k)
        
        # input(f"fti: {fti} - hti: {hti} - action_local: {best_action[0]} - action_off: {best_action[3]}")
        
        return np.array(best_action)
        
    def update_table(self, state, next_state, action, reward):
        state_idx = self.state_discretization(state)
        next_state_idx = self.state_discretization(next_state)
        
        f, x, g, h = action
        
        q_value = self.table[*state_idx, f, x, g, h]
        # for elem in state_idx:
        #     q_values = q_values[elem]
            
        # for a in action:
        #     q_values = q_values[a]
    
        next_q_value = self.table[*next_state_idx]
        # for elem in next_state_idx:
        #     next_q_values = next_q_values[elem]
        
        best_next_value = np.max(next_q_value)
        # for f in range(0, self.max_fps+1):
        #         for x in range(0, 3):
        #             for g in range(0, self.num_agents):
        #                 for h in range(0, self.max_fps + 1):
        #                     if(next_q_value[f][x][g][h] > best_next_value):
        #                         best_next_value = next_q_value[f][x][g][h]
                                
        self.table[*state_idx, f, x, g, h] = ((1 - self.alpha) * q_value) + (self.alpha * (reward + (self.gamma * best_next_value)))

    def update_epsilon(self):
        if(self.eps > self.eps_min):
            self.eps *= self.eps_dec