import numpy as np
import matplotlib.pyplot as plt
import random
import environment_simplified

class Agent:
    def __init__(self, 
                 datapath,
                 battery_capacity,
                 storage_capacity,
                 power_idle,
                 power_max,
                 delta_time,
                 proc_interval,
                 max_irradiation,
                 pv_efficiency,
                 pv_area,
                 fps,
                 seed,
                 battery_bins,
                 time_bins,
                 alpha,
                 gamma,
                 eps_min,
                 eps_dec,
                 eps_init,
                 episodes
                 ):
        
        self.env = environment_simplified.EnergyPVEnv(
            datapath,
            battery_capacity,
            storage_capacity,
            power_idle,
            power_max,
            delta_time,
            proc_interval,
            max_irradiation,
            pv_efficiency,
            pv_area,
            fps)

        # tecnical parameters
        self.fps = fps
        self.p_idle = power_idle
        self.p_max = power_max

        # hyperparameters
        self.seed = seed
        self.battery_bins = battery_bins
        self.time_bins = time_bins
        self.alpha = alpha
        self.gamma = gamma
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.eps = eps_init
        self.episodes = episodes
        self.battery_capacity = battery_capacity
        
        # 3d table, |battery_levls| x |time_levels| x |actions|        
        self.table = np.zeros((battery_bins, time_bins, (int(self.fps) + 1)))

    def choice_action(self, state, eps):
        b_idx, t_idx = self.state_discretization(state[0], state[1])
        # b_idx, t_idx = state[0], state[1]
        
        # print(f"Q drop: {self.table[b_idx, t_idx, 0]}, Q process: {self.table[b_idx, t_idx, 1]} - ", end="")
        if np.random.random() < eps:
            # print("RANDOM")
            return random.randint(0, int(self.fps))
        else:
            # print("DECIDED")
            return np.argmax(self.table[b_idx, t_idx])
        
    def state_discretization(self, b, t):
        battery_idx = int(b * self.battery_bins)
        time_idx = int(t * self.time_bins)

        battery_idx = min(self.battery_bins - 1, battery_idx)
        time_idx = min(self.time_bins - 1, time_idx)   

        return battery_idx, time_idx

    def update_eps(self):
        if self.eps > self.eps_min:
            self.eps = self.eps * self.eps_dec
        else:
            self.eps = self.eps_min

    def update_table(self, state, next_state, action, reward):
        b_idx, t_idx = self.state_discretization(state[0], state[1])
        next_b_idx, next_t_idx = self.state_discretization(next_state[0], next_state[1])
        
        # print(f"current_state: [{b_idx}, {t_idx}] - next_state: [{next_b_idx}, {next_t_idx}]")
        # print(f"Q(s', a'): {np.max(self.table[next_b_idx, next_t_idx])} - reward: {reward}")
        
        # print(f"\n State of Q({b_idx}, {t_idx}, {action}): {self.table[b_idx, t_idx, action]} -> ", end="")
        self.table[b_idx, t_idx, action] = (1 - self.alpha) * self.table[b_idx, t_idx, action] + (self.alpha * (reward + (self.gamma * np.max(self.table[next_b_idx, next_t_idx]))))
        # print(f"{self.table[b_idx, t_idx, action]}")
    
    def train(self):
        rewards = []
        cumulative_traces = []
        
        dropped_frames = []
        processed_frames = []
        storage = []
        stored = []
        daily_backlogs = []
        processed_stored_ratio = []

        battery = []
        battery_traces = []
        
        energy = []
        energy_traces = []
        
        irradiance = []
        daily_irradiance = []

        actions = []
        action_traces = []
        
        discharges = []
        
        for episode in range(self.episodes + 1):
            state, _ = self.env.reset(self.seed)
            partial_reward = 0
            battery_avg = 0
            avg_irrad = 0
            storage_temp = 0
            action_avg = 0
            discharge = 0
            
            action_trace = []        
            daily_energy = []
            energy.append(0.0)
            battery_trace = []            
        
            daily_backlog = []
        
            trace = []
            info = {}

            self.update_eps()            

            for j in range(self.env.max_steps):
                state = self.state_discretization(state[0], state[1])
                action = self.choice_action(state, self.eps)

                new_state, reward, terminated, truncated, info = self.env.step(action * 5)

                self.update_table(state, new_state, action, reward)
                partial_reward += reward
                state = new_state
                
                if(len(daily_irradiance) <= j):
                    daily_irradiance.append(info['irradiance'] * self.env.max_irrad)
                else:
                    daily_irradiance[j] += info['irradiance'] * self.env.max_irrad
                
                daily_energy.append(info['energy_consumption'])
                energy[episode] += info['energy_consumption']
                
                # if(reward < action * self.env.interval):
                #     action = int(reward / self.env.interval) 
                
                # battery.append(info['battery_level'] * self.battery_capacity)
                battery_avg += (info['battery_level'] * 100)
                avg_irrad += info['irradiance']
                storage_temp += info['storage_level']
                
                action_avg += (action)
                action_trace.append(action)

                # print(info['battery_level'])

                if(info['battery_level'] == 0.0):
                    discharge += 1
                    
                if(episode % int(self.episodes / 10) == 0):
                    daily_backlog.append(self.env.storage)
                    battery_trace.append(info['battery_level'] * 100)


                # input("Press enter to continue...")
            
            storage_temp /= self.env.max_steps
            stored.append(storage_temp)
            
            if(episode % int(self.episodes / 10) == 0):
                cumulative_traces.append(trace)
                battery_traces.append(battery_trace)
                action_traces.append(action_trace)
                energy_traces.append(daily_energy)
            
            # print(f"episode: {episode}/{self.episodes} - reward: {partial_reward} - eps: {self.eps}")
            print(f"episode: {episode}/{self.episodes} - reward: {round(partial_reward, 1)} - avg_batt: {round(battery_avg / self.env.max_steps, 2)} - eps: {self.eps}")
            # print(f"dropped: {info['frames_dropped']} - processed : {info['frames_processed']} - avg battery : {battery_avg / self.env.max_steps} - avg irradiance: {avg_irrad / self.env.max_steps}")
            # dropped_frames.append(info['frames_dropped'])
            
            battery.append(((battery_avg) / self.env.max_steps))
            irradiance.append(((avg_irrad ) / self.env.max_steps) * self.env.max_irrad)
            
            actions.append(action_avg / self.env.max_steps)
            processed_frames.append(info['frames_processed'])
            storage.append(info['storage_level'])
            
            try:
                processed_stored_ratio.append(info['frames_processed'] / info['storage_level'])
            except:
                processed_stored_ratio.append(0)
                
            discharges.append(discharge)
            
            # self.save_table(episode)
            
            rewards.append(partial_reward)
            # print(daily_backlog)
            
            if(len(daily_backlog) > 0):
                daily_backlogs.append(daily_backlog)
            # input("Press enter to continue...")
        
        self.plot_cumulative_trace(cumulative_traces)
        self.plot_daily_battery(battery_traces)
        self.plot_daily_action(action_traces)
        
        self.plot_battery(battery)
        
        # self.plot_daily_energy_consumption(energy_traces)
        
        self.plot_action(actions)
        self.plot_processed_stored_ratio(processed_stored_ratio)
        
        # self.plot_irradiance(irradiance)
        # self.plot_daily_irradiation(daily_irradiance)
        
        self.plot_battery_violations(discharges)
        self.plot_storage_daily(daily_backlogs)
        self.plot_storage(stored)
        
        return rewards, dropped_frames, processed_frames, battery, irradiance, storage

    ### daily metrics

    def plot_cumulative_trace(self, data):
        plt.title(f"B: {self.battery_capacity}, p_i: {self.p_idle}, p_F: {self.p_max}")
        plt.suptitle("Single agent - Cumulative rewards")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        
        for i in range(len(data)):
            plt.plot(data[i], label = f"{str(i * int(self.episodes/10))}-th episode")
        
        plt.grid()
        plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"cumulative_rewards_{self.battery_capacity}Wh_{self.fps}fps_qlt_{self.env.day}.pdf")
        plt.close()    
        
    def plot_storage(self, data):
        window = 10
        plt.title(f"B: {self.battery_capacity}, p_i: {self.p_idle}, p_F: {self.p_max}")
        plt.suptitle("Single agent - Average Backlog")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        
        plt.plot(range(window - 1, len(data)), np.convolve(data, np.ones(window)/window, mode='valid'), label = f"smooth", alpha = 1.0)
        
        plt.grid()
        plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"backlog_avg_{self.battery_capacity}Wh_{self.fps}fps_qlt_{self.env.day}.pdf")
        plt.close()    
            
    def plot_daily_battery(self, data):        
        plt.suptitle("Single agent - Daily battery")
        plt.title(f"B = {self.battery_capacity}, P_i = {self.p_idle}, P_f = {self.p_max}, fps = {self.env.fps}")

        plt.xlabel("Step")
        plt.ylabel("Battery")
        
        # plt.ylim(-0.5, 10)
        
        for i in range(len(data)):
            plt.plot(data[i], label = f"{str(i * int(self.episodes/10))}-th episode" )
        
        plt.grid()
        plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"battery_daily_{self.battery_capacity}Wh_{self.fps}fps_qlt_{self.env.day}.pdf")
        plt.close()  
        
    def plot_daily_action(self, data):
        window = 100        
        plt.suptitle("Single agent - Daily FPS")
        plt.title(f"B = {self.battery_capacity}, P_i = {self.p_idle}, P_f = {self.p_max}, max_fps = {self.env.fps}")

        plt.xlabel("Step")
        plt.ylabel("FPS")
        
        for i in range(len(data)):
            plt.plot(range(window - 1, len(data[i])), np.convolve(data[i], np.ones(window)/window, mode='valid'), label = f"{(i * int(self.episodes/10))}-th episode", alpha = 1.0)

        plt.grid()
        plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"action_daily_{self.battery_capacity}Wh_{self.fps}fps_qlt_{self.env.day}.pdf")
        plt.close()  
            
    def plot_daily_irradiation(self, data):
        plt.title("Daily average irradiance")
        
        window = 10
        n = len(data)
        for i in range(n):
            data[i] = data[i] / self.episodes
            
        plt.plot(data, label = "raw", alpha = 0.4)  
        plt.plot(range(window - 1, len(data)), np.convolve(data, np.ones(window) / window, mode='valid'), label = "smooth", alpha = 1.0)
        
        plt.grid()
        plt.legend()
        plt.savefig(f"daily_irradiance_{self.env.day}.pdf")
        plt.close()
    
    def plot_daily_energy_consumption(self, data):
        plt.title("Daily energy consumption")
        plt.xlabel("Steps")
        plt.ylabel("Energy (J)")
        
        window = 90
        n = len(data)
        
        for i in range(n):
            plt.plot(range(window - 1, len(data[i])), np.convolve(data[i], np.ones(window) / window, mode='valid'), label = f"{i * int(self.episodes/10)}-th episode", alpha = 1.0)
        
        plt.grid()
        # plt.legend()
        plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"daily_consumption_{self.env.day}.pdf")
        plt.close()
    
    def plot_battery_violations(self, data):
        window = 10        
        plt.suptitle("Single agent - Complete discharges")
        plt.title(f"B = {self.battery_capacity}, P_i = {self.p_idle}, P_f = {self.p_max}, max_fps = {self.env.fps}")

        plt.xlabel("Step")
        plt.ylabel("Discarges")
        
        # for i in range(len(data)):
        plt.plot(range(window - 1, len(data)), np.convolve(data, np.ones(window)/window, mode='valid'), label = f"smooth", alpha = 1.0)

        plt.grid()
        plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"discharges_{self.battery_capacity}Wh_{self.fps}fps_qlt_{self.env.day}.pdf")
        plt.close()
    
    ### training metrics

    def plot_rewards(self, data):
        window = 10
        plt.suptitle("Single agent - rewards")
        plt.title(f"B = {self.battery_capacity}, P_i = {self.p_idle}, P_f = {self.p_max}, fps = {self.env.fps}")
        
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.plot(range(window - 1, len(data)), np.convolve(data, np.ones(window)/window, mode='valid'), label = "smoothened", alpha = 1.0)
        plt.plot(data, label = "raw", alpha = 0.3)
        plt.grid()
        plt.legend()
        plt.savefig(f"rewards_plot_{self.battery_capacity}Wh_{self.fps}fps_qlt_{self.env.day}.pdf")
        plt.close()

    def plot_frames(self, dropped, processed):
        window = 10
        plt.suptitle("Single agent - frames")
        plt.title(f"B = {self.battery_capacity}, P_i = {self.p_idle}, P_f = {self.p_max}, fps = {self.env.fps}")
        plt.xlabel("Episodes")
        plt.ylabel("Frames")
        
        plt.plot(range(window - 1, len(dropped)), np.convolve(dropped, np.ones(window)/window, mode='valid'), label = "smoothened dropped", alpha = 1.0)
        plt.plot(dropped, label = "raw dropped", alpha = 0.3)
        plt.plot(range(window - 1, len(processed)), np.convolve(processed, np.ones(window)/window, mode='valid'), label = "smoothened processed", alpha = 1.0)
        plt.plot(processed, label = "raw processed", alpha = 0.3)
        
        plt.grid()
        plt.legend()
        plt.savefig(f"frames_plot_{self.battery_capacity}Wh_{self.fps}fps_qlt_{self.env.day}.pdf")
        plt.close()

    def plot_action(self, data):
        window = 10
        plt.suptitle("Single agent - Average fps")
        plt.title(f"B = {self.battery_capacity}, P_i = {self.p_idle}, P_f = {self.p_max}, max_fps = {self.env.fps}")
        plt.xlabel("Episodes")
        plt.ylabel("FPS")
        
        plt.plot(data, label = "raw fps", alpha = 0.3)
        plt.plot(range(window - 1, len(data)), np.convolve(data, np.ones(window)/window, mode='valid'), label = "smooth fps", alpha = 1.0)
        
        plt.grid()
        plt.legend()
        plt.savefig(f"action_plot_{self.battery_capacity}Wh_{self.fps}fps_qlt_{self.env.day}.pdf")
        plt.close()
        

    def plot_battery(self, data):
        
        # for elem in data:
        #     print(elem)
                    
        window = 10
        plt.suptitle("Single agent - battery level")
        plt.title(f"B = {self.battery_capacity}, P_i = {self.p_idle}, P_f = {self.p_max}, fps = {self.env.fps}")
        
        plt.xlabel("Episodes")
        plt.ylabel("Battery level")
        plt.plot(range(window - 1, len(data)), np.convolve(data, np.ones(window)/window, mode='valid'), label = "smoothened", alpha = 1.0)
        plt.plot(data, label = "raw", alpha = 0.3)
        plt.grid()
        plt.legend()
        plt.savefig(f"battery_plot_{self.battery_capacity}Wh_{self.fps}fps_qlt_{self.env.day}.pdf")
        plt.close()

    def plot_irradiance(self, data):
        avg = np.mean(data) * self.env.max_irrad
        window = 10
        plt.title("Average daily irradiance: " + str(round(avg, 2)) + " W/m^2")
        plt.xlabel("Episodes")
        plt.ylabel("Irradiance")
        plt.plot(range(window - 1, len(data)), np.convolve(data, np.ones(window)/window, mode='valid'), label = "smoothened", alpha = 1.0)
        plt.plot(data, label = "raw", alpha = 0.3)
        plt.grid()
        plt.legend()
        plt.savefig("irradiance_plot_{self.env.day}.pdf")
        plt.close()
        
    def plot_processed_stored_ratio(self, data):
        window = 10
        plt.suptitle("Single agent - Processed/stored")
        plt.title(f"B = {self.battery_capacity}, p_I = {self.p_idle}, p_F = {self.p_max}, fps = {self.env.fps}")
        plt.xlabel("Episodes")
        plt.ylabel("Frames")
        
        # plt.plot(range(window - 1, len(stored)), np.convolve(stored, np.ones(window)/window, mode='valid'), label = "smoothened stored", alpha = 1.0)
        # plt.plot(stored, label = "raw stored", alpha = 0.3)
        plt.plot(range(window - 1, len(data)), np.convolve(data, np.ones(window)/window, mode='valid'), label = "smoothened", alpha = 1.0)
        plt.plot(data, label = "raw", alpha = 0.3)
        plt.grid()
        plt.legend()
        plt.savefig(f"processed_stored_ratio_{self.battery_capacity}Wh_{self.fps}fps_{self.env.day}.pdf")
        plt.close()
        
    def plot_processed_storage(self, processed, stored):
        window = 10
        plt.suptitle("Single agent - Frames management")
        plt.title(f"B = {self.battery_capacity}, p_I = {self.p_idle}, p_F = {self.p_max}, fps = {self.env.fps}")
        plt.xlabel("Episodes")
        plt.ylabel("Frames")
        
        plt.plot(range(window - 1, len(stored)), np.convolve(stored, np.ones(window)/window, mode='valid'), label = "stored", alpha = 1.0)
        # plt.plot(stored, label = "raw stored", alpha = 0.3)
        plt.plot(range(window - 1, len(processed)), np.convolve(processed, np.ones(window)/window, mode='valid'), label = "processed", alpha = 1.0)
        # plt.plot(processed, label = "raw processed", alpha = 0.3)
        
        plt.grid()
        plt.legend()
        # plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
        # plt.tight_layout()
        plt.savefig(f"frames_qlt_{self.battery_capacity}Wh_{self.fps}fps_qlt_{self.env.day}.pdf")
        plt.close()
        
        plt.title("Single agent - Average storage")
        plt.xlabel("Episodes")
        plt.ylabel("Storage level")

        plt.plot(range(window - 1, len(stored)), np.convolve(stored, np.ones(window)/window, mode='valid'), label = f"stored", alpha = 1.0)

        plt.grid()
        plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
        plt.tight_layout()
        plt.savefig(f"storage_comparison_plot_{self.battery_capacity}Wh_{self.fps}fps_qlt_{self.env.day}.pdf")
        plt.close()

    def plot_storage_daily(self, data):
        window = 10
        plt.suptitle("Single agent - Daily storage")
        plt.title(f"B = {self.battery_capacity}, p_I = {self.p_idle}, p_F = {self.p_max}, fps = {self.env.fps}")

        plt.xlabel("Step")
        plt.ylabel("storage")
        
        for i in range(len(data)):
            plt.plot(range(window - 1, len(data[i])), np.convolve(data[i], np.ones(window)/window, mode='valid'), label = f"{(i * int(self.episodes/10))}-th ep", alpha = 1.0)

            # plt.plot(storage_traces[i], label = f"{str(i * int(self.episodes/10))}-th ep." )
        
        plt.grid()
        plt.legend()
        plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"storage_daily_{self.battery_capacity}Wh_{self.fps}fps_qlt_{self.env.day}.pdf")
        plt.close()

            
    # def save_table(self, episode, battery_bins, time_bins):
    #     array_drop = []
    #     for i in range(battery_bins):
    #         row = []
    #         for j in range(time_bins):
    #             row.append(self.table[i, j, 0])
    #         array_drop.append(row)
            
    #     array = np.array(array_drop)
    #     np.savetxt(f'./table_saves/table_ep{episode}_DROP.csv', array, delimiter=",")
        
    #     array_process = []
    #     for i in range(battery_bins):
    #         row = []
    #         for j in range(time_bins):
    #             row.append(self.table[i, j, 1])
    #         array_process.append(row)
            
    #     array = np.array(array_process)
    #     np.savetxt(f'./table_saves/table_ep{episode}_PROCESS.csv', array, delimiter=",")
        
        
############################################################################################


'''
    n    |   eps_dec |   eps_0 | eps_n
   ------------------------------------
    29   |  0.90     |   1.0   | 0.05
    32   |  0.91
    36   |  0.92
    42   |  0.93
    49   |  0.94
    59   |  0.95
    74   |  0.96
    99   |  0.97
    149  |  0.98
    299  |  0.99

'''