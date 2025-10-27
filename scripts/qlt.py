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
                 power_frame,
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
            power_idle,
            power_frame,
            delta_time,
            proc_interval,
            max_irradiation,
            pv_efficiency,
            pv_area,
            fps)

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
        self.table = np.zeros((battery_bins, time_bins, 2))

    def choice_action(self, state, eps):
        b_idx, t_idx = self.state_discretization(state[0], state[1])
        
        # print(f"Q drop: {self.table[b_idx, t_idx, 0]}, Q process: {self.table[b_idx, t_idx, 1]} - ", end="")
        if np.random.random() < eps:
            # print("RANDOM")
            return random.randint(0, 1)
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
        
        # print(f"\n State of Q({b_idx}, {t_idx}, {action}): {self.table[b_idx, t_idx, action]} -> ", end="")
        self.table[b_idx, t_idx, action] = (1 - self.alpha) * self.table[b_idx, t_idx, action] + (self.alpha * (reward + (self.gamma * np.max(self.table[next_b_idx, next_t_idx]))))
        # print(f"{self.table[b_idx, t_idx, action]}")
    
    def train(self):
        rewards = []
        dropped_frames = []
        processed_frames = []
        battery = []
        irradiance = []
        cumulative_traces = []
        battery_traces = []

        for episode in range(self.episodes + 1):
            state, _ = self.env.reset(self.seed)
            partial_reward = 0
            battery_avg = 0
            avg_irrad = 0
            battery = []
            
            trace = []
            info = {}

            self.update_eps()            

            for j in range(self.env.max_steps):
                state = self.state_discretization(state[0], state[1])
                action = self.choice_action(state, self.eps)
                
                new_state, reward, terminated, truncated, info = self.env.step(action)

                self.update_table(state, new_state, action, reward)
                partial_reward += reward
                state = new_state
                battery.append(info['battery_level'] * self.battery_capacity)
                battery_avg += info['battery_level']
                avg_irrad += info['irradiance']
                trace.append(partial_reward)
            
            if(episode % 50 == 0):
                cumulative_traces.append(trace)
                battery_traces.append(battery)
            
            print(f"episode: {episode}/{self.episodes} - reward: {partial_reward} - eps: {self.eps}")
            # print(f"dropped: {info['frames_dropped']} - processed : {info['frames_processed']} - avg battery : {battery_avg / self.env.max_steps} - avg irradiance: {avg_irrad / self.env.max_steps}")
            dropped_frames.append(info['frames_dropped'])
            processed_frames.append(info['frames_processed'])
            battery.append(battery_avg / self.env.max_steps)
            irradiance.append(avg_irrad / self.env.max_steps)
            
            # self.save_table(episode)
            
            rewards.append(partial_reward)
            # input("Press enter to continue...")
        
        # self.plot_cumulative_trace(cumulative_traces)
        # self.plot_daily_battery(battery_traces)

        return rewards, dropped_frames, processed_frames, battery, irradiance

    ### daily metrics

    def plot_cumulative_trace(self, data):
        plt.title("Q-Learning tabular - Cumulative rewards")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        
        for i in range(len(data)):
            plt.plot(data[i], label = f"{str(i*50)}-th episode")
        
        plt.grid()
        plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig("cumulative_rewards_qlt.pdf")
        plt.close()    
            
    def plot_daily_battery(self, data):
        plt.title("Q-Learning tabular - Daily battery")
        plt.xlabel("Step")
        plt.ylabel("Battery")
        
        for i in range(len(data)):
            plt.plot(data[i], label = f"{str(i*50)}-th battery" )
        
        plt.grid()
        plt.legend()
        plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig("battery_daily_qlt.pdf")
        plt.close()  
        
            
    ### training metrics

    def plot_rewards(self, data):
        window = 10
        plt.suptitle("Q-Learning tabular - rewards")
        plt.title(f"B = {self.env.battery_capacity}, e_I = {round(self.env.e_idle * 3600 / self.env.interval, 2)}, e_F = {round(self.env.e_frame * 3600, 2)}, fps = {self.env.fps}")
        
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.plot(range(window - 1, len(data)), np.convolve(data, np.ones(window)/window, mode='valid'), label = "smoothened", alpha = 1.0)
        plt.plot(data, label = "raw", alpha = 0.3)
        plt.grid()
        plt.legend()
        plt.savefig("rewards_plot_qlt.pdf")
        plt.close()

    def plot_frames(self, dropped, processed):
        window = 10
        plt.suptitle("Q-Learning tabular - frames")
        plt.title(f"B = {self.env.battery_capacity}, e_I = {round(self.env.e_idle * 3600 / self.env.interval, 2)}, e_F = {round(self.env.e_frame * 3600, 2)}, fps = {self.env.fps}")
        plt.xlabel("Episodes")
        plt.ylabel("Frames")
        
        plt.plot(range(window - 1, len(dropped)), np.convolve(dropped, np.ones(window)/window, mode='valid'), label = "smoothened dropped", alpha = 1.0)
        plt.plot(dropped, label = "raw dropped", alpha = 0.3)
        plt.plot(range(window - 1, len(processed)), np.convolve(processed, np.ones(window)/window, mode='valid'), label = "smoothened processed", alpha = 1.0)
        plt.plot(processed, label = "raw processed", alpha = 0.3)
        
        plt.grid()
        plt.legend()
        plt.savefig("frames_plot_qlt.pdf")
        plt.close()

    def plot_battery(self, data):
        window = 10
        plt.suptitle("Q-Learning tabular - battery level")
        plt.title(f"B = {self.env.battery_capacity}, e_I = {round(self.env.e_idle * 3600 / self.env.interval, 2)}, e_F = {round(self.env.e_frame * 3600, 2)}, fps = {self.env.fps}")
        
        plt.xlabel("Episodes")
        plt.ylabel("Battery level")
        plt.plot(range(window - 1, len(data)), np.convolve(data, np.ones(window)/window, mode='valid'), label = "smoothened", alpha = 1.0)
        plt.plot(data, label = "raw", alpha = 0.3)
        plt.grid()
        plt.legend()
        plt.savefig("battery_plot_qlt.pdf")
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
        plt.savefig("irradiance_plot.pdf")
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