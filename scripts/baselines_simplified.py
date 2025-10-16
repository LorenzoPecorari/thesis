import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from stable_baselines3 import DQN
import environment_simplified
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class DQN_agent:
    def __init__(
            self,
             datapath,
            battery_capacity,
            power_idle,
            power_frame,
            delta_time,
            proc_interval,
            max_irrad,
            pv_efficiency,
            pv_area,
            fps,
            seed,
            episodes
            ):

        self.env = environment_simplified.EnergyPVEnv(
            datapath,
            battery_capacity,
            power_idle,
            power_frame,
            delta_time,
            proc_interval,
            max_irrad,
            pv_efficiency,
            pv_area,
            fps
        )

        self.model = DQN("MlpPolicy",
                            self.env,
                            buffer_size=32000,
                            learning_starts=1000,
                            batch_size=32,
                            gamma = 0.9,
                            target_update_interval=1000,
                            exploration_fraction=0.3,
                            exploration_initial_eps=1.0,
                            exploration_final_eps=0.05,
                            )
        self.model.learn(total_timesteps = ((3600 / proc_interval) * 24 * episodes))
        self.episodes = episodes
        self.rewards = []
        self.frames_dropped = []
        self.frames_processed = []
        self.battery_levels = []
        self.seed = seed

    def train(self):
        for j in range(self.episodes):
            obs, info = self.env.reset(self.seed)
            reward = 0
            partial_reward = 0
            battery = 0

            for i in range(self.env.max_steps):
                # print(obs)
                # print(info)
                # print()
                
                action, _states = self.model.predict(obs, deterministic= False)
                obs, partial_reward, terminated, truncated, info = self.env.step(action)
                reward += partial_reward
                battery += info['battery_level']

            print(f"episode: {j}/{self.episodes} -  reward:{reward}")
            self.rewards.append(reward)
            self.frames_dropped.append(self.env.total_frames_dropped)
            self.frames_processed.append(self.env.total_frames_processed)
            self.battery_levels.append(battery / self.env.max_steps)

        self.plot_rewards(self.rewards)
        self.plot_frames(self.frames_dropped, self.frames_processed)
        self.plot_battery(self.battery_levels)

    def plot_rewards(self, data):
        window = 10
        plt.suptitle("DQN - rewards")
        plt.title(f"B = {self.env.battery_capacity}, e_I = {round(self.env.e_idle * 3600 / self.env.interval, 2)}, e_F = {round(self.env.e_frame * 3600, 2)}, fps = {self.env.fps}")
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.plot(range(window - 1, len(data)), np.convolve(data, np.ones(window)/window, mode='valid'), label = "smoothened", alpha = 1.0)
        plt.plot(data, label = "raw", alpha = 0.3)
        plt.grid()
        plt.legend()
        plt.savefig("rewards_plot_dqn.pdf")
        plt.close()

    def plot_frames(self, dropped, processed):
        window = 10
        plt.suptitle("DQN - frames")
        plt.title(f"B = {self.env.battery_capacity}, e_I = {round(self.env.e_idle * 3600 / self.env.interval, 2)}, e_F = {round(self.env.e_frame * 3600, 2)}, fps = {self.env.fps}")
        plt.xlabel("Episodes")
        plt.ylabel("Frames")
        
        plt.plot(range(window - 1, len(dropped)), np.convolve(dropped, np.ones(window)/window, mode='valid'), label = "smoothened dropped", alpha = 1.0)
        plt.plot(dropped, label = "raw processed", alpha = 0.3)
        plt.plot(range(window - 1, len(processed)), np.convolve(processed, np.ones(window)/window, mode='valid'), label = "smoothened processed", alpha = 1.0)
        plt.plot(processed, label = "raw processed", alpha = 0.3)
        
        plt.grid()
        plt.legend()
        plt.savefig("frames_plot_dqn.pdf")
        plt.close()

    def plot_frames(self, dropped, processed):
        window = 10
        plt.suptitle("DQN - frames")
        plt.title(f"B = {self.env.battery_capacity}, e_I = {round(self.env.e_idle * 3600 / self.env.interval, 2)}, e_F = {round(self.env.e_frame * 3600, 2)}, fps = {self.env.fps}")
        plt.xlabel("Episodes")
        plt.ylabel("Frames")
        
        plt.plot(range(window - 1, len(dropped)), np.convolve(dropped, np.ones(window)/window, mode='valid'), label = "smoothened dropped", alpha = 1.0)
        plt.plot(dropped, label = "raw processed", alpha = 0.3)
        plt.plot(range(window - 1, len(processed)), np.convolve(processed, np.ones(window)/window, mode='valid'), label = "smoothened processed", alpha = 1.0)
        plt.plot(processed, label = "raw processed", alpha = 0.3)
        
        plt.grid()
        plt.legend()
        plt.savefig("frames_plot_dqn.pdf")
        plt.close()

    def plot_battery(self, data):
        window = 10
        plt.suptitle("DQN - battery level")
        plt.title(f"B = {self.env.battery_capacity}, e_I = {round(self.env.e_idle * 3600 / self.env.interval, 2)}, e_F = {round(self.env.e_frame * 3600, 2)}, fps = {self.env.fps}")
        plt.xlabel("Episodes")
        plt.ylabel("Battery level")
        plt.plot(range(window - 1, len(data)), np.convolve(data, np.ones(window)/window, mode='valid'), label = "smoothened", alpha = 1.0)
        plt.plot(data, label = "raw", alpha = 0.3)
        plt.grid()
        plt.legend()
        plt.savefig("battery_plot_dqn.pdf")
        plt.close()

######################################################################################

# env params
datapath = '../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv'
battery_capacity = 6000000                # [Wh]
power_idle = 0.0                        # [W]
power_frame = 5.0                       # [W]
delta_time = 15 * 60                    # [sec]
proc_interval = 15 * 60                  # [sec]
max_irrad = 1200                        # [W/m^2]
pv_efficiency = 0.2
pv_area = 1.0
fps = 30
seed = "linear"
episodes = 350

agent = DQN_agent(datapath, battery_capacity, power_idle, power_frame, delta_time, proc_interval, max_irrad, pv_efficiency, pv_area, fps, seed, episodes)
agent.train()
