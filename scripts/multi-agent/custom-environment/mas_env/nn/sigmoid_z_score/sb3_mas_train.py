from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure

from custom_environment import CustomEnvironment
from env_wrapper import EnvWrapper

import numpy as np
import matplotlib.pyplot as plt

import torch

import time
import csv
import os

class SB3_MAS_Train:
    def __init__(self, 
                 num_agents,
                 num_episodes,
                 irradiance_datapaths,
                 delta_time,
                 proc_interval,
                 proc_rate,
                 arrival_rate,
                 eps_init,
                 eps_fin,
                 eps_dec,
                 battery_capacities,
                 panel_surfaces,
                 power_idle,
                 power_max,
                 train_freq,
                 w,
                 mode,
                 batch_size,
                 seed
                ):
        
        self.num_agents = num_agents
        self.num_episodes = num_episodes
        
        self.delta_time = delta_time
        self.proc_interval = proc_interval
        self.proc_rate = proc_rate
        
        self.battery_capacities = battery_capacities
        self.panel_surfaces = panel_surfaces
        
        self.power_idle = power_idle
        self.power_max = power_max
        
        self.seed = seed
        self.w = w
        self.mode = mode
        
        self.train_freq = train_freq
        self.grad_steps = 1
        self.batch_size = batch_size
        
        self.eps_init = eps_init
        self.eps_dec = eps_dec
        self.eps_fin = eps_fin
        self.eps = 0
        
        self.env = CustomEnvironment(
        num_agents,
        irradiance_datapaths,
        delta_time,
        proc_interval,
        proc_rate,
        arrival_rate,
        battery_capacities,
        panel_surfaces,
        power_idle,
        power_max,
        w,
        seed)
        
        self.max_steps = self.env.max_steps
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"CUDA Available: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print(f"CUDA NOT Available - using CPU")
        
        
        self.models = {i : DQN(
                policy="MlpPolicy",
                env=EnvWrapper(self.env, i),
                learning_rate=0.0001,
                buffer_size=10000,
                learning_starts=500,
                batch_size=self.batch_size,
                tau=1.0,
                gamma=0.99,
                train_freq=self.train_freq,
                gradient_steps=self.grad_steps,
                replay_buffer_class=None,
                replay_buffer_kwargs=None,
                optimize_memory_usage=False,
                n_steps=1,
                target_update_interval=((3600 / (proc_interval * 60))),
                exploration_fraction=1.0,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                max_grad_norm=1,
                stats_window_size=100,
                tensorboard_log=None,
                policy_kwargs=None,
                verbose=0,
                seed=None,
                device=mode,
                _init_setup_model=True
            )
          for i in range(0, num_agents)}
        
        for i in range(num_agents):
            self.models[i].set_logger(configure(None, ["stdout"]))
            print(f"Agent {i} device: {self.models[i].device}")
        
    
    def plot_rewards(self, rewards):
    
    # print(rewards)
    
        window = 10
        plt.suptitle("Multi-agent : rewards")
        plt.title(f"P_i = {self.power_idle}, P_f = {self.power_max}, fps = {self.proc_rate}, interval: {self.proc_interval}s")
        
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        
        for i in range(0, self.num_agents):
            # print(rewards[i])
            plt.plot(range(window - 1, len(rewards[i])), np.convolve(rewards[i], np.ones(window)/window, mode='valid'), label = f"smooth {self.battery_capacities[i]}Wh", alpha = 1.0)
            plt.plot(rewards[i], label = f"raw {self.battery_capacities[i]}Wh", alpha = 0.3)
        
        plt.grid()
        # plt.ylim(-10, 500)
        plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"rewards_plot_{self.num_episodes - 1}_{self.env.episode}_{self.proc_interval}_{self.w}_{self.num_agents}agents_{self.mode}.pdf")
        plt.close()

    
    def plot_battery_levels(self, levels):
        window = 10
        plt.suptitle("Multi-agent : battery levels")
        plt.title(f"P_i = {self.power_idle}, P_f = {self.power_max}, fps = {self.proc_rate}, interval: {self.proc_interval}s")
        
        plt.xlabel("Episodes")
        plt.ylabel("Battery")
        
        for i in range(0, self.env._num_agents):    
            # print(rewards[i])
            plt.plot(range(window - 1, len(levels[i])), np.convolve(levels[i], np.ones(window)/window, mode='valid'), label = f"smooth {self.battery_capacities[i]}Wh", alpha = 1.0)
            plt.plot(levels[i], label = f"raw {self.battery_capacities[i]}Wh", alpha = 0.3)
        
        plt.grid()
        plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"avg_battery_plot_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.w}_{self.num_agents}agents_{self.mode}.pdf")
        plt.close()


    def plot_battery_levels_individual(self, levels):
        window = 10
        # plt.suptitle("Multi-agent : battery levels")
        # plt.title(f"P_i = {power_idle}, P_f = {power_max}, fps = {proc_rate}, interval: {proc_interval}s")
        
        # plt.xlabel("Episodes")
        # plt.ylabel("Battery")
        
        for i in range(0, self.env._num_agents):
            plt.suptitle("Multi-agent : battery levels")
            plt.title(f"P_i = {self.power_idle}, P_f = {self.power_max}, fps = {self.proc_rate}, interval: {self.proc_interval}s")
            
            plt.xlabel("Episodes")
            plt.ylabel("Battery")
        
            # print(rewards[i])
            plt.plot(range(window - 1, len(levels[i])), np.convolve(levels[i], np.ones(window)/window, mode='valid'), label = f"smooth {self.battery_capacities[i]}Wh", alpha = 1.0)
            plt.plot(levels[i], label = f"raw {self.battery_capacities[i]}Wh", alpha = 0.3)
        
            plt.grid()
            plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
            plt.tight_layout()
            plt.savefig(f"avg_battery_plot_{self.battery_capacities[i]}Wh_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.w}_{self.num_agents}agents_{self.mode}.pdf")
            plt.close()

    
    def plot_backlogs(self, backlogs):
        window = 10
        plt.suptitle("Multi-agent : average backlog")
        plt.title(f"P_i = {self.power_idle}, P_f = {self.power_max}, fps = {self.proc_rate}, interval: {self.proc_interval}s")
        
        plt.xlabel("Timesteps")
        plt.ylabel("Baclog")
        
        for i in range(0, self.env._num_agents):
            # print(rewards[i])
            plt.plot(range(window - 1, len(backlogs[i])), np.convolve(backlogs[i], np.ones(window)/window, mode='valid'), label = f"smooth {self.battery_capacities[i]}Wh", alpha = 1.0)
            plt.plot(backlogs[i], label = f"raw {self.battery_capacities[i]}Wh", alpha = 0.3)
        
        plt.grid()
        # plt.legend()
        plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"backlog_plot_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.w}_{self.num_agents}agents_{self.mode}.pdf")
        plt.close()

    
    def plot_battery_daily(self, data):    
        for elem in range(0, self.env._num_agents):
                    
            window = 40
            plt.suptitle("Multi-agent : daily battery")
            plt.title(f"B: {self.env.battery_capacities[elem] / 3600} - P_i = {self.power_idle}, P_f = {self.power_max}, fps = {self.proc_rate}, interval: {self.proc_interval}s")
            
            plt.xlabel("Timesteps")
            plt.ylabel("Battery")
            for i in range(0, len(data[elem])):
                # print(rewards[i])
                plt.plot(range(window - 1, len(data[elem][i])), np.convolve(data[elem][i], np.ones(window)/window, mode='valid'), label = f"{i * (int((self.num_episodes-1) / 10))}-th episode", alpha = 1.0)
            
            plt.grid()
            plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
            plt.tight_layout()
            plt.savefig(f"battery_{int(self.env.battery_capacities[elem] / 3600)}Wh_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.w}_{self.num_agents}agents_{self.mode}.pdf")
            plt.close()
            
    
    def plot_backlog_daily(self, data):
        
        for elem in range(0, self.env._num_agents):
            
            window = 40
            plt.suptitle("Multi-agent : daily backlog")
            plt.title(f"B: {self.env.battery_capacities[elem] / 3600} - P_i = {self.power_idle}, P_f = {self.power_max}, fps = {self.proc_rate}, interval: {self.proc_interval}s")
            
            plt.xlabel("Timesteps")
            plt.ylabel("Backlog")
            for i in range(0, len(data[elem])):
                # print(rewards[i])
                plt.plot(range(window - 1, len(data[elem][i])), np.convolve(data[elem][i], np.ones(window)/window, mode='valid'), label = f"{i* (int((self.num_episodes-1) / 10))}-th episode", alpha = 1.0)
            
            plt.grid()
            # plt.legend()
            plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
            plt.tight_layout()
            plt.savefig(f"backlog_{int(self.env.battery_capacities[elem] / 3600)}Wh_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.w}_{self.num_agents}agents_{self.mode}.pdf")
            plt.close()        

    
    def plot_framerate(self, fs):
        window = 10
        plt.suptitle("Multi-agent : average framerate")
        plt.title(f"P_i = {self.power_idle}, P_f = {self.power_max}, fps = {self.proc_rate}, interval: {self.proc_interval}s")
        
        plt.xlabel("Episodes")
        plt.ylabel("Framerate")
        
        for i in range(0, self.env._num_agents):
            # print(rewards[i])
            plt.plot(range(window - 1, len(fs[i])), np.convolve(fs[i], np.ones(window)/window, mode='valid'), label = f"smooth {self.battery_capacities[i]}Wh", alpha = 1.0)
            plt.plot(fs[i], label = f"raw {self.battery_capacities[i]}Wh", alpha = 0.3)
        
        plt.grid()
        plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"framerate_plot_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.w}_{self.env._num_agents}agents_{self.mode}.pdf")
        plt.close()
        
    
    def plot_local_framerate(self, fs):
        window = 10
        plt.suptitle("Multi-agent : local average framerate")
        plt.title(f"P_i = {self.power_idle}, P_f = {self.power_max}, fps = {self.proc_rate}, interval: {self.proc_interval}s")
        
        plt.xlabel("Episodes")
        plt.ylabel("Backlog")
        
        for i in range(0, self.env._num_agents):
            # print(rewards[i])
            plt.plot(range(window - 1, len(fs[i])), np.convolve(fs[i], np.ones(window)/window, mode='valid'), label = f"smooth {self.battery_capacities[i]}Wh", alpha = 1.0)
            plt.plot(fs[i], label = f"raw {self.battery_capacities[i]}Wh", alpha = 0.3)
        
        plt.grid()
        plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"local_framerate_plot_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.w}_{self.env._num_agents}agents_{self.mode}.pdf")
        plt.close()
      
      
    def plot_offloading_framerate(self, fs):
        window = 10
        plt.suptitle("Multi-agent : offloading average framerate")
        plt.title(f"P_i = {self.power_idle}, P_f = {self.power_max}, fps = {self.proc_rate}, interval: {self.proc_interval}s")
        
        plt.xlabel("Episodes")
        plt.ylabel("Backlog")
        
        for i in range(0, self.env._num_agents):
            # print(rewards[i])
            plt.plot(range(window - 1, len(fs[i])), np.convolve(fs[i], np.ones(window)/window, mode='valid'), label = f"smooth {self.battery_capacities[i]}Wh", alpha = 1.0)
            # plt.plot(fs[i], label = f"raw {self.battery_capacities[i]}Wh", alpha = 0.3)
        
        plt.grid()
        plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"offloading_framerate_plot_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.w}_{self.env._num_agents}agents_{self.mode}.pdf")
        plt.close()
        
    def plot_offloading_matchings(self, fs):
        window = int(self.num_episodes / 100)
        plt.suptitle("Multi-agent : average offloading matchings")
        plt.title(f"P_i = {self.power_idle}, P_f = {self.power_max}, fps = {self.proc_rate}, interval: {self.proc_interval}s")
        
        plt.xlabel("Episodes")
        plt.ylabel("Matchings")
        # plt.ylim(0, 10)
        
        for i in range(0, self.env._num_agents):
            # print(rewards[i])
            plt.plot(range(window - 1, len(fs[i])), np.convolve(fs[i], np.ones(window)/window, mode='valid'), label = f"smooth {self.battery_capacities[i]}Wh", alpha = 1.0)
            # plt.plot(fs[i], label = f"raw {self.battery_capacities[i]}Wh", alpha = 0.3)
        
        plt.grid()
        plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"offloading_matchings_plot_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.w}_{self.env._num_agents}agents_{self.mode}.pdf")
        plt.close()

    def save_battery_csv(self, battery_daily):
        
        for agent_id in range(self.num_agents):
            filename = f"./csvs/csvs_batch_{self.batch_size}/battery_{self.battery_capacities[agent_id]}_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.num_agents}agents_{self.mode}.csv"
            
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                
                if len(battery_daily[agent_id]) > 0 and len(battery_daily[agent_id][0]) > 0:
                    n_timesteps = len(battery_daily[agent_id][0])
                    header = [f't{i}' for i in range(n_timesteps)]
                    writer.writerow(header)
                    
                    for episode_data in battery_daily[agent_id]:
                        writer.writerow(episode_data)
            
            print(f"saved: {filename}")
    
    def save_backlog_csv(self, backlog_daily):
        
        for agent_id in range(self.num_agents):
            filename = f"./csvs/csvs_batch_{self.batch_size}/backlog_{self.battery_capacities[agent_id]}_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.num_agents}agents_{self.mode}.csv"
            
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                
                if len(backlog_daily[agent_id]) > 0 and len(backlog_daily[agent_id][0]) > 0:
                    n_timesteps = len(backlog_daily[agent_id][0])
                    header = [f't{i}' for i in range(n_timesteps)]
                    writer.writerow(header)
                    
                    for episode_data in backlog_daily[agent_id]:
                        writer.writerow(episode_data)
            
            print(f"saved: {filename}")

    def save_rewards_csv(self, rewards):
        os.makedirs('./csvs/csvs_batch_{self.batch_size}', exist_ok=True)
        
        for agent_id in range(self.num_agents):
            filename = f"./csvs/csvs_batch_{self.batch_size}/rewards_agent_{self.battery_capacities[agent_id]}_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.num_agents}agents_{self.mode}.csv"
            
            with open(filename, "w") as file:
                for elem in rewards[agent_id]:
                    file.write(str(float(elem)) + "\n")
                
        print(f"saved: {filename}")
        
    def save_time_csv(self, times):
        os.makedirs('./csvs/csvs_batch_{self.batch_size}', exist_ok=True)
        filename = f"./csvs/csvs_batch_{self.batch_size}/time_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.num_agents}agents_{self.mode}.csv"

        with open(filename, "w") as file:
            for elem in times:
                file.write(str(float(elem)) + "\n")
                
        print(f"saved: {filename}")

    def save_framerate_csv(self, fs, hs, framerates):
        os.makedirs(f'./csvs/csvs_batch_{self.batch_size}', exist_ok=True)
        
        for agent_id in range(self.num_agents):
            filename = f"./csvs/csvs_batch_{self.batch_size}/local_framerate_{self.battery_capacities[agent_id]}_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.num_agents}agents_{self.mode}.csv"
            with open(filename, "w") as file:
                for elem in fs[agent_id]:
                    file.write(str(float(elem)) + "\n")

                print(f"saved: {filename}")

            
            filename = f"./csvs/csvs_batch_{self.batch_size}/offloading_framerate_{self.battery_capacities[agent_id]}_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.num_agents}agents_{self.mode}.csv"
            with open(filename, "w") as file:
                for elem in hs[agent_id]:
                    file.write(str(float(elem)) + "\n")
                print(f"saved: {filename}")

            
            filename = f"./csvs/csvs_batch_{self.batch_size}/total_framerate_{self.battery_capacities[agent_id]}_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.num_agents}agents_{self.mode}.csv"
            with open(filename, "w") as file:
                for elem in framerates[agent_id]:
                    file.write(str(float(elem)) + "\n")
                print(f"saved: {filename}")
        
    def save_offloading_matchings_csv(self, hs_matchings):
        os.makedirs(f'./csvs/csvs_batch_{self.batch_size}', exist_ok=True)
        
        for agent_id in range(self.num_agents):
            filename = f"./csvs/csvs_batch_{self.batch_size}/matchings_{self.battery_capacities[agent_id]}_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.num_agents}agents_{self.mode}.csv"
            with open(filename, "w") as file:
                for elem in hs_matchings[agent_id]:
                    file.write(str(int(elem)) + "\n")
                print(f"saved: {filename}")

    def decode(self, encoded_action):
        fti = int(encoded_action / (3 * self.num_agents * (self.proc_rate + 1)))
        r = int(encoded_action % (3 * self.num_agents * (self.proc_rate + 1)))
        
        xti = int(r / (self.num_agents * (self.proc_rate + 1)))
        r = int(r % ((self.num_agents * (self.proc_rate + 1))))
        
        gti = int(r / (self.proc_rate + 1))
        r = int(r % (self.proc_rate + 1))

        hti = r

        # print([fti, xti, gti, hti])        
        return [fti, xti, gti, hti]
    
    
    def get_action(self, agent_id, obs, actions_encoded, actions):
        if(np.random.random() < self.eps):
            action = self.models[agent_id].action_space.sample()
        else:
            action, _ = self.models[agent_id].predict(obs, deterministic=False)
        
        actions_encoded[agent_id] = action
        actions[agent_id] = self.decode(action)
        
    
    def update_epsilon(self):
        if(self.eps > self.eps_fin):
            self.eps = max(self.eps * self.eps_dec, self.eps_fin)
    
    def train(self):
        self.eps = self.eps_init
        
        rewards_plot = [[] for agent in range(0, self.num_agents)]    
        batteries = [[] for agent in range(0, self.num_agents)]
        batteries_local = [0 for i in range(0, self.num_agents)]
        battery_daily = [[] for agent in range(0, self.num_agents)]

        backlogs = [[] for agent in range(0, self.num_agents)]
        backlogs_local = [0 for i in range(0, self.num_agents)]
        backlogs_daily = [[] for agent in range(0, self.num_agents)]

        fs = [[] for i in range(0, self.num_agents)]
        hs = [[] for i in range(0, self.num_agents)]
        hs_matchings = [[] for agent in range(0, self.num_agents)]
        framerates = [[] for i in range(0, self.num_agents)]
        
        times = []

        total_timesteps = self.num_episodes * self.max_steps
        
        for i in range(0, self.num_episodes):
            temp = time.time()
            
            obs = self.env.reset()
            
            rewards_episode = {agent: 0.0 for agent in range(self.num_agents)}
            obs = obs[0]
    
            battery_daily_temp = [[] for agent in range(0, self.num_agents)]
            backlog_daily_temp = [[] for agent in range(0, self.num_agents)]
            
            for agent_id in range(0, self.num_agents):
                batteries_local[agent_id] = 0
                backlogs_local[agent_id] = 0
                
            step = 0
            self.update_epsilon()
            
            while self.env.agents:
                actions_encoded = {}
                actions = {}
                current_timestep = i * self.max_steps + step
                progress_remaining = 1.0 - (current_timestep / total_timesteps)
                
                actions_encoded = {}
                actions = {}
                
                for agent_id in range(0, self.num_agents):
                    self.models[agent_id]._current_progress_remaining = progress_remaining     
                    if(np.random.random() < self.eps):
                        action = self.models[agent_id].action_space.sample()
                    else:
                        action, _ = self.models[agent_id].predict(obs[agent_id], deterministic=False)
                    
                    actions_encoded[agent_id] = action
                    actions[agent_id] = self.decode(action)               
                    # self.get_action(agent_id, obs[agent_id], actions_encoded, actions)
                
                # print(actions)
                
                next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
                                    
                for agent_id in range(0, self.num_agents):
                    done = terminations[agent_id] or truncations[agent_id]
                    rewards_episode[agent_id] += rewards[agent_id]
                    action_encoded = actions_encoded[agent_id]
                                      
                    self.models[agent_id].replay_buffer.add(
                        obs = obs[agent_id],
                        next_obs = next_obs[agent_id],
                        action = np.array([action_encoded]),
                        reward = np.array(rewards[agent_id]),
                        done = np.array([done]),
                        infos = [{}]
                    )
                    
                    self.models[agent_id].num_timesteps += 1
                    
                    batteries_local[agent_id] += self.env.battery_energies[agent_id]
                    backlogs_local[agent_id] += self.env.backlogs[agent_id]
                    
                    if (self.models[agent_id].num_timesteps > self.models[agent_id].learning_starts and
                        self.models[agent_id].num_timesteps % self.train_freq == 0):
                        self.models[agent_id].train(gradient_steps=self.grad_steps, batch_size=self.batch_size)
                    
                    if(i % int(self.num_episodes/10) == 0):
                        battery_daily_temp[agent_id].append(self.env.battery_energies[agent_id]/ self.env.battery_capacities[agent_id])
                        backlog_daily_temp[agent_id].append(self.env.backlogs[agent_id])
                    
                    rewards_episode[agent_id] = round(rewards_episode[agent_id], 2)
            
                obs = next_obs    
                step += 1
                    
            temp = time.time() - temp
            times.append(temp)

            # if(i % int(100) == 0):
            print(f"Episode {i + 1}/{self.num_episodes} - rewards: {rewards_episode} - eps: {round(self.eps, 2)} - time: {temp}")
            
            for agent_id in range(0, self.num_agents):            
                fs[agent_id].append(self.env.fs[agent_id] / self.env.max_steps)

                if(self.env.hs_counter[agent_id] > 0):
                    hs[agent_id].append(self.env.hs[agent_id] / self.env.max_steps)
                    # hs[agent_id].append(self.env.hs[agent_id] / self.env.hs_counter[agent_id])
                else:
                    hs[agent_id].append(0.0)
                framerates[agent_id].append(fs[agent_id][-1] + hs[agent_id][-1])
                rewards_plot[agent_id].append(rewards_episode[agent_id]) 
                batteries[agent_id].append((batteries_local[agent_id] / self.env.battery_capacities[agent_id]) / self.env.max_steps)        
                backlogs[agent_id].append(backlogs_local[agent_id] / self.env.max_steps)            

                hs_matchings[agent_id].append(int(self.env.hs_counter[agent_id]))

                if(i % int(self.num_episodes / 10) == 0):
                    battery_daily[agent_id].append(battery_daily_temp[agent_id])
                    backlogs_daily[agent_id].append(backlog_daily_temp[agent_id])

                self.env.fs[agent_id] = 0
                self.env.hs[agent_id] = 0
                self.env.hs_counter[agent_id] = 0

                
        self.plot_rewards(rewards_plot)  
        self.plot_backlogs(backlogs)
        self.plot_battery_levels(batteries)
        self.plot_backlog_daily(backlogs_daily)
        self.plot_battery_daily(battery_daily)
        self.plot_framerate(framerates)
        self.plot_local_framerate(fs)
        self.plot_offloading_framerate(hs)
        self.plot_offloading_matchings(hs_matchings)
        
        self.save_battery_csv(battery_daily)
        self.save_backlog_csv(backlogs_daily)
        self.save_rewards_csv(rewards_plot)
        self.save_time_csv(times)
        self.save_framerate_csv(fs, hs, framerates)
        self.save_offloading_matchings_csv(hs_matchings)