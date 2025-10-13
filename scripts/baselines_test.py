import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from stable_baselines3 import DQN
import environment
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# env params
datapath = '../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv'
datapath_2 = '../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2023.csv'
battery_capacity = 100                  # [Wh]
power_idle = 2.5                        # [W]
power_frame = 7.0                       # [W]
delta_time = 15 * 60                    # [sec]
proc_interval = (1/60) * 60                  # [sec]
max_irrad = 1200                        # [W/m^2]
pv_efficiency = 0.2
pv_area = 1.0

env = environment.EnergyPVEnv(
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
env.reset(None)
print("env done!")

env2 = environment.EnergyPVEnv(
        datapath_2,
        battery_capacity,
        power_idle,
        power_frame,
        delta_time,
        proc_interval,
        max_irrad,
        pv_efficiency,
        pv_area
    )
env2.reset(None)
print("env2 done!")

model = DQN("MlpPolicy",
            env,
            buffer_size=32000,
            learning_starts=1000,
            batch_size=32,
            gamma = 0.9,
            target_update_interval=1000,
            exploration_fraction=0.3,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            )

model.learn(total_timesteps=100000)
model.save("dqn_energy_pv")

model = DQN.load("dqn_energy_pv")

rewards_env = []

# train over first dataset
for j in range(100):
    obs, info = env.reset(None)
    reward = 0
    partial_reward = 0

    for i in range(env.max_steps):
        print(obs)
        print(info)
        print()
        
        action, _states = model.predict(obs, deterministic= True)
        obs, partial_reward, terminated, truncated, info = env.step(action)
        reward += partial_reward

    rewards_env.append(reward)


model = DQN("MlpPolicy",
            env2,
            buffer_size=32000,
            learning_starts=1000,
            batch_size=32,
            gamma = 0.9,
            target_update_interval=1000,
            exploration_fraction=0.3,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            )

plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.grid()
plt.plot(rewards_env)
plt.savefig("rewards_plot.pdf")

# plt.plot(range(window - 1, len(rewards_env2)), np.convolve(rewards_env2, np.ones(window)/window, mode='valid'), label = "Env 2023")
# plt.legend()
# plt.title("Rewards comparison")

# model.learn(total_timesteps=100000)
# model = DQN.load("dqn_energy_pv")

# rewards_env2 = []
# # train over second dataset
# for j in range(100):
#     obs, info = env2.reset(None)
#     reward = 0
#     partial_reward = 0

#     for i in range(env2.max_steps):
#         print(obs)
#         print(info)
#         print()
        
#         action, _states = model.predict(obs, deterministic= True)
#         obs, partial_reward, terminated, truncated, info = env2.step(action)
#         reward += partial_reward

#     rewards_env2.append(reward)

# window = 100
# plt.xlabel("Episodes")
# plt.ylabel("Rewards")
# plt.grid()
# plt.plot(range(window - 1, len(rewards_env)), np.convolve(rewards_env, np.ones(window)/window, mode='valid'), label = "Env 2024")
# plt.plot(range(window - 1, len(rewards_env2)), np.convolve(rewards_env2, np.ones(window)/window, mode='valid'), label = "Env 2023")
# plt.legend()
# plt.title("Rewards comparison")
# plt.savefig("rewards_both_datasets.pdf")
