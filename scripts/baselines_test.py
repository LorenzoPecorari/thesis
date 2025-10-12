import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from stable_baselines3 import DQN
import environment
import matplotlib
import matplotlib.pyplot as plt

# env params
datapath = '../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv'
datapath_2 = '../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2023.csv'
battery_capacity = 100                  # [Wh]
power_idle = 2.5                        # [W]
power_frame = 7.0                       # [W]
delta_time = 15 * 60                    # [sec]
proc_interval = 1 * 60                  # [sec]
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

env2 = env = environment.EnergyPVEnv(
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
env.reset(None)

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

model.learn(total_timesteps=10000)
model.save("dqn_energy_pv")

model = DQN.load("dqn_energy_pv")

rewards = [0]
partial_reward = 0
reward = 0

obs, info = env.reset(None)
for i in range(env.max_steps):
    print(obs)
    print(info)
    print()
    
    action, _states = model.predict(obs, deterministic= True)
    obs, partial_reward, terminated, truncated, info = env.step(action)
    reward += partial_reward
    rewards.append(reward)
    
    if terminated or truncated:
        obs, info = env.reset(None)

rewards2 = [0]
partial_reward2 = 0
reward2 = 0

obs, info = env2.reset(None)
for i in range(env2.max_steps):
    print(obs)
    print(info)
    print()
    
    action, _states = model.predict(obs, deterministic= True)
    obs, partial_reward2, terminated, truncated, info = env2.step(action)
    reward2 += partial_reward2
    rewards2.append(reward2)
    
    if terminated or truncated:
        obs, info = env2.reset(None)

plt.plot(rewards, rewards2)
plt.show()
    
# vec_env = model.get_env()
# obs = vec_env.reset()

# for i in range(100000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     print(obs, reward, done)
