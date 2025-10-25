from qlt import Agent
import matplotlib.pyplot as plt
import numpy as np

datapath = '../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2023.csv'
battery_capacity = 2000              # [Wh]
power_idle = 0.0                        # [W]
power_frame = 5.0                       # [W]
delta_time = 15 * 60                    # [sec]
proc_interval = 1 * 60                 # [sec]                     
pv_efficiency = 0.2
pv_area = 1.0
fps = 30
seed = "linear"
max_irradiation = 1200

battery_bins = 10
time_bins = 10
alpha = 0.05
gamma = 0.9
eps_min = 0.05
eps_dec = 0.97
eps_init = 1.0
episodes = 365

window = 10
plt.subplots(figsize=(8, 6))
plt.suptitle("Q-Learning tabular - rewards comparison")
plt.title(f"fps = {fps}, p_I = {power_idle}, p_F = {power_frame}")

plt.xlabel("Episodes")

plt.ylabel("Rewards")

for i in range(1, 6):
    agent = Agent(
                datapath,
                 battery_capacity * i,
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
    )
    
    results = agent.train()
    
    plt.plot(range(window - 1, len(results[0])), np.convolve(results[0], np.ones(window)/window, mode='valid'), label = f"{battery_capacity * (i) / 1000}kWh", alpha = 1.0)
    # plt.plot(results[0], label = f"{battery_capacity * (i+1)}kWh raw", alpha = 0.3)
    
    # plt.plot(range(window - 1, len(results[0])), np.convolve(results[0], np.ones(window)/window, mode='valid'), label = f"{battery_capacity * i / 1000}kWh smooth", alpha = 1.0)
    # plt.plot(results[0], label = f"{battery_capacity * i / 1000}kWh raw", alpha = 0.3)
    
# plt.ylim(1.1*1e6, 1.3*1e6)
plt.grid()
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
plt.tight_layout()
plt.savefig(f"battery_rewards_comparison_plot_qlt_{fps}fps.pdf")
# plt.ylim(1.1*1e6, 1.3*1e6)
# plt.savefig("battery_rewards_comparison_plot_qlt_zoom.pdf")

plt.close()


    
    