from qlt import Agent
import matplotlib.pyplot as plt
import numpy as np


# for realistic simulations, battery_capacity = 25Wh or 100Wh and fps_max = 30

# datapath = '../../../dataset/merged_2023-2024.csv'
datapath = '../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv'
# datapath = '../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2023.csv'
# datapath = '../../../dataset/csv_42.51676443693097_12.526882609673224_fixed_23_180_PT15M.csv'
battery_capacity = 25            # [Wh]
backlog_capacity = 54000
power_idle = 2.6                       # [W]
power_max = 6.0                    # [W]
delta_time = 15 * 60                    # [sec]
proc_interval = (1) * 60                 # [sec]                     
pv_efficiency = 0.2
# pv_area = 10 / (1200 * pv_efficiency)
pv_area = (1.0)
fps = 20
arrival_rate = 15

seed = "fixed_winter"
# seed = "fixed_summer"
# seed = "linear"

max_irradiation = 1000

battery_bins = 15
time_bins = 15
alpha = 0.05
gamma = 0.99
eps_min = 0.05
eps_dec = 0.997
eps_init = 1.0

episodes = 2001
# episodes = 365*3


def multiple_train(num_agents):
    window = 10
    # plt.subplots(figsize=(8, 6))
    # plt.title(f"fps = {fps}, p_I = {power_idle}, p_F = {power_max}, backlog = {backlog_capacity}")
    # plt.ylim(1000, 1500)
    # plt.title(f"fps = {fps}, p_I = {power_idle}W, p_F = {power_max}W, battery = {battery_capacity}Wh")    

    # plt.suptitle("Q-Learning tabular - rewards comparison")
    # print("title: Q-Learning tabular - rewards comparison")
    # plt.title(f"p_I = {power_idle}W, p_F = {power_max}W")        
    # plt.xlabel("Episodes")
    # plt.ylabel("Rewards")

    rewards = []

    for i in range(num_agents):
        agent = Agent(
                datapath,
                 battery_capacity * max(1, (4 * i)),
                 backlog_capacity,
                 power_idle,
                 power_max,
                 delta_time,
                 proc_interval,
                 max_irradiation,
                 pv_efficiency,
                 pv_area * (1/(i+1)),
                 fps,
                 arrival_rate,
                #  fps * (i+1),
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
        rewards.append(results[0])

        # plt.plot(range(window - 1, len(results[0])), np.convolve(results[0], np.ones(window)/window, mode='valid'), label = f"{int(battery_capacity * max(1, (4 * i)))}Wh", alpha = 1.0)        
        # plt.plot(range(window - 1, len(results[0])), np.convolve(results[0], np.ones(window)/window, mode='valid'), label = f"{backlog_capacity * i / 1000}k ", alpha = 1.0)        
        # plt.plot(range(window - 1, len(results[0])), np.convolve(results[0], np.ones(window)/window, mode='valid'), label = f"{battery_capacity * (i+1) / 1000} kWh ", alpha = 1.0)

    # rewards = []

    # for i in range(num_agents):
    #     agent = Agent(
    #             datapath,
    #              int(battery_capacity / 4),
    #              backlog_capacity,
    #              power_idle,
    #              power_max,
    #              delta_time,
    #              proc_interval,
    #              max_irradiation,
    #              pv_efficiency,
    #              pv_area,
    #              fps,
    #             #  fps * (i+1),
    #              seed,
    #              battery_bins,
    #              time_bins,
    #              alpha,
    #              gamma,
    #              eps_min,
    #              eps_dec,
    #              eps_init,
    #              episodes
    #     )

    #     results = agent.train()
    #     rewards.append(results[0])

    #     plt.plot(range(window - 1, len(results[0])), np.convolve(results[0], np.ones(window)/window, mode='valid'), label = f"{fps * (i+1)}fps - {int(battery_capacity / 4)}Wh", alpha = 1.0)        
    
    plt.suptitle("Single agent - rewards comparison")
    plt.title(f"p_I = {power_idle}W, p_F = {power_max}W")        
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    
    for i in range(0, num_agents):
        plt.plot(rewards[i], label = f"raw {int(battery_capacity * max(1, (4 * i)))}Wh", alpha = 0.3)
        plt.plot(range(window - 1, len(rewards[i])), np.convolve(rewards[i], np.ones(window)/window, mode='valid'), label = f"smooth {int(battery_capacity * max(1, (4 * i)))}Wh", alpha = 1.0)        

    plt.grid()
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
    plt.tight_layout()
    if(seed == "fixed_winter"):
        plt.savefig(f"fps_rewards_comparison_plot_qlt_356.pdf")
    elif(seed == "fixed_summer"):
        plt.savefig(f"fps_rewards_comparison_plot_qlt_173.pdf")
    else:
        plt.savefig(f"fps_rewards_comparison_plot_qlt.pdf")
    
    plt.close()

def battery_frames_rewards_train():
    window = 10
    plt.suptitle("Q-Learning tabular - rewards comparison")
    plt.title(f"p_I = {power_idle}W, p_F = {power_max}W")        
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")

    processed = []
    stored = []

    # Training agents
    for i in range(2):
        battery = battery_capacity if i == 0 else battery_capacity / 4
        
        for fps_mult in [1, 2]:
            current_fps = fps * fps_mult
            
            agent = Agent(
                datapath, battery, backlog_capacity,
                power_idle, power_max, delta_time, proc_interval,
                max_irradiation, pv_efficiency, pv_area,
                current_fps,
                seed, battery_bins, time_bins,
                alpha, gamma, eps_min, eps_dec, eps_init, episodes
            )

            results = agent.train()
            
            # results = [rewards, dropped, processed, battery, irradiance, backlog]
            #            [0]      [1]      [2]        [3]      [4]         [5]
            
            label = f"{current_fps}fps - {battery}Wh"
            
            processed.append({label: results[2]})
            stored.append({label: results[5]})
            
            # Plot rewards
            plt.plot(
                range(window - 1, len(results[0])), 
                np.convolve(results[0], np.ones(window)/window, mode='valid'), 
                label=label, 
                alpha=1.0
            )

    plt.grid()
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
    plt.tight_layout()
    plt.savefig(f"fps_rewards_comparison_plot_qlt_355.pdf")
    plt.close()
    
    # Plot processed frames
    plt.figure()
    plt.suptitle("Q-Learning tabular - Frames processed")
    plt.title(f"p_I = {power_idle}W, p_F = {power_max}W")
    plt.xlabel("Episodes")
    plt.ylabel("Frames processed")
    
    for item in processed:
        for label, data in item.items():
            plt.plot(
                range(window - 1, len(data)), 
                np.convolve(data, np.ones(window)/window, mode='valid'),
                label=label,
                alpha=1.0
            )
    
    plt.grid()
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
    plt.tight_layout()
    plt.savefig(f"fps_processed_comparison_plot_qlt_355.pdf")
    plt.close()
    
    # Plot backlog
    plt.figure()
    plt.suptitle("Q-Learning tabular - backlog level")
    plt.title(f"p_I = {power_idle}W, p_F = {power_max}W")
    plt.xlabel("Episodes")
    plt.ylabel("backlog level")
    
    for item in stored:
        for label, data in item.items():
            plt.plot(
                range(window - 1, len(data)), 
                np.convolve(data, np.ones(window)/window, mode='valid'),
                label=label,
                alpha=1.0
            )
    
    plt.grid()
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
    plt.tight_layout()
    plt.savefig(f"fps_backlog_comparison_plot_qlt_355.pdf")
    plt.close()

def single_train():
    window = 10
    # plt.subplots(figsize=(8, 6))
    rewards = []

    agent = Agent(
            datapath,
            battery_capacity,
            backlog_capacity,
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
    )

    results = agent.train()
    rewards.append(results[0])
    
    # seed = "fixed_winter"
    
    # new_res = agent.train()
    # for elem in new_res:
    #     results.append(elem)

    plt.suptitle("Q-Learning tabular - rewards")
    plt.title(f"B = {battery_capacity}, fps = {fps}, p_I = {power_idle}, p_F = {power_max}")

    plt.xlabel("Episodes")
    plt.ylabel("Rewards")

    plt.plot(range(window - 1, len(results[0])), np.convolve(results[0], np.ones(window)/window, mode='valid'), label = "smmoth", alpha = 1.0)
    plt.plot(results[0], label = "raw", alpha = 0.3)
    
    plt.grid()
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
    plt.tight_layout()
    plt.savefig(f"rewards_{battery_capacity}Wh_{fps}fps_qlt_355.pdf")
    plt.close()
    
    # agent.plot_battery(results[3])
    agent.plot_processed_backlog(results[2], results[5])
    # agent.plot_frames(results[1], results[2])

multiple_train(2)
# single_train()
# battery_frames_rewards_train()

# window = 10
# plt.subplots(figsize=(8, 6))
# plt.suptitle("Q-Learning tabular - rewards comparison")
# plt.title(f"fps = {fps}, p_I = {power_idle}, p_F = {power_max}")

# plt.xlabel("Episodes")

# plt.ylabel("Rewards")

# agent = Agent(
#                 datapath,
#                  battery_capacity,
#                  power_idle,
#                  power_max,
#                  delta_time,
#                  proc_interval,
#                  max_irradiation,
#                  pv_efficiency,
#                  pv_area,
#                  fps,
#                  seed,
#                  battery_bins,
#                  time_bins,
#                  alpha,
#                  gamma,
#                  eps_min,
#                  eps_dec,
#                  eps_init,
#                  episodes
#     )

# results = agent.train()

# agent.plot_rewards(results[0])

# for i in range(1, 6):
#     agent = Agent(
#                 datapath,
#                  battery_capacity * i,
#                  power_idle,
#                  power_max,
#                  delta_time,
#                  proc_interval,
#                  max_irradiation,
#                  pv_efficiency,
#                  pv_area,
#                  fps,
#                  seed,
#                  battery_bins,
#                  time_bins,
#                  alpha,
#                  gamma,
#                  eps_min,
#                  eps_dec,
#                  eps_init,
#                  episodes
#     )
    
#     results = agent.train()
    
#     plt.plot(range(window - 1, len(results[0])), np.convolve(results[0], np.ones(window)/window, mode='valid'), label = f"{battery_capacity * (i) / 1000}kWh", alpha = 1.0)
#     # plt.plot(results[0], label = f"{battery_capacity * (i+1)}kWh raw", alpha = 0.3)
    
#     # plt.plot(range(window - 1, len(results[0])), np.convolve(results[0], np.ones(window)/window, mode='valid'), label = f"{battery_capacity * i / 1000}kWh smooth", alpha = 1.0)
#     # plt.plot(results[0], label = f"{battery_capacity * i / 1000}kWh raw", alpha = 0.3)
    
# # plt.ylim(1.1*1e6, 1.3*1e6)
# plt.grid()
# plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
# plt.tight_layout()
# plt.savefig(f"battery_rewards_comparison_plot_qlt_{fps}fps_355.pdf")
# # plt.ylim(1.1*1e6, 1.3*1e6)
# # plt.savefig("battery_rewards_comparison_plot_qlt_zoom_355.pdf")

# plt.close()


    
    