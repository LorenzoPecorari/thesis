from qlt import Agent
import matplotlib.pyplot as plt
import numpy as np


# for realistic simulations, battery_capacity = 25Wh or 100Wh and fps_max = 30

datapath = '../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2023.csv'
battery_capacity = 100             # [Wh]
storage_capacity = 54000
power_idle = 0.0                        # [W]
power_frame = 5.0                       # [W]
delta_time = 15 * 60                    # [sec]
proc_interval = (1/12) * 60                 # [sec]                     
pv_efficiency = 0.2
pv_area = 1.0
fps = 15
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

def multiple_train(num_agents):
    window = 10
    # plt.subplots(figsize=(8, 6))
    plt.suptitle("Q-Learning tabular - rewards comparison")

    plt.title(f"p_I = {power_idle}W, p_F = {power_frame}W, battery = {battery_capacity}Wh")        
    # plt.title(f"fps = {fps}, p_I = {power_idle}W, p_F = {power_frame}W, battery = {battery_capacity}Wh")    
    # plt.title(f"fps = {fps}, p_I = {power_idle}, p_F = {power_frame}, storage = {storage_capacity}")
    # plt.ylim(1000, 1500)

    plt.xlabel("Episodes")
    plt.ylabel("Rewards")

    rewards = []

    for i in range(num_agents):
        agent = Agent(
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
                 fps * (i+1),
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

        plt.plot(range(window - 1, len(results[0])), np.convolve(results[0], np.ones(window)/window, mode='valid'), label = f"{fps * (i+1)}fps - {battery_capacity}Wh", alpha = 1.0)        
        # plt.plot(range(window - 1, len(results[0])), np.convolve(results[0], np.ones(window)/window, mode='valid'), label = f"{storage_capacity * i / 1000}k ", alpha = 1.0)        
        # plt.plot(range(window - 1, len(results[0])), np.convolve(results[0], np.ones(window)/window, mode='valid'), label = f"{battery_capacity * (i+1) / 1000} kWh ", alpha = 1.0)

    rewards = []

    for i in range(num_agents):
        agent = Agent(
                datapath,
                 battery_capacity / 4,
                 storage_capacity,
                 power_idle,
                 power_frame,
                 delta_time,
                 proc_interval,
                 max_irradiation,
                 pv_efficiency,
                 pv_area,
                 fps * (i+1),
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

        plt.plot(range(window - 1, len(results[0])), np.convolve(results[0], np.ones(window)/window, mode='valid'), label = f"{fps * (i+1)}fps - {battery_capacity / 4}Wh", alpha = 1.0)        


    plt.grid()
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
    plt.tight_layout()
    plt.savefig(f"fps_rewards_comparison_plot_qlt.pdf")
    # plt.savefig(f"epsilon_rewards_comparison_plot_qlt_{fps}fps_{battery_capacity / 1000}kWh.pdf")
    # plt.savefig(f"storage_rewards_comparison_plot_qlt_{fps}fps_{battery_capacity / 1000}kWh.pdf")
    # plt.savefig(f"battery_rewards_comparison_plot_qlt_{fps}fps_{storage_capacity / 1000}k.pdf")
    plt.close()

def battery_frames_rewards_train():
    window = 10
    # plt.subplots(figsize=(8, 6))
    plt.suptitle("Q-Learning tabular - rewards comparison")

    plt.title(f"p_I = {power_idle}W, p_F = {power_frame}W")        
    # plt.title(f"fps = {fps}, p_I = {power_idle}W, p_F = {power_frame}W, battery = {battery_capacity}Wh")    
    # plt.title(f"fps = {fps}, p_I = {power_idle}, p_F = {power_frame}, storage = {storage_capacity}")
    # plt.ylim(1000, 1500)

    plt.xlabel("Episodes")
    plt.ylabel("Rewards")

    # rewards = []
    dropped = []
    processed = []
    stored = []

    # training agents and plotting rewards
    for i in range(2):
        battery = battery_capacity
        if(i != 0):
            battery = battery_capacity / 4
        
        agent = Agent(
                datapath,
                 battery,
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
        )

        results = agent.train()
        dropped.append({f"{battery}Wh - {fps}fps": results[1]})
        processed.append({f"{battery}Wh - {fps}fps": results[2]})
        stored.append({f"{battery}Wh - {fps}fps": results[5]})
        plt.plot(range(window - 1, len(results[0])), np.convolve(results[0], np.ones(window)/window, mode='valid'), label = f"{fps}fps - {battery}Wh", alpha = 1.0)        


        agent = Agent(
                datapath,
                 battery,
                 storage_capacity,
                 power_idle,
                 power_frame,
                 delta_time,
                 proc_interval,
                 max_irradiation,
                 pv_efficiency,
                 pv_area,
                 fps * 2,
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
        # rewards.append({f"{battery_capacity}Wh - {fps*(i+1)}": results[0]})
        dropped.append({f"{battery}Wh - {fps*2}fps": results[1]})
        processed.append({f"{battery}Wh - {fps*2}fps": results[2]})
        stored.append({f"{battery}Wh - {fps*2}fps": results[5]})
        plt.plot(range(window - 1, len(results[0])), np.convolve(results[0], np.ones(window)/window, mode='valid'), label = f"{fps * 2}fps - {battery}Wh", alpha = 1.0)        


        # plt.plot(range(window - 1, len(results[0])), np.convolve(results[0], np.ones(window)/window, mode='valid'), label = f"{storage_capacity * i / 1000}k ", alpha = 1.0)        
        # plt.plot(range(window - 1, len(results[0])), np.convolve(results[0], np.ones(window)/window, mode='valid'), label = f"{battery_capacity * (i+1) / 1000} kWh ", alpha = 1.0)

    plt.grid()
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
    plt.tight_layout()
    plt.savefig(f"fps_rewards_comparison_plot_qlt.pdf")
    # plt.savefig(f"epsilon_rewards_comparison_plot_qlt_{fps}fps_{battery_capacity / 1000}kWh.pdf")
    # plt.savefig(f"storage_rewards_comparison_plot_qlt_{fps}fps_{battery_capacity / 1000}kWh.pdf")
    # plt.savefig(f"battery_rewards_comparison_plot_qlt_{fps}fps_{storage_capacity / 1000}k.pdf")
    plt.close()

    # plotting dropped and processed frames
    plt.suptitle("Q-Learning tabular - frames comparison")
    plt.title(f"p_I = {power_idle}W, p_F = {power_frame}W")        
    plt.xlabel("Episodes")
    plt.ylabel("Frames")

    keys = []
    for elem in dropped:
        K = elem.keys()
        for k in K:
            keys.append(k)

    print(len(dropped))        

    for i in range(len(keys)):
        plt.plot(range(window - 1, len(dropped[i][keys[i]])), np.convolve(dropped[i][keys[i]], np.ones(window)/window, mode='valid'), label = f"drop {keys[i]}", alpha = 1.0)
        plt.plot(range(window - 1, len(processed[i][keys[i]])), np.convolve(processed[i][keys[i]], np.ones(window)/window, mode='valid'), label = f"proc {keys[i]}", alpha = 1.0)

    plt.grid()
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
    plt.tight_layout()
    plt.savefig(f"fps_frames_comparison_plot_qlt.pdf")
    plt.close()

    # plotting storage uses
    plt.suptitle("Q-Learning tabular - storage comparison")
    plt.title(f"p_I = {power_idle}W, p_F = {power_frame}W")        
    plt.xlabel("Episodes")
    plt.ylabel("Storage level")

    for i in range(len(keys)):
        plt.plot(range(window - 1, len(stored[i][keys[i]])), np.convolve(stored[i][keys[i]], np.ones(window)/window, mode='valid'), label = f"stored {keys[i]}", alpha = 1.0)

    plt.grid()
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
    plt.tight_layout()
    plt.savefig(f"storage_comparison_plot_qlt.pdf")


def single_train():
    window = 10
    # plt.subplots(figsize=(8, 6))
    plt.suptitle("Q-Learning tabular - rewards")
    plt.title(f"B = {battery_capacity}, fps = {fps}, p_I = {power_idle}, p_F = {power_frame}")

    plt.xlabel("Episodes")

    plt.ylabel("Rewards")

    rewards = []

    agent = Agent(
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
    )

    results = agent.train()
    rewards.append(results[0])
        
    plt.plot(results[0], label = "raw", alpha = 0.3)
    plt.plot(range(window - 1, len(results[0])), np.convolve(results[0], np.ones(window)/window, mode='valid'), label = "smmoth", alpha = 1.0)

    plt.grid()
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
    plt.tight_layout()
    plt.savefig(f"rewards_storage_qlt_{fps}fps.pdf")
    plt.close()
    
    agent.plot_frames(results[1], results[2])

# multiple_train(2)
# single_train()
battery_frames_rewards_train()

# window = 10
# plt.subplots(figsize=(8, 6))
# plt.suptitle("Q-Learning tabular - rewards comparison")
# plt.title(f"fps = {fps}, p_I = {power_idle}, p_F = {power_frame}")

# plt.xlabel("Episodes")

# plt.ylabel("Rewards")

# agent = Agent(
#                 datapath,
#                  battery_capacity,
#                  power_idle,
#                  power_frame,
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
#                  power_frame,
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
# plt.savefig(f"battery_rewards_comparison_plot_qlt_{fps}fps.pdf")
# # plt.ylim(1.1*1e6, 1.3*1e6)
# # plt.savefig("battery_rewards_comparison_plot_qlt_zoom.pdf")

# plt.close()


    
    