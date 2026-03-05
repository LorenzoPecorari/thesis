import numpy as np
import matplotlib.pyplot as plt
import csv

battery_capacities = [
    25, 
    100,  
    50,   
    37,   
    65,   
    80,   
    40,   
    75,   
    55,   
    90    
]


def compute_avg_rewards(episodes, day, interval, num_agents, conf1, conf2):
    rewards1 = [0 for episode in range(episodes+1)]
    rewards2 = [0 for episode in range(episodes+1)]
    
    
    for agent in range(num_agents):
        with open(f'./{conf1}/csvs/csvs_batch_256/rewards_agent_{battery_capacities[agent]}_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
            csvFile = csv.reader(file)
            cnt = 0
            
            for line in csvFile:
                rewards1[cnt] += float(line[0])
                cnt += 1
            
    for id in range(episodes+1):
        rewards1[id] /= num_agents

    for agent in range(num_agents):
        with open(f'./{conf2}/csvs/csvs_batch_256/rewards_agent_{battery_capacities[agent]}_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
            csvFile = csv.reader(file)
            cnt = 0
            
            for line in csvFile:
                rewards2[cnt] += float(line[0])
                cnt += 1
            
    for id in range(episodes+1):
        rewards2[id] /= num_agents
             
    window = 10
    plt.suptitle("Average rewards")
    plt.title(f"Episodes: {episodes}, Day: {day}, Interval: {interval}, num_agents: {num_agents}, Mode: cuda")
    
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    
    plt.plot(range(window - 1, len(rewards1)), np.convolve(rewards1, np.ones(window)/window, mode='valid'), label = f"{conf1}", alpha = 1.0)
    plt.plot(range(window - 1, len(rewards2)), np.convolve(rewards2, np.ones(window)/window, mode='valid'), label = f"{conf2}", alpha = 1.0)

    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./comparisons/rewards/average_episodes_{episodes}_{day}_{interval}_{num_agents}agents_{conf1}_{conf2}_cuda.pdf")
    plt.close()

def triple_compute_avg_rewards(episodes, day, interval, num_agents, conf1, conf2, conf3):
    rewards1 = [0 for episode in range(episodes+1)]
    rewards2 = [0 for episode in range(episodes+1)]
    rewards3 = [0 for episode in range(episodes+1)]
    
    for agent in range(num_agents):
        with open(f'./{conf1}/csvs/csvs_batch_256/rewards_agent_{battery_capacities[agent]}_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
            csvFile = csv.reader(file)
            cnt = 0
            
            for line in csvFile:
                rewards1[cnt] += float(line[0])
                cnt += 1
            
    for id in range(episodes+1):
        rewards1[id] /= num_agents

    for agent in range(num_agents):
        with open(f'./{conf2}/csvs/csvs_batch_256/rewards_agent_{battery_capacities[agent]}_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
            csvFile = csv.reader(file)
            cnt = 0
            
            for line in csvFile:
                rewards2[cnt] += float(line[0])
                cnt += 1
            
    for id in range(episodes+1):
        rewards2[id] /= num_agents
        
    for agent in range(num_agents):
        with open(f'./{conf3}/csvs/csvs_batch_256/rewards_agent_{battery_capacities[agent]}_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
            csvFile = csv.reader(file)
            cnt = 0
            
            for line in csvFile:
                rewards3[cnt] += float(line[0])
                cnt += 1
            
    for id in range(episodes+1):
        rewards3[id] /= num_agents
             
    window = 10
    plt.suptitle("Average rewards")
    plt.title(f"Episodes: {episodes}, Day: {day}, Interval: {interval}, num_agents: {num_agents}, Mode: cuda")
    
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    
    plt.plot(range(window - 1, len(rewards1)), np.convolve(rewards1, np.ones(window)/window, mode='valid'), label = f"{conf1}", alpha = 1.0)
    plt.plot(range(window - 1, len(rewards2)), np.convolve(rewards2, np.ones(window)/window, mode='valid'), label = f"{conf2}", alpha = 1.0)
    plt.plot(range(window - 1, len(rewards3)), np.convolve(rewards3, np.ones(window)/window, mode='valid'), label = f"{conf3}", alpha = 1.0)

    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./comparisons/rewards/average_episodes_{episodes}_{day}_{interval}_{num_agents}agents_{conf1}_{conf2}_{conf3}_cuda.pdf")
    plt.close()

            
compute_avg_rewards(4000, 355, 60, 5, "reduced_states", "tanh_z_score")
# triple_compute_avg_rewards(4000, 355, 60, 5, "aggregated_states", "reduced_states", "local_only")
