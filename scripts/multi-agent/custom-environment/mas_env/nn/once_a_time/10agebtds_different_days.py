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

days = [172, 355]

def compute_avg_rewards(episodes, interval, num_agents):
    rewards = [[0 for i in range(episodes+1)] for i in range(len(days))]

    for i in range(len(days)):    
        for agent in range(num_agents):
            with open(f'./csv_different_days/{days[i]}/rewards_agent_{battery_capacities[agent]}_{episodes}_{days[i]}_{interval}_{num_agents}agents_cuda.csv') as file:
                csvFile = csv.reader(file)
                cnt = 0
                
                for line in csvFile:
                    if(cnt < episodes):
                        rewards[i][cnt] += float(line[0])
                        cnt += 1
                
        for id in range(episodes):
            rewards[i][id] /= num_agents
                
    window = 10
    plt.suptitle("Average rewards")
    plt.title(f"Episodes: {episodes}, Interval: {interval}, num_agents: {num_agents}, Mode: cuda")
    
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    
    for i in range(len(days)):
        plt.plot(range(window - 1, len(rewards[i])), np.convolve(rewards[i], np.ones(window)/window, mode='valid'), label = f"{days[i]}", linewidth = 2.0,  alpha = 1.0)

    plt.grid()
    plt.legend()
    # plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    # plt.tight_layout()

    plt.savefig(f"./comparisons/diff_days_average_rewards_episodes_{episodes}_{interval}_{num_agents}agents_cuda.pdf")
    plt.close()

def compute_avg_matchings(episodes, day, interval, num_agents):
    rewards = [[0 for i in range(episodes+1)] for i in range(len(days))]

    for i in range(len(days)):    
        for agent in range(num_agents):
            with open(f'./csv_different_days/{days[i]}/matchings_agent_{battery_capacities[agent]}_{episodes}_{days[i]}_{interval}_{num_agents}agents_cuda.csv') as file:
                csvFile = csv.reader(file)
                cnt = 0
                
                for line in csvFile:
                    if(cnt < episodes):
                        rewards[i][cnt] += float(line[0])
                        cnt += 1
                
        for id in range(episodes):
            rewards[i][id] /= num_agents
                
    window = 10
    plt.suptitle("Average rewards")
    plt.title(f"Episodes: {episodes}, Interval: {interval}, num_agents: {num_agents}, Mode: cuda")
    
    plt.xlabel("Episodes")
    plt.ylabel("Matchings")
    
    for i in range(len(days)):
        plt.plot(range(window - 1, len(rewards[i])), np.convolve(rewards[i], np.ones(window)/window, mode='valid'), label = f"{days[i]}", linewidth = 2.0,  alpha = 1.0)

    plt.grid()
    plt.legend()
    # plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    # plt.tight_layout()

    plt.savefig(f"./comparisons/diff_days_average_rewards_episodes_{episodes}_{interval}_{num_agents}agents_cuda.pdf")
    plt.close()


def compute_avg_framerates(episodes, interval, num_agents):
    rewards = [[0 for i in range(episodes+1)] for i in range(len(days))]

    for i in range(len(days)):    
        for agent in range(num_agents):
            with open(f'./csv_different_days/{days[i]}/total_framerate_{battery_capacities[agent]}_{episodes}_{days[i]}_{interval}_{num_agents}agents_cuda.csv') as file:
                csvFile = csv.reader(file)
                cnt = 0
                
                for line in csvFile:
                    if(cnt < episodes):
                        rewards[i][cnt] += float(line[0])
                        cnt += 1
                
        for id in range(episodes):
            rewards[i][id] /= num_agents
                
    window = 10
    plt.suptitle("Average rewards")
    plt.title(f"Episodes: {episodes}, Interval: {interval}, num_agents: {num_agents}, Mode: cuda")
    
    plt.xlabel("Episodes")
    plt.ylabel("Framerate")
    
    for i in range(len(days)):
        plt.plot(range(window - 1, len(rewards[i])), np.convolve(rewards[i], np.ones(window)/window, mode='valid'), label = f"{days[i]}", linewidth = 2.0,  alpha = 1.0)

    plt.grid()
    plt.legend()
    # plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    # plt.tight_layout()

    plt.savefig(f"./comparisons/diff_days_average_framerate_episodes_{episodes}_{interval}_{num_agents}agents_cuda.pdf")
    plt.close()
def compute_avg_backlog(episodes, interval, num_agents):
    samplings = 10
    sample_episodes = [i * int(episodes / 10) for i in range(samplings)]

    # { day: array di samplings valori medi }
    results = {}

    for day in days:
        all_agents = []

        for agent in range(num_agents):
            with open(f'./csv_different_days/{day}/backlog_{battery_capacities[agent]}_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
                csvFile = csv.reader(file)
                next(csvFile)  # salta header

                agent_episodes = []
                for line in csvFile:
                    timesteps = [float(val) for val in line]
                    agent_episodes.append(np.mean(timesteps))

                all_agents.append(agent_episodes)

        # media tra agenti: shape (num_agents, samplings) → (samplings,)
        results[day] = np.mean(np.array(all_agents), axis=0)

    plt.suptitle("Average backlog (sampled episodes)")
    plt.title(f"Episodes: {episodes}, Interval: {interval}, num_agents: {num_agents}, Mode: cuda")
    plt.xlabel("Episode")
    plt.ylabel("Average Backlog")

    for day, data in results.items():
        plt.plot(sample_episodes, data, 'o-', label=f"day {day}",
                 alpha=1.0, markersize=8, linewidth=2)

    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(f"./comparisons/diff_days_average_backlog_{episodes}_{interval}_{num_agents}agents_cuda.pdf")
    plt.close()


def compute_avg_battery(episodes, interval, num_agents):
    samplings = 10
    sample_episodes = [i * int(episodes / 10) for i in range(samplings)]

    results = {}

    for day in days:
        all_agents = []

        for agent in range(num_agents):
            with open(f'./csv_different_days/{day}/battery_{battery_capacities[agent]}_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
                csvFile = csv.reader(file)
                next(csvFile)  # salta header

                agent_episodes = []
                for line in csvFile:
                    timesteps = [float(val) for val in line]
                    agent_episodes.append(np.mean(timesteps))

                all_agents.append(agent_episodes)

        results[day] = np.mean(np.array(all_agents), axis=0)

    plt.suptitle("Average battery (sampled episodes)")
    plt.title(f"Episodes: {episodes}, Interval: {interval}, num_agents: {num_agents}, Mode: cuda")
    plt.xlabel("Episode")
    plt.ylabel("Average Battery")

    for day, data in results.items():
        plt.plot(sample_episodes, data, 'o-', label=f"day {day}",
                 alpha=1.0, markersize=8, linewidth=2)

    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(f"./comparisons/diff_days_average_battery_{episodes}_{interval}_{num_agents}agents_cuda.pdf")
    plt.close()

compute_avg_rewards(999, 60, 10)
compute_avg_framerates(999, 60, 10)
compute_avg_backlog(999, 60, 10)
compute_avg_battery(999, 60, 10)
# compute_avg_framerate(1000, 355, 60, 10)
# compute_avg_backlog(1000, 355, 60, 10)
# compute_avg_battery(1000, 355, 60, 10)
# compute_avg_matchings(1000, 355, 60, 10)