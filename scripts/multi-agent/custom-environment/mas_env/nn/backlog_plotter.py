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

def double_compute_avg_backlog_daily(episodes, day, interval, num_agents, conf1, conf2):
    
    backlog1_samples = []
    backlog2_samples = []
    
    backlog1_all_agents = []
    for agent in range(num_agents):
        with open(f'./{conf1}/csvs/csvs_batch_256/backlog_{battery_capacities[agent]}_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
            csvFile = csv.reader(file)
            next(csvFile)
            
            agent_episodes = []
            for line in csvFile:
                timesteps = [float(val) for val in line]
                episode_mean = np.mean(timesteps)
                agent_episodes.append(episode_mean)
            
            backlog1_all_agents.append(agent_episodes)
    
    backlog1_all_agents = np.array(backlog1_all_agents)
    backlog1_samples = np.mean(backlog1_all_agents, axis=0)
    
    backlog2_all_agents = []
    for agent in range(num_agents):
        with open(f'./{conf2}/csvs/csvs_batch_256/backlog_{battery_capacities[agent]}_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
            csvFile = csv.reader(file)
            next(csvFile)
            
            agent_episodes = []
            for line in csvFile:
                timesteps = [float(val) for val in line]
                episode_mean = np.mean(timesteps)
                agent_episodes.append(episode_mean)
            
            backlog2_all_agents.append(agent_episodes)
    
    backlog2_all_agents = np.array(backlog2_all_agents)
    backlog2_samples = np.mean(backlog2_all_agents, axis=0)
    
    plt.suptitle("Average backlog (sampled episodes)")
    plt.title(f"Episodes: {episodes}, Day: {day}, Interval: {interval}, num_agents: {num_agents}, Mode: cuda")
    
    sample_episodes = [i * int(episodes / 10) for i in range(len(backlog1_samples))]
    
    plt.xlabel("Episode")
    plt.ylabel("Average Backlog")
    
    plt.plot(sample_episodes, backlog1_samples, 'o-', label=f"{conf1}", alpha=1.0, markersize=8, linewidth=2)
    plt.plot(sample_episodes, backlog2_samples, 's-', label=f"{conf2}", alpha=1.0, markersize=8, linewidth=2)

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./comparisons/backlog/average_backlog_sampled_{episodes}_{day}_{interval}_{num_agents}agents_{conf1}_{conf2}_cuda.pdf")
    plt.close()


def triple_compute_avg_backlog_daily(episodes, day, interval, num_agents, conf1, conf2, conf3):
    backlog1_samples = []
    backlog2_samples = []
    backlog3_samples = []
    
    backlog1_all_agents = []
    for agent in range(num_agents):
        with open(f'./{conf1}/csvs/csvs_batch_256/backlog_{battery_capacities[agent]}_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
            csvFile = csv.reader(file)
            next(csvFile)
            
            agent_episodes = []
            for line in csvFile:
                timesteps = [float(val) for val in line]
                episode_mean = np.mean(timesteps)
                agent_episodes.append(episode_mean)
            
            backlog1_all_agents.append(agent_episodes)
    
    backlog1_all_agents = np.array(backlog1_all_agents)
    backlog1_samples = np.mean(backlog1_all_agents, axis=0)
    
    backlog2_all_agents = []
    for agent in range(num_agents):
        with open(f'./{conf2}/csvs/csvs_batch_256/backlog_{battery_capacities[agent]}_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
            csvFile = csv.reader(file)
            next(csvFile)
            
            agent_episodes = []
            for line in csvFile:
                timesteps = [float(val) for val in line]
                episode_mean = np.mean(timesteps)
                agent_episodes.append(episode_mean)
            
            backlog2_all_agents.append(agent_episodes)
    
    backlog2_all_agents = np.array(backlog2_all_agents)
    backlog2_samples = np.mean(backlog2_all_agents, axis=0)
    
    backlog3_all_agents = []
    for agent in range(num_agents):
        with open(f'./{conf3}/csvs/csvs_batch_256/backlog_{battery_capacities[agent]}_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
            csvFile = csv.reader(file)
            next(csvFile)
            
            agent_episodes = []
            for line in csvFile:
                timesteps = [float(val) for val in line]
                episode_mean = np.mean(timesteps)
                agent_episodes.append(episode_mean)
            
            backlog3_all_agents.append(agent_episodes)
    
    backlog3_all_agents = np.array(backlog3_all_agents)
    backlog3_samples = np.mean(backlog3_all_agents, axis=0)
    
    plt.suptitle("Average backlog (sampled episodes)")
    plt.title(f"Episodes: {episodes}, Day: {day}, Interval: {interval}, num_agents: {num_agents}, Mode: cuda")
    
    sample_episodes = [i * int(episodes / 10) for i in range(len(backlog1_samples))]
    
    plt.xlabel("Episode")
    plt.ylabel("Average Backlog")
    
    plt.plot(sample_episodes, backlog1_samples, 'o-', label=f"{conf1}", alpha=1.0, markersize=8, linewidth=2)
    plt.plot(sample_episodes, backlog2_samples, 's-', label=f"{conf2}", alpha=1.0, markersize=8, linewidth=2)
    plt.plot(sample_episodes, backlog3_samples, '^-', label=f"{conf3}", alpha=1.0, markersize=8, linewidth=2)

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./comparisons/backlog/average_backlog_sampled_{episodes}_{day}_{interval}_{num_agents}agents_{conf1}_{conf2}_{conf3}_cuda.pdf")
    plt.close()
    

def triple_plot_backlog_evolution_daily(episodes, day, interval, num_agents, conf1, conf2, conf3):
    
    configs = [conf1, conf2, conf3]
    colors = {'aggregated_states': 'blue', 'reduced_states': 'green', 'local_only': 'red'}
    
    for config_name in configs:
        
        all_agents_data = []
        for agent in range(num_agents):
            with open(f'./{config_name}/csvs/csvs_batch_256/backlog_{battery_capacities[agent]}_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
                csvFile = csv.reader(file)
                next(csvFile)
                
                agent_episodes = []
                for line in csvFile:
                    timesteps = [float(val) for val in line]
                    agent_episodes.append(timesteps)
                
                all_agents_data.append(agent_episodes)
        
        all_agents_data = np.array(all_agents_data)
        mean_data = np.mean(all_agents_data, axis=0)
        
        plt.suptitle(f"Daily Backlog Evolution - {config_name.upper()}")
        plt.title(f"Episodes: {episodes}, num_agents: {num_agents}")
        plt.xlabel("Timestep (minutes)")
        plt.ylabel("Average Backlog")
        
        window = 40
        for i, episode_data in enumerate(mean_data):
            episode_num = i * int(episodes / 10)
            
            smoothed = np.convolve(episode_data, np.ones(window)/window, mode='valid')
            
            plt.plot(
                range(window - 1, len(episode_data)), 
                smoothed, 
                label=f"Episode {episode_num}",
                alpha=0.8,
                linewidth=1.5
            )
        
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./comparisons/backlog/daily_evolution_{config_name}_{episodes}_{day}_{interval}_{num_agents}agents_cuda.pdf")
        plt.close()


double_compute_avg_backlog_daily(4000, 355, 60, 5, "reduced_states", "sigmoid_z_score")

# triple_compute_avg_backlog_daily(4000, 355, 60, 5, "aggregated_states", "reduced_states", "local_only")
# triple_plot_backlog_evolution_daily(4000, 355, 60, 5, "aggregated_states", "reduced_states", "local_only")

