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

def double_compute_avg_battery_daily(episodes, day, interval, num_agents, conf1, conf2):
    """
    Plotta battery medio dei 12 episodi campionati
    Ogni episodio ha 1440 timesteps (24h * 60min)
    """
    
    # Liste per memorizzare media battery di ogni episodio campionato
    battery1_samples = []
    battery2_samples = []
    
    # ==== CONFIG 1 ====
    battery1_all_agents = []
    for agent in range(num_agents):
        with open(f'./{conf1}/csvs/csvs_batch_256/battery_{battery_capacities[agent]}_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
            csvFile = csv.reader(file)
            next(csvFile)  # Skip header
            
            agent_episodes = []
            for line in csvFile:
                # Ogni riga = 1 episodio con 1440 timesteps
                timesteps = [float(val) for val in line]
                # Calcola media di questo episodio
                episode_mean = np.mean(timesteps)
                agent_episodes.append(episode_mean)
            
            battery1_all_agents.append(agent_episodes)
    
    # Media cross-agent per ogni episodio campionato
    battery1_all_agents = np.array(battery1_all_agents)  # Shape: [num_agents, 12]
    battery1_samples = np.mean(battery1_all_agents, axis=0)  # Shape: [12]
    
    # ==== CONFIG 2 ====
    battery2_all_agents = []
    for agent in range(num_agents):
        with open(f'./{conf2}/csvs/csvs_batch_256/battery_{battery_capacities[agent]}_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
            csvFile = csv.reader(file)
            next(csvFile)
            
            agent_episodes = []
            for line in csvFile:
                timesteps = [float(val) for val in line]
                episode_mean = np.mean(timesteps)
                agent_episodes.append(episode_mean)
            
            battery2_all_agents.append(agent_episodes)
    
    battery2_all_agents = np.array(battery2_all_agents)
    battery2_samples = np.mean(battery2_all_agents, axis=0)
    
    # ==== PLOT ====
    plt.suptitle("Average battery (sampled episodes)")
    plt.title(f"Episodes: {episodes}, Day: {day}, Interval: {interval}, num_agents: {num_agents}, Mode: cuda")
    
    # X-axis: episodio effettivo (0, 400, 800, ..., 4000)
    sample_episodes = [i * int(episodes / 10) for i in range(len(battery1_samples))]
    
    plt.xlabel("Episode")
    plt.ylabel("Average battery")
    
    plt.plot(sample_episodes, battery1_samples, 'o-', label=f"{conf1}", alpha=1.0, markersize=8, linewidth=2)
    plt.plot(sample_episodes, battery2_samples, 's-', label=f"{conf2}", alpha=1.0, markersize=8, linewidth=2)

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./comparisons/battery/average_battery_sampled_{episodes}_{day}_{interval}_{num_agents}agents_{conf1}_{conf2}_cuda.pdf")
    plt.close()
    
    print(f"✅ Plotted {len(battery1_samples)} sampled episodes")
    print(f"   Episode numbers: {sample_episodes}")
    print(f"\n📊 Final values:")
    print(f"   {conf1}: {battery1_samples[-1]:.2f}")
    print(f"   {conf2}: {battery2_samples[-1]:.2f}")


def triple_compute_avg_battery_daily(episodes, day, interval, num_agents, conf1, conf2, conf3):
    """
    Plotta battery medio dei 12 episodi campionati
    Ogni episodio ha 1440 timesteps (24h * 60min)
    """
    
    # Liste per memorizzare media battery di ogni episodio campionato
    battery1_samples = []
    battery2_samples = []
    battery3_samples = []
    
    # ==== CONFIG 1 ====
    battery1_all_agents = []
    for agent in range(num_agents):
        with open(f'./{conf1}/csvs/csvs_batch_256/battery_{battery_capacities[agent]}_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
            csvFile = csv.reader(file)
            next(csvFile)  # Skip header
            
            agent_episodes = []
            for line in csvFile:
                # Ogni riga = 1 episodio con 1440 timesteps
                timesteps = [float(val) for val in line]
                # Calcola media di questo episodio
                episode_mean = np.mean(timesteps)
                agent_episodes.append(episode_mean)
            
            battery1_all_agents.append(agent_episodes)
    
    # Media cross-agent per ogni episodio campionato
    battery1_all_agents = np.array(battery1_all_agents)  # Shape: [num_agents, 12]
    battery1_samples = np.mean(battery1_all_agents, axis=0)  # Shape: [12]
    
    # ==== CONFIG 2 ====
    battery2_all_agents = []
    for agent in range(num_agents):
        with open(f'./{conf2}/csvs/csvs_batch_256/battery_{battery_capacities[agent]}_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
            csvFile = csv.reader(file)
            next(csvFile)
            
            agent_episodes = []
            for line in csvFile:
                timesteps = [float(val) for val in line]
                episode_mean = np.mean(timesteps)
                agent_episodes.append(episode_mean)
            
            battery2_all_agents.append(agent_episodes)
    
    battery2_all_agents = np.array(battery2_all_agents)
    battery2_samples = np.mean(battery2_all_agents, axis=0)
    
    # ==== CONFIG 3 ====
    battery3_all_agents = []
    for agent in range(num_agents):
        with open(f'./{conf3}/csvs/csvs_batch_256/battery_{battery_capacities[agent]}_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
            csvFile = csv.reader(file)
            next(csvFile)
            
            agent_episodes = []
            for line in csvFile:
                timesteps = [float(val) for val in line]
                episode_mean = np.mean(timesteps)
                agent_episodes.append(episode_mean)
            
            battery3_all_agents.append(agent_episodes)
    
    battery3_all_agents = np.array(battery3_all_agents)
    battery3_samples = np.mean(battery3_all_agents, axis=0)
    
    # ==== PLOT ====
    plt.suptitle("Average battery (sampled episodes)")
    plt.title(f"Episodes: {episodes}, Day: {day}, Interval: {interval}, num_agents: {num_agents}, Mode: cuda")
    
    # X-axis: episodio effettivo (0, 400, 800, ..., 4000)
    sample_episodes = [i * int(episodes / 10) for i in range(len(battery1_samples))]
    
    plt.xlabel("Episode")
    plt.ylabel("Average battery")
    
    plt.plot(sample_episodes, battery1_samples, 'o-', label=f"{conf1}", alpha=1.0, markersize=8, linewidth=2)
    plt.plot(sample_episodes, battery2_samples, 's-', label=f"{conf2}", alpha=1.0, markersize=8, linewidth=2)
    plt.plot(sample_episodes, battery3_samples, '^-', label=f"{conf3}", alpha=1.0, markersize=8, linewidth=2)

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./comparisons/battery/average_battery_sampled_{episodes}_{day}_{interval}_{num_agents}agents_{conf1}_{conf2}_{conf3}_cuda.pdf")
    plt.close()
    
    print(f"✅ Plotted {len(battery1_samples)} sampled episodes")
    print(f"   Episode numbers: {sample_episodes}")
    print(f"\n📊 Final values:")
    print(f"   {conf1}: {battery1_samples[-1]:.2f}")
    print(f"   {conf2}: {battery2_samples[-1]:.2f}")
    print(f"   {conf3}: {battery3_samples[-1]:.2f}")


def triple_plot_battery_evolution_daily(episodes, day, interval, num_agents, conf1, conf2, conf3):
    """
    BONUS: Plotta evoluzione battery durante il giorno per ogni episodio campionato
    """
    
    configs = [conf1, conf2, conf3]
    colors = {'aggregated_states': 'blue', 'reduced_states': 'green', 'local_only': 'red'}
    
    for config_name in configs:
        
        # Carica dati per tutti gli agenti di questa config
        all_agents_data = []
        for agent in range(num_agents):
            with open(f'./{config_name}/csvs/csvs_batch_256/battery_{battery_capacities[agent]}_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
                csvFile = csv.reader(file)
                next(csvFile)
                
                agent_episodes = []
                for line in csvFile:
                    timesteps = [float(val) for val in line]
                    agent_episodes.append(timesteps)
                
                all_agents_data.append(agent_episodes)
        
        # Media cross-agent
        all_agents_data = np.array(all_agents_data)  # Shape: [num_agents, 12, 1440]
        mean_data = np.mean(all_agents_data, axis=0)   # Shape: [12, 1440]
        
        # Plot
        plt.suptitle(f"Daily Battery Evolution - {config_name.upper()}")
        plt.title(f"Episodes: {episodes}, num_agents: {num_agents}")
        plt.xlabel("Timestep (minutes)")
        plt.ylabel("Average battery")
        
        window = 40
        for i, episode_data in enumerate(mean_data):
            episode_num = i * int(episodes / 10)
            
            # Smooth
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
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"./comparisons/battery/daily_evolution_{config_name}_{episodes}_{day}_{interval}_{num_agents}agents_cuda.pdf")
        plt.close()
        
        print(f"✅ Saved daily evolution plot for {config_name}")


double_compute_avg_battery_daily(4000, 355, 60, 5, "reduced_states", "sigmoid_z_score")

# triple_compute_avg_battery_daily(4000, 355, 60, 5, "aggregated_states", "reduced_states", "local_only")
# triple_plot_battery_evolution_daily(4000, 355, 60, 5, "aggregated_states", "reduced_states", "local_only")

