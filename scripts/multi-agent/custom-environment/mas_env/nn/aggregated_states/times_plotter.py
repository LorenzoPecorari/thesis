import numpy as np
import matplotlib.pyplot as plt
import csv

def main(episodes, day, interval, num_agents):
    non_parallelized_local = 0
    non_parallelized_vector = []
    
    parallelized_local = 0
    parallelized_vector = []
    
    with open(f'./csvs/time_{episodes}_{day}_{interval}_{num_agents}agents.csv') as file:
        csvFile = csv.reader(file)
        for line in csvFile:
            non_parallelized_local += float(line[0])
            non_parallelized_vector.append(non_parallelized_local)
            
    with open(f'./csvs/parallelized_time_{episodes}_{day}_{interval}_{num_agents}agents.csv') as file:
        csvFile = csv.reader(file)
        for line in csvFile:
            parallelized_local += float(line[0])
            parallelized_vector.append(parallelized_local)
            
    window = 10
    plt.suptitle("Time comparison")
    plt.title(f"Episodes: {episodes}, Day: {day}, Interval: {interval}, num_agents: {num_agents}")
    
    plt.xlabel("Episodes")
    plt.ylabel("Battery")
    
    # plt.plot(range(window - 1, len(levels[i])), np.convolve(levels[i], np.ones(window)/window, mode='valid'), label = f"smooth {self.battery_capacities[i]}Wh", alpha = 1.0)
    plt.plot(non_parallelized_vector, label = f"Non parallelized", alpha = 0.8)
    plt.plot(parallelized_vector, label = f"parallelized", alpha = 0.8)
    
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"time_comparison_{episodes}_{day}_{interval}_{num_agents}agents.pdf")
    plt.close()
    
    print(non_parallelized_local / 3600 , parallelized_local / 3600)


main(4000, 172, 60, 4)