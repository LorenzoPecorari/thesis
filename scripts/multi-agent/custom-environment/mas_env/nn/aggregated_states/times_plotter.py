import numpy as np
import matplotlib.pyplot as plt
import csv

def main(episodes, day, interval, num_agents):
    non_parallelized_local = 0
    non_parallelized_vector = []
    non_parallelized_vector_single = []
    
    parallelized_local = 0
    parallelized_vector = []
    parallelized_vector_single = []
    
    process_parallelized_local = 0
    process_parallelized_vector = []
    process_parallelized_vector_single = []
    
    with open(f'./csvs/time_{episodes}_{day}_{interval}_{num_agents}agents.csv') as file:
        csvFile = csv.reader(file)
        for line in csvFile:
            non_parallelized_vector_single.append(float(line[0]))
            non_parallelized_local += float(line[0])
            non_parallelized_vector.append(non_parallelized_local)
            
    # with open(f'./csvs/parallelized_time_{episodes}_{day}_{interval}_{num_agents}agents.csv') as file:
    #     csvFile = csv.reader(file)
    #     for line in csvFile:
    #         parallelized_vector_single.append(float(line[0]))
    #         parallelized_local += float(line[0])
    #         parallelized_vector.append(parallelized_local)
     
    with open(f'./csvs/parallelized_time_{episodes}_{day}_{interval}_{num_agents}agents_processes.csv') as file:
        csvFile = csv.reader(file)
        for line in csvFile:
            process_parallelized_vector_single.append(float(line[0]))
            process_parallelized_local += float(line[0])
            process_parallelized_vector.append(parallelized_local)
            
    window = 10
    plt.suptitle("Time comparison")
    plt.title(f"Episodes: {episodes}, Day: {day}, Interval: {interval}, num_agents: {num_agents}")
    
    plt.xlabel("Episodes")
    plt.ylabel("Time [s]")
    
    # plt.plot(range(window - 1, len(levels[i])), np.convolve(levels[i], np.ones(window)/window, mode='valid'), label = f"smooth {self.battery_capacities[i]}Wh", alpha = 1.0)
    plt.plot(non_parallelized_vector, label = f"Non parallelized", alpha = 0.8)
    # plt.plot(parallelized_vector, label = f"Parallelized - threads", alpha = 0.8)
    plt.plot(process_parallelized_vector, label = f"Parallelized - processes", alpha = 0.8)
    
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"time_comparison_{episodes}_{day}_{interval}_{num_agents}agents.pdf")
    plt.close()
    
    print(non_parallelized_local / 3600 , parallelized_local / 3600)
    
    # for i in range(0, len(non_parallelized_vector)):
    #     plt.plot(range(window - 1, len(non_parallelized_vector[i])), np.convolve(non_parallelized_local[i], np.ones(window)/window, mode='valid'), label = f"smooth non parallelized", alpha = 1.0)
    plt.suptitle("Time comparison - episode")
    plt.title(f"Episodes: {episodes}, Day: {day}, Interval: {interval}, num_agents: {num_agents}")
    
    plt.xlabel("Episodes")
    plt.ylabel("Time [s]")
    
    # for i in range(0, len(non_parallelized_vector_single)):
    plt.plot(range(window - 1, len(non_parallelized_vector_single)), np.convolve(non_parallelized_vector_single, np.ones(window)/window, mode='valid'), label = f"Non parallelized", alpha = 1.0, color = "blue")
    plt.plot(non_parallelized_vector_single, label = f"Non parallelized", alpha = 0.3, color = "blue")
    # plt.plot(range(window - 1, len(parallelized_vector_single)), np.convolve(parallelized_vector_single, np.ones(window)/window, mode='valid'), label = f"Parallelized - threads", alpha = 1.0, color = "green")
    # plt.plot(parallelized_vector_single, alpha = 0.3, color = "green")
    plt.plot(range(window - 1, len(process_parallelized_vector_single)), np.convolve(process_parallelized_vector_single, np.ones(window)/window, mode='valid'), label = f"Parallelized - processes", alpha = 1.0, color = "red")
    plt.plot(parallelized_vector_single, alpha = 0.3, color = "red")
    
    # plt.plot(parallelized_vector_single, label = f"Non parallelized", alpha = 0.8)
    
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"time_comparison_episode_{episodes}_{day}_{interval}_{num_agents}agents.pdf")
    plt.close()


main(4000, 355, 60, 5)