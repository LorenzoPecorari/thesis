import numpy as np
import matplotlib.pyplot as plt
import csv

def main(episodes, day, interval, num_agents, mode):
    
    non_parallelized_local = 0
    non_parallelized_vector = []
    non_parallelized_vector_single = []
    
    parallelized_local = 0
    parallelized_vector = []
    parallelized_vector_single = []
    
    process_parallelized_local = 0
    process_parallelized_vector = []
    process_parallelized_vector_single = []
    
    with open(f'./csvs/csvs_batch_128/time_{episodes}_{day}_{interval}_{num_agents}agents_{mode}.csv') as file:
        csvFile = csv.reader(file)
        for line in csvFile:
            non_parallelized_vector_single.append(float(line[0]))
            non_parallelized_local += float(line[0])
            non_parallelized_vector.append(non_parallelized_local)
            
    with open(f'./csvs/csvs_batch_128/parallelized_time_{episodes}_{day}_{interval}_{num_agents}agents_threads_{mode}.csv') as file:
        csvFile = csv.reader(file)
        for line in csvFile:
            parallelized_vector_single.append(float(line[0]))
            parallelized_local += float(line[0])
            parallelized_vector.append(parallelized_local)
     
    with open(f'./csvs/csvs_batch_128/parallelized_time_{episodes}_{day}_{interval}_{num_agents}agents_processes_{mode}.csv') as file:
        csvFile = csv.reader(file)
        for line in csvFile:
            process_parallelized_vector_single.append(float(line[0]))
            process_parallelized_local += float(line[0])
            process_parallelized_vector.append(process_parallelized_local)
            
    window = 10
    plt.suptitle("Total time taken by execution comparison")
    plt.title(f"Episodes: {episodes}, Day: {day}, Interval: {interval}, num_agents: {num_agents}, Mode: {mode}")
    
    plt.xlabel("Episodes")
    plt.ylabel("Time [s]")
    
    # plt.plot(range(window - 1, len(levels[i])), np.convolve(levels[i], np.ones(window)/window, mode='valid'), label = f"smooth {self.battery_capacities[i]}Wh", alpha = 1.0)
    plt.plot(range(window - 1, len(non_parallelized_vector)), np.convolve(non_parallelized_vector, np.ones(window)/window, mode='valid'), label = f"Non parallelized", alpha = 1.0, color = "blue")
    plt.plot(range(window - 1, len(parallelized_vector)), np.convolve(parallelized_vector, np.ones(window)/window, mode='valid'), label = f"Parallelized - threads", alpha = 1.0, color = "red")
    plt.plot(range(window - 1, len(process_parallelized_vector)), np.convolve(process_parallelized_vector, np.ones(window)/window, mode='valid'), label = f"Parallelized - processes", alpha = 1.0, color = "green")

    # plt.plot(non_parallelized_vector_single, label = f"Non parallelized", alpha = 0.8)
    # plt.plot(parallelized_vector_single, label = f"Parallelized - threads", alpha = 0.8)
    # plt.plot(process_parallelized_vector_single, label = f"Parallelized - processes", alpha = 0.8)
    
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"time_comparison_{episodes}_{day}_{interval}_{num_agents}agents_{mode}.pdf")
    plt.close()
    
    # print(non_parallelized_local / 3600 , parallelized_local / 3600)
    
    # for i in range(0, len(non_parallelized_vector)):
    #     plt.plot(range(window - 1, len(non_parallelized_vector[i])), np.convolve(non_parallelized_local[i], np.ones(window)/window, mode='valid'), label = f"smooth non parallelized", alpha = 1.0)
    plt.suptitle("Time taken per episode comparison")
    plt.title(f"Episodes: {episodes}, Day: {day}, Interval: {interval}, num_agents: {num_agents}, Mode: {mode}")
    
    plt.xlabel("Episodes")
    plt.ylabel("Time [s]")
    
    # for i in range(0, len(non_parallelized_vector_single)):
    plt.plot(range(window - 1, len(non_parallelized_vector_single)), np.convolve(non_parallelized_vector_single, np.ones(window)/window, mode='valid'), label = f"Non parallelized", alpha = 1.0, color = "blue")
    # plt.plot(non_parallelized_vector_single, label = f"Non parallelized", alpha = 0.3, color = "blue")
    plt.plot(range(window - 1, len(parallelized_vector_single)), np.convolve(parallelized_vector_single, np.ones(window)/window, mode='valid'), label = f"Parallelized - threads", alpha = 1.0, color = "red")
    # plt.plot(parallelized_vector_single, alpha = 0.3, color = "red")
    plt.plot(range(window - 1, len(process_parallelized_vector_single)), np.convolve(process_parallelized_vector_single, np.ones(window)/window, mode='valid'), label = f"Parallelized - processes", alpha = 1.0, color = "green")
    # plt.plot(parallelized_vector_single, alpha = 0.3, color = "green")
    
    # plt.plot(parallelized_vector_single, label = f"Non parallelized", alpha = 0.8)
    
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"time_comparison_episode_{episodes}_{day}_{interval}_{num_agents}agents_{mode}.pdf")
    plt.close()


def cpu_vs_gpu(episodes, day, interval, num_agents, mode):
    
    cpu_local = 0
    cpu_vector = []
    cpu_vector_single = []
    
    gpu_local = 0
    gpu_vector = []
    gpu_vector_single = []
    
    if(mode == "parallelized_threads"):
        with open(f'./csvs/csvs_batch_128/parallelized_time_{episodes}_{day}_{interval}_{num_agents}agents_threads_cpu.csv') as file:
            csvFile = csv.reader(file)
            for line in csvFile:
                cpu_vector_single.append(float(line[0]))
                cpu_local += float(line[0])
                cpu_vector.append(cpu_local)
        
        
        with open(f'./csvs/csvs_batch_128/parallelized_time_{episodes}_{day}_{interval}_{num_agents}agents_threads_cuda.csv') as file:
            csvFile = csv.reader(file)
            for line in csvFile:
                gpu_vector_single.append(float(line[0]))
                gpu_local += float(line[0])
                gpu_vector.append(gpu_local)
    elif(mode == "parallelized_processes"):
        with open(f'./csvs/csvs_batch_128/parallelized_time_{episodes}_{day}_{interval}_{num_agents}agents_processes_cpu.csv') as file:
            csvFile = csv.reader(file)
            for line in csvFile:
                cpu_vector_single.append(float(line[0]))
                cpu_local += float(line[0])
                cpu_vector.append(cpu_local)
        
        
        with open(f'./csvs/csvs_batch_128/parallelized_time_{episodes}_{day}_{interval}_{num_agents}agents_processes_cuda.csv') as file:
            csvFile = csv.reader(file)
            for line in csvFile:
                gpu_vector_single.append(float(line[0]))
                gpu_local += float(line[0])
                gpu_vector.append(gpu_local)
    else:
        with open(f'./csvs/csvs_batch_128/time_{episodes}_{day}_{interval}_{num_agents}agents_cpu.csv') as file:
            csvFile = csv.reader(file)
            for line in csvFile:
                cpu_vector_single.append(float(line[0]))
                cpu_local += float(line[0])
                cpu_vector.append(cpu_local)
        
        
        with open(f'./csvs/csvs_batch_128/time_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
            csvFile = csv.reader(file)
            for line in csvFile:
                gpu_vector_single.append(float(line[0]))
                gpu_local += float(line[0])
                gpu_vector.append(gpu_local)
        
    window = 10
    plt.suptitle("Total time taken by execution comparison")
    plt.title(f"Episodes: {episodes}, Day: {day}, Interval: {interval}, num_agents: {num_agents}, Mode: {mode}")
    
    plt.xlabel("Episodes")
    plt.ylabel("Time [s]")
    
    if(mode == "parallelized_"):
        mode = "parallelized"
    elif(mode == ""):
        mode = "sequential"
        
    print("mode: ", mode == "")
    
    # plt.plot(range(window - 1, len(levels[i])), np.convolve(levels[i], np.ones(window)/window, mode='valid'), label = f"smooth {self.battery_capacities[i]}Wh", alpha = 1.0)
    plt.plot(range(window - 1, len(cpu_vector)), np.convolve(cpu_vector, np.ones(window)/window, mode='valid'), label = f"{mode} - cpu", alpha = 1.0, color = "blue")
    plt.plot(range(window - 1, len(gpu_vector)), np.convolve(gpu_vector, np.ones(window)/window, mode='valid'), label = f"{mode} - cuda", alpha = 1.0, color = "red")

    # plt.plot(non_parallelized_vector_single, label = f"Non parallelized", alpha = 0.8)
    # plt.plot(parallelized_vector_single, label = f"Parallelized - threads", alpha = 0.8)
    # plt.plot(process_parallelized_vector_single, label = f"Parallelized - processes", alpha = 0.8)
    
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"time_comparison_{episodes}_{day}_{interval}_{num_agents}agents_{mode}.pdf")
    plt.close()

    plt.suptitle("Time taken per episode comparison")
    plt.title(f"Episodes: {episodes}, Day: {day}, Interval: {interval}, num_agents: {num_agents}, Mode: {mode}")
    
    plt.xlabel("Episodes")
    plt.ylabel("Time [s]")
    
    plt.plot(range(window - 1, len(cpu_vector_single)), np.convolve(cpu_vector_single, np.ones(window)/window, mode='valid'), label = f"{mode} - cpu", alpha = 1.0, color = "blue")
    plt.plot(range(window - 1, len(gpu_vector_single)), np.convolve(gpu_vector_single, np.ones(window)/window, mode='valid'), label = f"{mode} - gpu", alpha = 1.0, color = "red")
    
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"time_comparison_episode_{episodes}_{day}_{interval}_{num_agents}agents_{mode}.pdf")
    plt.close()


main(4000, 355, 60, 5, "cuda")
# cpu_vs_gpu(4000, 355, 60, 5, "sequential")