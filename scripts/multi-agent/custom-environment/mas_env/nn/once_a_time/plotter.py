import numpy as np
import matplotlib.pyplot as plt
import os
import csv

folder_path = "./csvs_plots"
day = 355
interval = 60

def plot_rewards(folder_path):
    elements = os.listdir(folder_path + "/rewards")
    print(elements)
        
    rewards = [[] for i in range(len(elements))]

    for f in range(len(elements)):
        print(f'{folder_path}/rewards/{elements[f]}')
        with open(f'{folder_path}/rewards/{elements[f]}') as file:
            csvFile = csv.reader(file)
            
            for line in csvFile:
                rewards[f].append(float(line[0]))
            
            # rewards[f] /= cnt         
            
    print(rewards)       
    
    window = 50
    plt.suptitle("Average rewards")
    plt.title(f"Episodes: {len(rewards[0])-1}, Day: {day}, Interval: {interval},  Mode: cuda")
    
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    
    for id in range(len(elements)):
        plt.plot(range(window - 1, len(rewards[id])), np.convolve(rewards[id], np.ones(window)/window, mode='valid'), label = f"{elements[id].split('_')[2]}Wh", alpha = 1.0)

    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./comparisons/rewards_once_time_cuda.pdf")
    plt.close()
    

def plot_framerate(folder_path):
    elements = os.listdir(folder_path + "/framerates")
    print(elements)
        
    framerates = [[] for i in range(len(elements))]

    for f in range(len(elements)):
        print(f'{folder_path}/framerates/{elements[f]}')
        with open(f'{folder_path}/framerates/{elements[f]}') as file:
            csvFile = csv.reader(file)
            
            for line in csvFile:
                framerates[f].append(float(line[0]))
            
            # rewards[f] /= cnt         
            
    print(framerates)       
    
    window = 50
    plt.suptitle("Average framerate")
    plt.title(f"Episodes: {len(framerates[0])-1}, Day: {day}, Interval: {interval},  Mode: cuda")
    
    plt.xlabel("Episodes")
    plt.ylabel("Framerate")
    
    for id in range(len(elements)):
        plt.plot(range(window - 1, len(framerates[id])), np.convolve(framerates[id], np.ones(window)/window, mode='valid'), label = f"{elements[id].split('_')[2]}Wh", alpha = 1.0)

    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./comparisons/fps_once_time_cuda.pdf")
    plt.close()

def plot_batteries(folder_path):
    elements = os.listdir(folder_path + "/batteries")
    print(elements)
        
    batteries = [[] for i in range(len(elements))]

    for f in range(len(elements)):
        print(f'{folder_path}/batteries/{elements[f]}')
        with open(f'{folder_path}/batteries/{elements[f]}') as file:
            csvFile = csv.reader(file)
            next(csvFile)
            
            for line in csvFile:
                timesteps = [float(val) for val in line]
                episode_mean = np.mean(timesteps)
                batteries[f].append(episode_mean)
            
            print(batteries[f])
            
    print(batteries)
    
    window = 1
    plt.suptitle("Average battery")
    plt.title(f"Episodes: {2000}, Day: {day}, Interval: {interval},  Mode: cuda")

    plt.xlabel("Episodes")
    plt.ylabel("Battery")

    for id in range(len(elements)):
        plt.plot(range(window - 1, len(batteries[id])), np.convolve(batteries[id], np.ones(window)/window, mode='valid'), label = f"{elements[id].split('_')[1]}Wh", alpha = 1.0)

    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./comparisons/batteries_once_time_cuda.pdf")
    plt.close()
    
def plot_backlogs(folder_path):
    elements = os.listdir(folder_path + "/backlogs")
    print(elements)
        
    batteries = [[] for i in range(len(elements))]

    for f in range(len(elements)):
        print(f'{folder_path}/backlogs/{elements[f]}')
        with open(f'{folder_path}/backlogs/{elements[f]}') as file:
            csvFile = csv.reader(file)
            next(csvFile)
            
            for line in csvFile:
                timesteps = [float(val) for val in line]
                episode_mean = np.mean(timesteps)
                batteries[f].append(episode_mean)
            
            print(batteries[f])
            
    print(batteries)
    
    window = 1
    plt.suptitle("Average backlog")
    plt.title(f"Episodes: {2000}, Day: {day}, Interval: {interval},  Mode: cuda")

    plt.xlabel("Episodes")
    plt.ylabel("Backlog")

    for id in range(len(elements)):
        plt.plot(range(window - 1, len(batteries[id])), np.convolve(batteries[id], np.ones(window)/window, mode='valid'), label = f"{elements[id].split('_')[1]}Wh", alpha = 1.0)

    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./comparisons/backlogs_once_time_cuda.pdf")
    plt.close()
    
# print(len(os.listdir(folder_path)))

# plot_rewards(folder_path)
# plot_framerate(folder_path)

# plot_batteries(folder_path)
plot_backlogs(folder_path)