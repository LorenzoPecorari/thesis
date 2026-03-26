import numpy as np
import matplotlib.pyplot as plt
import csv
import os

def plot_batteries(timesteps, episodes):
    
    csv_path = "./saved_results/"
    
    files = os.listdir(csv_path)
    
    batteries = {"25": [], "100":[]}
    
    if("multi-agents_2agents_25Wh_355_4000_battery.csv" in files and
       "multi-agents_2agents_100Wh_355_4000_battery.csv"):
        
        with open(csv_path + "multi-agents_2agents_25Wh_355_4000_battery.csv") as file_25:
            csvFile = csv.reader(file_25)
            
            for line in csvFile:
                batteries["25"].append(float(line[0]))
                
        with open(csv_path + "multi-agents_2agents_100Wh_355_4000_battery.csv") as file_100:
            csvFile = csv.reader(file_100)
            
            for line in csvFile:
                batteries["100"].append(float(line[0]))
                
    window = 10
    plt.suptitle("Multi-agent : battery levels")
    plt.title(f"P_i = 2.6, P_f = 6.0, fps = 15, interval: 60s")
    
    plt.xlabel("Episodes")
    plt.ylabel("battery")
    
    for key in batteries.keys():
        # print(rewards[i])
        print(batteries[key])
        plt.plot(range(window - 1, len(batteries[key])), np.convolve(batteries[key], np.ones(window)/window, mode='valid'), label = f"smooth {key}Wh", alpha = 1.0)
        plt.plot(batteries[key], label = f"raw {key}Wh", alpha = 0.3)
    
    plt.grid()
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig(f"avg_battery_plot_{episodes-1}_355_60_1_2agents.pdf")
    plt.close()
                
        
        
                
plot_batteries(1440, 4001)