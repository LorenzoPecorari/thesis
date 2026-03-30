from qlt import Agent
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

datapath = '../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv'

# system params
battery_capacities = [25, 100]  # [Wh] - Two agents to compare
backlog_capacity = 54000
power_idle = 2.6                # [W]
power_max = 6.0                 # [W]
delta_time = 15 * 60            # [sec]
proc_interval = 60              # [sec]
pv_efficiency = 0.2
pv_area = 1.0                   # [m^2]
fps = 20
arrival_rate = 15
max_irradiation = 1000          # [W/m^2]

# training params

# SEEDS:
#   fixed_winter -> day 355
#   fixed_summer -> day 172
#   linear -> from 0 to 365
#   None -> random day at each episode
seed = "fixed_winter"

battery_bins = 15
time_bins = 15
alpha = 0.05
gamma = 0.99
eps_min = 0.05
eps_dec = 0.9975
eps_init = 1.0
episodes = 2000

# trains two agents (25Wh and 100Wh) and compares their performances
# plots: rewards, battery, backlog, framerate.
def train_and_compare():
    
    print("training and comparison: 25Wh vs 100Wh")
    
    results_dict = {}
    
    for battery_capacity in battery_capacities:
        print(f"Training agent: {battery_capacity}Wh")
        
        agent = Agent(
            datapath,
            battery_capacity,
            backlog_capacity,
            power_idle,
            power_max,
            delta_time,
            proc_interval,
            max_irradiation,
            pv_efficiency,
            pv_area,
            fps,
            arrival_rate,
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
        
        # [rewards, dropped, processed, battery, irradiance, backlog]
        results = agent.train()
        agent.save_table()
        
        results_dict[battery_capacity] = {
            'rewards': results[0],
            'battery': results[3],
            'backlog': results[5],
            'processed': results[2],
            'framerate': results[6]
        }
        
        input(results_dict[battery_capacity]['framerate'])
        save_framerate_csv(battery_capacity, results_dict[battery_capacity]['framerate'])
        print(f"\nagent {battery_capacity}Wh training complete")
    
    plot_all_comparisons(results_dict)


def plot_all_comparisons(results_dict):
    
    plot_comparison(
        results_dict,
        metric='rewards',
        title='Single agent - rewards comparison',
        ylabel='Rewards',
        filename='rewards_comparison_qlt.pdf'
    )
    
    plot_comparison(
        results_dict,
        metric='battery',
        title='Single agent - Average Battery',
        ylabel='Battery (%)',
        filename='battery_comparison_qlt.pdf'
    )
    
    plot_comparison(
        results_dict,
        metric='backlog',
        title='Single agent - Average Backlog',
        ylabel='Backlog level',
        filename='backlog_comparison_qlt.pdf'
    )
    
    plot_comparison(
        results_dict,
        metric='framerate',
        title='Single agent - Average fps',
        ylabel='FPS',
        filename='framerate_comparison_qlt.pdf'
    )


def plot_comparison(results_dict, metric, title, ylabel, filename):
    window = 10
    
    plt.figure()
    plt.suptitle(title)
    plt.title(f"p_I = {power_idle}W, p_F = {power_max}W")
    plt.xlabel("Episodes")
    plt.ylabel(ylabel)
    
    for battery_capacity in battery_capacities:
        data = results_dict[battery_capacity][metric]
        label = f"{battery_capacity}Wh"
        
        plt.plot(data, label=f"raw {label}", alpha=0.3)
        
        smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
        plt.plot(
            range(window - 1, len(data)),
            smoothed,
            label=f"smooth {label}",
            alpha=1.0
        )
    
    plt.grid()
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"Saved: {filename}")


def save_framerate_csv(battery_capacity, framerate_data):
    os.makedirs("csv", exist_ok=True)
    
    filename = f"csv/framerate_{battery_capacity}Wh_{episodes}ep.csv"
    with open(filename, "w") as file:
        for fps in framerate_data:
            file.write(f"{float(fps)}\n")
    
    print(f"saved: {filename}")


def plot_framerate(folder_path):
    elements = os.listdir(folder_path)
    print(elements)
        
    framerates = [[] for i in range(len(elements))]

    for f in range(len(elements)):
        print(f'{folder_path}/{elements[f]}')
        with open(f'{folder_path}/{elements[f]}') as file:
            csvFile = csv.reader(file)
            
            for line in csvFile:
                framerates[f].append(float(line[0]))
            
    print(framerates)       
    
    window = 50
    plt.suptitle("Average framerate")
    plt.title(f"Episodes: {len(framerates[0])-1}, Day: 355, Interval: 60")
    
    plt.xlabel("Episodes")
    plt.ylabel("Framerate")
    
    for id in range(len(elements)):
        plt.plot(framerates[id], label = f"raw {elements[id].split('_')[1]}", alpha = 0.2)
        plt.plot(range(window - 1, len(framerates[id])), np.convolve(framerates[id], np.ones(window)/window, mode='valid'), label = f"smooth {elements[id].split('_')[1]}", alpha = 1.0)

    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./comparisons/fps_comparison_sa.pdf")
    plt.close()
    
def plot_backlogs(folder_path):
    elements = os.listdir(folder_path)
    print(elements)
        
    framerates = [[] for i in range(len(elements))]

    for f in range(len(elements)):
        print(f'{folder_path}/{elements[f]}')
        if("4000" in f'{folder_path}/{elements[f]}'):
            with open(f'{folder_path}/{elements[f]}') as file:
                csvFile = csv.reader(file)
                
                for line in csvFile:
                    framerates[f].append(float(line[0]))
                            
    print(framerates)       
    
    window = 50
    plt.suptitle("Average backlog")
    plt.title(f"Episodes: {len(framerates[0])-1}, Day: 355, Interval: 60")
    
    plt.xlabel("Episodes")
    plt.ylabel("Backlog")
    
    for id in range(len(elements)):
        plt.plot(framerates[id], label = f"raw {elements[id].split('_')[1]}", alpha = 0.2)

    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./comparisons/backlog_comparison_sa.pdf")
    plt.close()
    

if __name__ == "__main__":
    train_and_compare()
    # plot_framerate("./csv_framerates")
    plot_backlogs("./csv_backlogs")
    
    # results_dict = {}
    # for battery_capacity in battery_capacities:
    #     agent = Agent(...)
    #     results = agent.train()
    #     results_dict[battery_capacity] = {...}
    # plot_all_subplots(results_dict)