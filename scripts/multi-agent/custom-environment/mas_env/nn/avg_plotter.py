import numpy as np
import matplotlib.pyplot as plt
import csv
import os

agents = 3
timesteps = 1440
episodes = 4001

# folder_path = "./csv_n2"
folder_path = "./csv_avg_plots"

def rewards_plotter(folder_path, episodes, agents):
    # agents = 1
    elements = os.listdir(folder_path)
    
    rewards = {elem: [0 for i in range(episodes)] for elem in elements}
    # print(rewards)
    
    for elem in elements:
        inenr_dir = folder_path + "/" + elem + "/rewards/"
        elems = os.listdir(inenr_dir)
        # agents = len(elems)
        
        agent_id = 0
        for e in elems:
            
            if("rewards" in e) and ((str(agents) + "agents") in e):
                with open(inenr_dir + e) as f:
                    
                    csvFile = csv.reader(f)    
                    idx = 0
                    for line in csvFile:
                        rewards[elem][idx] += float(line[0])
                        idx += 1
                        
                    agent_id += 1
                        
        if(agent_id == 0):
            print(elem, rewards.keys())
            rewards.pop(elem)
                        
    print(f"len rewards: {len(rewards[elem])}, agents: {agents}")
    for elem in rewards.keys():
        for i in range(len(rewards[elem])):
            rewards[elem][i] /= agents
                
                
    # print(rewards)
    window = 100
    plt.suptitle("Multi-agent : average rewards")
    
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    
    for i in rewards.keys():
        # input(len(rewards[i]))
        # plt.plot(rewards[i], label = f"{i}", alpha = 0.3)
        plt.plot(range(window - 1, len(rewards[i])), np.convolve(rewards[i], np.ones(window)/window, mode='valid'), label = f"{i} smooth", linewidth = 2.0, alpha = 1.0)
        # plt.plot(rewards[i],  alpha = 0.3)
    
    plt.grid()
    plt.legend()
    # plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig(f"rewards_comparison_{agents}agents.pdf")
    plt.close()
    
def framerate_plotter(folder_path, episodes, agents):
    # agents = 1
    elements = os.listdir(folder_path)
    
    rewards = {elem: [0 for i in range(episodes)] for elem in elements}
    print(rewards.keys())
    
    for elem in elements:
        inenr_dir = folder_path + "/" + elem + "/framerate/"
        elems = os.listdir(inenr_dir)
        # agents = len(elems)
        
        agent_id = 0
        for e in elems:
            
            if("framerate" in e) and ((str(agents) + "agents") in e):
                with open(inenr_dir + e) as f:
                    
                    csvFile = csv.reader(f)
                
                    idx = 0
                    for line in csvFile:
                        rewards[elem][idx] += float(line[0])
                        idx += 1
                    
                    agent_id += 1
        
        if(agent_id == 0):
            print(elem, rewards.keys())
            rewards.pop(elem)
    
    for elem in rewards.keys():
        for i in range(len(rewards[elem])):
            rewards[elem][i] /= agents
                
                
    # print(rewards)
    window = 100
    plt.suptitle("Multi-agent : average framerate")
    
    plt.xlabel("Episodes")
    plt.ylabel("Framerate")
    
    for i in rewards.keys():
        # input(len(rewards[i]))
        # plt.plot(rewards[i], label = f"{i}", alpha = 0.3)
        plt.plot(range(window - 1, len(rewards[i])), np.convolve(rewards[i], np.ones(window)/window, mode='valid'), label = f"{i} smooth", linewidth = 2.0, alpha = 1.0)
        # plt.plot(rewards[i], alpha = 0.3)
    
    plt.grid()
    plt.legend()
    # plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig(f"framerate_comparison_{agents}agents.pdf")
    plt.close()
    
 
def backlogs_plotter(folder_path, timesteps, agents):
    elements = os.listdir(folder_path)
    # agents = 1
    samplings = 11
    
    print(elements)
    configurational_backlogs = {}
    
    for elem in elements:

        backlogs = {}
        inner_path = folder_path + "/" + elem
        
        elems = os.listdir(inner_path)
        
        if("backlog" in elems):
            path = inner_path + "/backlog/"
            
            # agents = len(os.listdir(path))
            agent_id = 0
            
            backlogs = [0 for j in range(timesteps)]
            
            for filename in os.listdir(path):
                if(str(agents) + "agents") in filename:
                    with open(path + filename) as f:
                        csvFile = csv.reader(f)
                        idx = 0
                        
                        for line in csvFile:
                        
                            if(idx > 0):
                                inner_idx = 0
                                for e in line:
                                    backlogs[inner_idx] += int(e)
                                    inner_idx += 1
                            
                            idx += 1
                            
                    agent_id += 1
                
        for i in range(len(backlogs)):
            backlogs[i] /= (agents * samplings)
            
        if(agent_id == 0):
            configurational_backlogs.pop(elem)
        else:
            configurational_backlogs[elem] = backlogs
    
        
    # print(configurational_backlogs)
    
        window = 10
        plt.suptitle("Multi-agent : average obacklogs")
        
        plt.xlabel("Episodes")
        plt.ylabel("Backlog")
        
        for i in configurational_backlogs.keys():
            # print(rewards[i])
            # plt.plot(range(window - 1, len(configurational_backlogs[i])), np.convolve(len(configurational_backlogs[i]), np.ones(window)/window, mode='valid'), label = f"i", linewidth = 2.0, alpha = 1.0)
            plt.plot(configurational_backlogs[i], label = f"{i}", linewidth = 2.0, alpha = 1.0)
        
        plt.grid()
        plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"backlog_comparison_{agents}agents.pdf")
        plt.close()

def battery_plotter(folder_path, timesteps, agents):
    elements = os.listdir(folder_path)
    # agents = 1
    samplings = 11
    
    print(elements)
    configurational_backlogs = {}
    
    for elem in elements:

        backlogs = {}
        inner_path = folder_path + "/" + elem
        
        elems = os.listdir(inner_path)
        
        if("backlog" in elems):
            path = inner_path + "/battery/"
            
            # agents = len(os.listdir(path))
            agent_id = 0
            
            backlogs = [0 for j in range(timesteps)]
            
            for filename in os.listdir(path):
                if(str(agents) + "agents") in filename:
                    with open(path + filename) as f:
                        csvFile = csv.reader(f)
                        idx = 0
                        
                        for line in csvFile:
                        
                            if(idx > 0):
                                inner_idx = 0
                                for e in line:
                                    backlogs[inner_idx] += float(e)
                                    inner_idx += 1
                            
                            idx += 1
                        
                    agent_id += 1
                
        for i in range(len(backlogs)):
            backlogs[i] /= (agents * samplings)
            
        configurational_backlogs[elem] = backlogs
        
    # print(configurational_backlogs)
    
        window = 10
        plt.suptitle("Multi-agent : average battery")
        
        plt.xlabel("Episodes")
        plt.ylabel("Battery")
        
        for i in configurational_backlogs.keys():
            # print(rewards[i])
            # plt.plot(range(window - 1, len(configurational_backlogs[i])), np.convolve(len(configurational_backlogs[i]), np.ones(window)/window, mode='valid'), label = f"i", linewidth = 2.0, alpha = 1.0)
            plt.plot(configurational_backlogs[i], label = f"{i}", linewidth = 2.0, alpha = 1.0)
        
        plt.grid()
        plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"battery_comparison_{agents}agents.pdf")
        plt.close()
        
def battery_sampled_plotter(folder_path, timesteps, agents):
    elements = os.listdir(folder_path)
    # agents = 1
    samplings = 11
    
    print(elements)
    configurational_backlogs = {}
    
    for elem in elements:
        # input(elem)
        backlogs = {}
        inner_path = folder_path + "/" + elem
        
        elems = os.listdir(inner_path)
        
        agent_id = 0
        if("battery" in elems):
            path = inner_path + "/battery/"
            
            # agents = len(os.listdir(path))
            
            backlogs = [0 for j in range(samplings)]
            
            for filename in os.listdir(path):
                # input(agents)
                # print(filename, (str(agents) + "agents") in filename)
                # input()
                
                if(str(agents) + "agents") in filename:
                    print(filename, (str(agents) + "agents") in filename)
                
                    with open(path + filename) as f:
                        csvFile = csv.reader(f)
                        idx = 0
                        
                        for line in csvFile:
                            # input(backlogs)
                        
                            if(idx > 0):
                                for e in line:
                                    backlogs[idx-1] += float(e)
                            
                            idx += 1
                        
                    agent_id += 1
                                
        for i in range(len(backlogs)):
            backlogs[i] /= (agents * timesteps)
                        
        if(agent_id != 0):
            configurational_backlogs[elem] = backlogs
        # else:
        #     configurational_backlogs.pop(elem)
            
        # input(configurational_backlogs)

    window = 10
    plt.suptitle("Multi-agent : average battery")
    
    plt.xlabel("Episodes/400")
    plt.ylabel("Battery")
    
    for i in configurational_backlogs.keys():
        # print(rewards[i])
        # plt.plot(range(window - 1, len(configurational_backlogs[i])), np.convolve(len(configurational_backlogs[i]), np.ones(window)/window, mode='valid'), label = f"i", linewidth = 2.0, alpha = 1.0)
        plt.plot(configurational_backlogs[i], 'o-', label = f"{i}", linewidth = 2.0, alpha = 1.0)
    
    
    plt.grid()
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig(f"battery_comparison_{agents}agents_sampled.pdf")
    plt.close()
    
def backlog_sampled_plotter(folder_path, timesteps, agents):
    elements = os.listdir(folder_path)
    # agents = 1
    samplings = 11
    
    print(elements)
    configurational_backlogs = {}
    
    for elem in elements:

        backlogs = {}
        inner_path = folder_path + "/" + elem
        
        elems = os.listdir(inner_path)
        
        if("backlog" in elems):
            path = inner_path + "/backlog/"
            
            # agents = len(os.listdir(path))
            agent_id = 0
            
            backlogs = [0 for j in range(samplings)]
            
            for filename in os.listdir(path):
                if(str(agents) + "agents") in filename:
                    with open(path + filename) as f:
                        csvFile = csv.reader(f)
                        idx = 0
                        
                        for line in csvFile:
                        
                            if(idx > 0):
                                for e in line:
                                    backlogs[idx-1] += float(e)
                            
                            idx += 1
                                                    
                    agent_id += 1
                                
        for i in range(len(backlogs)):
            backlogs[i] /= (agents * timesteps)
            
        if(agent_id != 0):
            configurational_backlogs[elem] = backlogs
        # else:
        #     configurational_backlogs.pop(elem)
        
        # input(configurational_backlogs)

    window = 10
    plt.suptitle("Multi-agent : average backlog")
    
    plt.xlabel("Episodes/400")
    plt.ylabel("Backlog")
    
    for i in configurational_backlogs.keys():
        # print(rewards[i])
        # plt.plot(range(window - 1, len(configurational_backlogs[i])), np.convolve(len(configurational_backlogs[i]), np.ones(window)/window, mode='valid'), label = f"i", linewidth = 2.0, alpha = 1.0)
        plt.plot(configurational_backlogs[i], 'o-', label = f"{i}", linewidth = 2.0, alpha = 1.0)
    
    
    plt.grid()
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig(f"backlog_comparison_{agents}agents_sampled.pdf")
    plt.close()

def file_cutter(folder_path, timesteps, frequency):
    folders = os.listdir(folder_path)
    # input(folders)
    
    for folder in folders:
        
        path = folder_path + folder + "/"
        files = os.listdir(path)
        # input(files)
            
        for file in files:
            # print(f"{folder}_{file.split('_')[3]}_{file.split('_')[4]}_{file.split('_')[5]}.csv")
            # print(path + file)
            # input(file.split('_'))
            # input(f"{file.split('_')[2]}_{file.split('_')[3]}_{file.split('_')[4]}.csv")
            
            with open(path + f"{folder}_{file.split('_')[2]}_{file.split('_')[3]}_{file.split('_')[4]}.csv", "w+") as fw:
                csvFileWrite = csv.writer(fw)
                csvFileWrite.writerow(f"t{i}" for i in range(timesteps))
                
                with open(path+file) as fr:
                    idx = 0
                    csvFileRead = csv.reader(fr)
                    
                    for line in csvFileRead:
                        if(idx % frequency == 0):
                            csvFileWrite.writerow(line[0] for i in range(timesteps))
                        
                        idx += 1
    
    
if __name__ == "__main__":
    # battery_plotter(folder_path, 1440)
    battery_sampled_plotter(folder_path, 1440, agents)
    
    # backlogs_plotter(folder_path, 1440, 3)
    backlog_sampled_plotter(folder_path, 1440, agents)
    
    rewards_plotter(folder_path, 4001, agents)
    framerate_plotter(folder_path, 4001, agents)

    # file_cutter(folder_path + "/Tabular/", timesteps, 400)
