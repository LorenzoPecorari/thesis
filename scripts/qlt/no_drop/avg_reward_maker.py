import csv
import os

def file_reader(folder_path):
    elems = os.listdir(folder_path)
    rewards = [[], []]
    print(rewards[0])
    
    for i in range(0, len(elems)):
        file = elems[i]
        print(file)
        with open(folder_path + "/" + file) as f:
            csvFile = csv.reader(f)
            
            for line in csvFile:
                rewards[i] += line
    
    avg_rew = []
    for i in range(0, len(rewards[0])):
        avg_rew.append((float(rewards[0][i]) + float(rewards[1][i]))/2)
        
    print(avg_rew)
    

if(__name__ == "__main__"):
    folder_path = "./csvs"
    file_reader(folder_path)