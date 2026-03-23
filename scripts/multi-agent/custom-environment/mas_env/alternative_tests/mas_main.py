import math
import numpy as np
import matplotlib.pyplot as plt
import os

from custom_environment import CustomEnvironment
from tabular_agent import TabularAgent

# function for initial testing of environment
def test_policy(env, num_episodes):
    final_rewards = []
    
    agents = []
    framerates  =[[] for i in range(0, env._num_agents)]
    fs = [[] for i in range(0, env._num_agents)]
    hs = [[] for i in range(0, env._num_agents)]
    
    # num_episodes = 8001
    
    for agent in env.possible_agents:
        agents.append(TabularAgent(agent, 
                                   env.battery_capacities[agent],
                                   1e10,
                                   env.p_idle,
                                   env.p_max,
                                   delta_time,
                                   proc_interval,
                                   env.panel_surfaces,
                                   proc_rate,
                                   arrival_rate,
                                   env._num_agents,
                                   battery_bins=15,
                                   time_bins=15,
                                #    battery_bins=4,    # for 3 agents
                                #    time_bins=4,       # for 3 agents
                                   alpha=0.05,
                                   gamma=0.99,
                                   eps_min=0.05,
                                #    eps_dec=0.839,     # ~ 33 episodes   
                                   eps_dec=0.9985,      # ~ 2500 episodes
                                #    eps_dec=0.9985,    # ~ 2000 episodes
                                #    eps_dec=0.997,     # ~ 1000 episodes
                                    # eps_dec=0.9975,     # ~ 1200 episodes
                                #    eps_dec=0.99,      # ~  300 episodes
                                #    eps_dec=0.999,     # ~ 3000 episodes
                                   eps_init=1.0,
                                   episodes=num_episodes,
                                   day=env.episode
                                   ))
    
    rewards_for_plot = [[] for i in range(0, env._num_agents)]
    rewards_daily = [[[] for j in range(0, math.ceil(num_episodes / int((num_episodes - 1) / 10)))] for i in range(0, env._num_agents)]

    battery_levels = [[] for i in range(0, env._num_agents)]
    battery_daily = [[[] for j in range(0, math.ceil(num_episodes / int((num_episodes - 1) / 10)))] for i in range(0, env._num_agents)]
    
    backlogs_average = [[] for i in range(0, env._num_agents)]
    backlogs_daily = [[[] for j in range(0, math.ceil(num_episodes / int((num_episodes - 1) / 10)))] for i in range(0, env._num_agents)]

    actual_offloading = [[] for i in range(0, env._num_agents)]

    # reward_batteries = [[] for i in range(0, env._num_agents)]
    # reward_frames = [[] for i in range(0, env._num_agents)]
    # reward_cooperation = [[] for i in range(0, env._num_agents)]
    # reward_backlog = [[] for i in range(0, env._num_agents)]
    
    sent = [[] for i in range(0, env._num_agents)]
    received = [[] for i in range(0, env._num_agents)]    
    
    
    for episode in range(0, num_episodes):
        framerates_temp = [0 for i in range(0, env._num_agents)]

        off = [[] for i in range(0, env._num_agents)]
        send_hi = [[] for i in range(0, env._num_agents)]
        recv_hi = [[] for i in range(0, env._num_agents)]
    
        observations, _ = env.reset()
        # input(observations)
        episode_rewards = {agent: 0 for agent in env.possible_agents}
        step = 0
        
        batteries = [0.0 for i in range(0, env._num_agents)]
        backlogs = [0.0 for i in range(0, env._num_agents)]
        
        while env.agents:
            # print(f"\n === Step: {step} ===")
            # for elem in observations:
                # print(f"observation {elem}: {observations[elem]}")
            
            # print(f"observations: {observations}")
            
            actions = {
                # agent: [0, 0, 0, 0]
                agent_id: agents[agent_id].choice_action(observations[agent_id])
                for agent_id in env.agents
            }
            
            # print(f"chosen actions: {actions}")
            
            new_observations, rewards, terminations, truncations, infos = env.step(actions)
            # print(f"Observations: {observations}\nRewards: {rewards}\nTerminations:{terminations}\nTruncations: {truncations}\nInfos: {infos}\n")
            # print(f"Observations: {observations}\nNew Observations: {new_observations}")
            
            for agent in rewards:

                episode_rewards[agent] += rewards[agent]
                # print(f"agent: {agent} - battery: {batteries[agent]} - obs: {new_observations[agent][0]}")
                batteries[agent] += new_observations[agent][0]
                backlogs[agent] += env.backlogs[agent]
                
                if(episode % int((num_episodes - 1) / 10) == 0):
                    battery_daily[agent][int(episode / int((num_episodes - 1) / 10))].append(env.battery_energies[agent] / env.battery_capacities[agent])
                    rewards_daily[agent][int(episode / int((num_episodes - 1) / 10))].append(rewards[agent])
                    backlogs_daily[agent][int(episode / int((num_episodes - 1) / 10))].append(env.backlogs[agent])
                        
                    if(actions[agent][1] == 0):
                        send_hi[agent].append(0)
                        recv_hi[agent].append(0)
                        off[agent].append(0)
                    else:
                        if(actions[agent][1] == 1):
                            send_hi[agent].append(actions[agent][3])
                            recv_hi[agent].append(0)  
                            
                        elif(actions[agent][1] == 2):
                            send_hi[agent].append(0)
                            recv_hi[agent].append(actions[agent][3])

                        off[agent].append(env.hs[agent])
                
            # update agents at each iteration
            for id in range(0, env._num_agents):
                agents[id].update_table(observations[id], new_observations[id], actions[id], rewards[id])
                # print(f"agent: {id} - {actions[id][0]} , {actions[id][3]}")
                # input()
                if((actions[id][0] + actions[id][3]) > proc_rate):
                    framerates_temp[id] += (actions[id][0])
                else:
                    framerates_temp[id] += (actions[id][0] + actions[id][3])
                    

            observations = new_observations    
                
            step += 1

        total_reward = sum(episode_rewards.values())
        final_rewards.append(total_reward)
        
        framerates_to_print = []

        
        for agent in range(0, env._num_agents):
            rewards_for_plot[agent].append(episode_rewards[agent])
            fs[agent].append(env.fs[agent] / step)
            if(env.hs_counter[agent] > 0):
                hs[agent].append(env.hs[agent] / env.hs_counter[agent])
            else:
                hs[agent].append(0.0)

            battery_levels[agent].append((batteries[agent] / step))
            # print(f"agent: {agent} - battery_level: {batteries[agent]} - battery: {battery_levels[agent]}")
            batteries[agent] = []
            
            backlogs_average[agent].append(backlogs[agent] / step)
            backlogs[agent] = []

            if(len(hs[agent]) > 0):
                framerates_to_print.append([round(float(fs[agent][-1]), 3), round(float(hs[agent][-1]), 3)])                        
            else:
                framerates_to_print.append([round(float(fs[agent][-1]), 3), 0.0])                                        
            # framerates_to_print.append([round(float(fs[agent][-1]), 3), round(float(send_hi[agent][0]/send_hi[agent][1]), 3), round(float(recv_hi[agent][0]/recv_hi[agent][1]), 3), round(float(hs[agent][-1]), 3)])
            framerates[agent].append(framerates_temp[agent] / step)
            
            env.fs[agent] = 0
            env.hs[agent] = 0
            env.hs_counter[agent] = 0
            
            # reward_batteries[agent].append(env.r_battery[agent])
            # reward_frames[agent].append(env.r_frames[agent])
            # reward_cooperation[agent].append(env.r_cooperation[agent])
            # reward_backlog[agent].append(env.r_backlog[agent])
            
            # env.r_battery[agent] = 0
            # env.r_frames[agent] = 0
            # env.r_cooperation[agent] = 0
            # env.r_backlog[agent] = 0
            
            if(len(send_hi[agent]) > 0 and len(recv_hi[agent])):
                sent[agent].append(send_hi[agent])
                received[agent].append(recv_hi[agent])
                
            if(len(off[agent])):
                actual_offloading[agent].append(off[agent])
                # input(actual_offloading)
            
            # input(actual_offloading)
            
            send_hi[agent] = []
            recv_hi[agent] = []
            
            agents[agent].update_epsilon()
        
        print(f"Episode: {episode+1}/{num_episodes} - rewards: {episode_rewards} - framerates: {framerates_to_print}")
        
    # for agent in range(0, env._num_agents):
    #     agents[agent].save_table() 
        
    plot_rewards(rewards_for_plot)
    plot_local_framerate(fs)
    plot_offloading_framerate(hs)
    plot_framerate(framerates)
    plot_battery_levels(battery_levels)
    plot_backlogs(backlogs_average)
    
    plot_reward_daily(rewards_daily)
    plot_battery_daily(battery_daily)
    plot_backlog_daily(backlogs_daily)
    
    plot_actual_offloading_daily(actual_offloading)
    
    # plot_rewards_batteries(reward_batteries)
    # plot_rewards_backlog(reward_backlog)
    # plot_rewards_cooperation(reward_cooperation)
    # plot_rewards_frames(reward_frames)
    
    plot_recvd_daily(received)
    plot_sent_daily(sent)
    
    save_results(battery_levels, rewards_for_plot, backlogs_average, framerates, num_episodes, num_agents)

def save_results(batteries, rewards, backlogs, framerates, num_episodes, num_agents):
    for id in range(0, len(env.possible_agents)):
        filepath = f"./saved_results/multi-agents_{num_agents}agents_{int(env.battery_capacities[id]/3600)}Wh_{env.episode}_{num_episodes-1}_battery.csv"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, "w") as file:
            for elem in batteries[id]:
                file.write(str(float(elem)) + "\n")
        
        filepath = f"./saved_results/multi-agents_{num_agents}agents_{int(env.battery_capacities[id]/3600)}Wh_{env.episode}_{num_episodes-1}_rewards.csv"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, "w") as file:
            for elem in rewards[id]:
                file.write(str(float(elem)) + "\n")
                
        filepath = f"./saved_results/multi-agents_{num_agents}agents_{int(env.battery_capacities[id]/3600)}Wh_{env.episode}_{num_episodes-1}_framerate.csv"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, "w") as file:
            for elem in framerates[id]:
                file.write(str(float(elem)) + "\n")
        
        filepath = f"./saved_results/multi-agents_{num_agents}agents_{int(env.battery_capacities[id]/3600)}Wh_{env.episode}_{num_episodes-1}_backlogs.csv"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, "w") as file:
            for elem in backlogs[id]:
                file.write(str(float(elem)) + "\n")
        
                

def plot_rewards(rewards):
    
    # print(rewards)
    
    window = 10
    plt.suptitle("Multi-agent : rewards")
    plt.title(f"P_i = {power_idle}, P_f = {power_max}, fps = {proc_rate}, interval: {proc_interval}s")
    
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    
    for i in range(0, env._num_agents):
        # print(rewards[i])
        plt.plot(range(window - 1, len(rewards[i])), np.convolve(rewards[i], np.ones(window)/window, mode='valid'), label = f"smooth {batteries[i]}Wh", alpha = 1.0)
        plt.plot(rewards[i], label = f"raw {batteries[i]}Wh", alpha = 0.3)
    
    plt.grid()
    # plt.ylim(-10, 500)
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig(f"rewards_plot_{num_episodes - 1}_{env.episode}_{proc_interval}_{w}_{env._num_agents}agents.pdf")
    plt.close()
    
def plot_local_framerate(fs):
    window = 10
    plt.suptitle("Multi-agent : local average framerate")
    plt.title(f"P_i = {power_idle}, P_f = {power_max}, fps = {proc_rate}, interval: {proc_interval}s")
    
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    
    for i in range(0, env._num_agents):
        # print(rewards[i])
        plt.plot(range(window - 1, len(fs[i])), np.convolve(fs[i], np.ones(window)/window, mode='valid'), label = f"smooth {batteries[i]}Wh", alpha = 1.0)
        plt.plot(fs[i], label = f"raw {batteries[i]}Wh", alpha = 0.3)
    
    plt.grid()
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig(f"local_framerate_plot_{num_episodes-1}_{env.episode}_{proc_interval}_{w}_{env._num_agents}agents.pdf")
    plt.close()
    
def plot_framerate(fs):
    window = 10
    plt.suptitle("Multi-agent : average framerate")
    plt.title(f"P_i = {power_idle}, P_f = {power_max}, fps = {proc_rate}, interval: {proc_interval}s")
    
    plt.xlabel("Episodes")
    plt.ylabel("Framerate")
    
    for i in range(0, env._num_agents):
        # print(rewards[i])
        plt.plot(range(window - 1, len(fs[i])), np.convolve(fs[i], np.ones(window)/window, mode='valid'), label = f"smooth {batteries[i]}Wh", alpha = 1.0)
        plt.plot(fs[i], label = f"raw {batteries[i]}Wh", alpha = 0.3)
    
    plt.grid()
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig(f"framerate_plot_{num_episodes-1}_{env.episode}_{proc_interval}_{w}_{env._num_agents}agents.pdf")
    plt.close()
    
def plot_battery_levels(levels):
    window = 10
    plt.suptitle("Multi-agent : battery levels")
    plt.title(f"P_i = {power_idle}, P_f = {power_max}, fps = {proc_rate}, interval: {proc_interval}s")
    
    plt.xlabel("Episodes")
    plt.ylabel("battery")
    
    for i in range(0, env._num_agents):
        # print(rewards[i])
        plt.plot(range(window - 1, len(levels[i])), np.convolve(levels[i], np.ones(window)/window, mode='valid'), label = f"smooth {batteries[i]}Wh", alpha = 1.0)
        plt.plot(levels[i], label = f"raw {batteries[i]}Wh", alpha = 0.3)
    
    plt.grid()
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig(f"avg_battery_plot_{num_episodes-1}_{env.episode}_{proc_interval}_{w}_{env._num_agents}agents.pdf")
    plt.close()
    
def plot_offloading_framerate(fs):
    window = 10
    plt.suptitle("Multi-agent : offloading average framerate")
    plt.title(f"P_i = {power_idle}, P_f = {power_max}, fps = {proc_rate}, interval: {proc_interval}s")
    
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    
    for i in range(0, env._num_agents):
        # print(rewards[i])
        plt.plot(range(window - 1, len(fs[i])), np.convolve(fs[i], np.ones(window)/window, mode='valid'), label = f"smooth {batteries[i]}Wh", alpha = 1.0)
        plt.plot(fs[i], label = f"raw {batteries[i]}Wh", alpha = 0.3)
    
    plt.grid()
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig(f"offloading_framerate_plot_{num_episodes-1}_{env.episode}_{proc_interval}_{w}_{env._num_agents}agents.pdf")
    plt.close()
    
def plot_backlogs(backlogs):
    window = 10
    plt.suptitle("Multi-agent : average backlog")
    plt.title(f"P_i = {power_idle}, P_f = {power_max}, fps = {proc_rate}, interval: {proc_interval}s")
    
    plt.xlabel("Timesteps")
    plt.ylabel("Rewards")
    
    for i in range(0, env._num_agents):
        # print(rewards[i])
        plt.plot(range(window - 1, len(backlogs[i])), np.convolve(backlogs[i], np.ones(window)/window, mode='valid'), label = f"smooth {batteries[i]}Wh", alpha = 1.0)
        plt.plot(backlogs[i], label = f"raw {batteries[i]}Wh", alpha = 0.3)
    
    plt.grid()
    # plt.legend()
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig(f"backlog_plot_{num_episodes-1}_{env.episode}_{proc_interval}_{w}_{env._num_agents}agents.pdf")
    plt.close()
    
def plot_reward_daily(data):
    
    for elem in range(0, env._num_agents):
        
        window = 40
        plt.suptitle("Multi-agent : daily reward")
        plt.title(f"B: {env.battery_capacities[elem] / 3600} - P_i = {power_idle}, P_f = {power_max}, fps = {proc_rate}, interval: {proc_interval}s")
        
        plt.xlabel("Timesteps")
        plt.ylabel("Rewards")
        for i in range(0, len(data[elem])):
            # print(rewards[i])
            plt.plot(range(window - 1, len(data[elem][i])), np.convolve(data[elem][i], np.ones(window)/window, mode='valid'), label = f"{i * (int((num_episodes-1) / 10))}-th episode", alpha = 1.0)
        
        plt.grid()
        plt.legend()
        # plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"rewards_{int(env.battery_capacities[elem] / 3600)}Wh_{num_episodes-1}_{env.episode}_{proc_interval}_{w}_{env._num_agents}agents.pdf")
        plt.close()

def plot_battery_daily(data):
    
    print(len(data))
    
    for elem in range(0, env._num_agents):
                
        window = 40
        plt.suptitle("Multi-agent : daily battery")
        plt.title(f"B: {env.battery_capacities[elem] / 3600} - P_i = {power_idle}, P_f = {power_max}, fps = {proc_rate}, interval: {proc_interval}s")
        
        plt.xlabel("Timesteps")
        plt.ylabel("Battery")
        for i in range(0, len(data[elem])):
            # print(rewards[i])
            plt.plot(range(window - 1, len(data[elem][i])), np.convolve(data[elem][i], np.ones(window)/window, mode='valid'), label = f"{i * (int((num_episodes-1) / 10))}-th episode", alpha = 1.0)
        
        plt.grid()
        plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"battery_{int(env.battery_capacities[elem] / 3600)}Wh_{num_episodes-1}_{env.episode}_{proc_interval}_{w}_{env._num_agents}agents.pdf")
        plt.close()

def plot_backlog_daily(data):
    
    for elem in range(0, env._num_agents):
        
        window = 40
        plt.suptitle("Multi-agent : daily backlog")
        plt.title(f"B: {env.battery_capacities[elem] / 3600} - P_i = {power_idle}, P_f = {power_max}, fps = {proc_rate}, interval: {proc_interval}s")
        
        plt.xlabel("Timesteps")
        plt.ylabel("Backlog")
        for i in range(0, len(data[elem])):
            # print(rewards[i])
            plt.plot(range(window - 1, len(data[elem][i])), np.convolve(data[elem][i], np.ones(window)/window, mode='valid'), label = f"{i* (int((num_episodes-1) / 10))}-th episode", alpha = 1.0)
        
        plt.grid()
        # plt.legend()
        plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"backlog_{int(env.battery_capacities[elem] / 3600)}Wh_{num_episodes-1}_{env.episode}_{proc_interval}_{w}_{env._num_agents}agents.pdf")
        plt.close()
        
def plot_actual_offloading_daily(data):
    for elem in range(0, env._num_agents):
                
        window = 40
        plt.suptitle("Multi-agent : daily offloading")
        plt.title(f"B: {env.battery_capacities[elem] / 3600} - P_i = {power_idle}, P_f = {power_max}, fps = {proc_rate}, interval: {proc_interval}s")
        
        plt.xlabel("Timestep")
        plt.ylabel("FPS")
        for i in range(0, len(data[elem])):
            # print(rewards[i])
            plt.plot(range(window - 1, len(data[elem][i])), np.convolve(data[elem][i], np.ones(window)/window, mode='valid'), label = f"{i * (int((num_episodes-1) / 10))}-th episode", alpha = 1.0)
        
        plt.grid()
        plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"offloading_actual_{int(env.battery_capacities[elem] / 3600)}Wh_{num_episodes-1}_{env.episode}_{proc_interval}_{w}_{env._num_agents}agents.pdf")
        plt.close()

def plot_rewards_batteries(rewards):
    
    # print(rewards)
    
    window = 10
    plt.suptitle("Multi-agent : rewards")
    plt.title(f"P_i = {power_idle}, P_f = {power_max}, fps = {proc_rate}, interval: {proc_interval}s")
    
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    
    for i in range(0, env._num_agents):
        # print(rewards[i])
        plt.plot(range(window - 1, len(rewards[i])), np.convolve(rewards[i], np.ones(window)/window, mode='valid'), label = f"smooth {batteries[i]}Wh", alpha = 1.0)
        plt.plot(rewards[i], label = f"raw {batteries[i]}Wh", alpha = 0.3)
    
    plt.grid()    
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig(f"rewards_batteries_plot_{num_episodes-1}_{env.episode}_{proc_interval}_{w}_{env._num_agents}agents.pdf")
    plt.close()

def plot_rewards_frames(rewards):
    
    # print(rewards)
    
    window = 10
    plt.suptitle("Multi-agent : rewards frames")
    plt.title(f"P_i = {power_idle}, P_f = {power_max}, fps = {proc_rate}, interval: {proc_interval}s")
    
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    
    for i in range(0, env._num_agents):
        # print(rewards[i])
        plt.plot(range(window - 1, len(rewards[i])), np.convolve(rewards[i], np.ones(window)/window, mode='valid'), label = f"smooth {batteries[i]}Wh", alpha = 1.0)
        plt.plot(rewards[i], label = f"raw {batteries[i]}Wh", alpha = 0.3)
    
    plt.grid()    
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig(f"rewards_frames_plot_{num_episodes-1}_{env.episode}_{proc_interval}_{w}_{env._num_agents}agents.pdf")
    plt.close()

def plot_rewards_cooperation(rewards):
    
    # print(rewards)
    
    window = 10
    plt.suptitle("Multi-agent : rewards cooperation")
    plt.title(f"P_i = {power_idle}, P_f = {power_max}, fps = {proc_rate}, interval: {proc_interval}s")
    
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    
    for i in range(0, env._num_agents):
        # print(rewards[i])
        plt.plot(range(window - 1, len(rewards[i])), np.convolve(rewards[i], np.ones(window)/window, mode='valid'), label = f"smooth {batteries[i]}Wh", alpha = 1.0)
        plt.plot(rewards[i], label = f"raw {batteries[i]}Wh", alpha = 0.3)
    
    plt.grid()    
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig(f"rewards_cooperation_plot_{num_episodes-1}_{env.episode}_{proc_interval}_{w}_{env._num_agents}agents.pdf")
    plt.close()

def plot_rewards_backlog(rewards):
    
    # print(rewards)
    
    window = 10
    plt.suptitle("Multi-agent : rewards backlog")
    plt.title(f"P_i = {power_idle}, P_f = {power_max}, fps = {proc_rate}, interval: {proc_interval}s")
    
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    
    for i in range(0, env._num_agents):
        # print(rewards[i])
        plt.plot(range(window - 1, len(rewards[i])), np.convolve(rewards[i], np.ones(window)/window, mode='valid'), label = f"smooth {batteries[i]}Wh", alpha = 1.0)
        plt.plot(rewards[i], label = f"raw {batteries[i]}Wh", alpha = 0.3)
    
    plt.grid()    
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig(f"rewards_backlog_plot_{num_episodes-1}_{env.episode}_{proc_interval}_{w}_{env._num_agents}agents.pdf")
    plt.close()


def plot_sent_daily(data):
    
    for elem in range(0, env._num_agents):
        
        window = 40
        plt.suptitle("Multi-agent : daily sent frames")
        plt.title(f"B: {env.battery_capacities[elem] / 3600} - P_i = {power_idle}, P_f = {power_max}, fps = {proc_rate}, interval: {proc_interval}s")
        
        plt.xlabel("Timestep")
        plt.ylabel("Framerate")
        for i in range(0, len(data[elem])):
            # print(rewards[i])
            plt.plot(range(window - 1, len(data[elem][i])), np.convolve(data[elem][i], np.ones(window)/window, mode='valid'), label = f"{i * (int((num_episodes-1) / 10))}-th episode", alpha = 1.0)
        
        plt.grid()
        plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"rewards_sent_{int(env.battery_capacities[elem] / 3600)}Wh_{num_episodes-1}_{env.episode}_{proc_interval}_{w}_{env._num_agents}agents.pdf")
        plt.close()

def plot_recvd_daily(data):
    
    for elem in range(0, env._num_agents):
        
        window = 40
        plt.suptitle("Multi-agent : daily received frames")
        plt.title(f"B: {env.battery_capacities[elem] / 3600} - P_i = {power_idle}, P_f = {power_max}, fps = {proc_rate}, interval: {proc_interval}s")
        
        plt.xlabel("Timestep")
        plt.ylabel("Framerate")
        for i in range(0, len(data[elem])):
            # print(rewards[i])
            plt.plot(range(window - 1, len(data[elem][i])), np.convolve(data[elem][i], np.ones(window)/window, mode='valid'), label = f"{i * (int((num_episodes-1) / 10))}-th episode", alpha = 1.0)
        
        plt.grid()
        plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"rewards_recvd_{int(env.battery_capacities[elem] / 3600)}Wh_{num_episodes-1}_{env.episode}_{proc_interval}_{w}_{env._num_agents}agents.pdf")
        plt.close()


if __name__ == "__main__":
    
    # env params
    delta_time = 15 * 60
    proc_interval = 1 * 60
    
    proc_rate = 20
    arrival_rate = 15
    
    num_agents = 2
    num_episodes = 4001

    power_idle = 2.6
    power_max = 6.0

    # batteries = [25, 100]
    # panel_surfaces = [1.0, 0.5]

    
    batteries = [25, 100, 50]
    panel_surfaces = [1.0, 0.5, 0.75]
    
    w = 1
    
    # batteries = [25, 100, 50]
    # panel_surfaces = [1.0, 0.5, 0.75]
    
    # irradiance datafiles
    # irradiance_datapaths = [
    #     '../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv',
    #     '../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv',
    #     '../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv'
    # ]
    
    irradiance_datapaths = [
        '../../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv',
        '../../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv',
        '../../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv'    
    ]
    
    # env constructor
    env = CustomEnvironment(num_agents, irradiance_datapaths, delta_time, proc_interval, proc_rate, arrival_rate, batteries, panel_surfaces, power_idle, power_max, w)
    test_policy(env, num_episodes)
    