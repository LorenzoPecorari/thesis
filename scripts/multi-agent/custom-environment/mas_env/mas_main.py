import numpy as np
import matplotlib.pyplot as plt
from custom_environment import CustomEnvironment
from tabular_agent import TabularAgent

# function for initial testing of environment
def test_policy(env, num_episodes):
    final_rewards = []
    
    agents = []
    fs = [[] for i in range(0, env._num_agents)]
    hs = [[] for i in range(0, env._num_agents)]
    
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
                                   battery_bins=10,
                                   time_bins=10,
                                   alpha=0.1,
                                   gamma=0.99,
                                   eps_min=0.05,
                                   eps_dec=0.998,
                                #    eps_dec=0.98,
                                   eps_init=1.0,
                                   episodes=num_episodes
                                   ))
    
    rewards_for_plot = [[] for i in range(0, env._num_agents)]
    
    for episode in range(0, num_episodes):
        observations, _ = env.reset()
        episode_rewards = {agent: 0 for agent in env.possible_agents}
        step = 0
        
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
            
            # print(f"actions: {actions}")
            # print(f"actions: {actions}")

            new_observations, rewards, terminations, truncations, infos = env.step(actions)
            # print(f"Observations: {observations}\nRewards: {rewards}\nTerminations:{terminations}\nTruncations: {truncations}\nInfos: {infos}\n")
            # print(f"Observations: {observations}\nNew Observations: {new_observations}")
            
            for agent in rewards:
                episode_rewards[agent] += rewards[agent]

            # update agents at each iteration
            for id in range(0, env._num_agents):
                agents[id].update_table(observations[id], new_observations[id], actions[id], rewards[id])
            
            observations = new_observations    
                
            step += 1
        
        total_reward = sum(episode_rewards.values())
        final_rewards.append(total_reward)
        
        framerates_to_print = []
        
        for agent in range(0, env._num_agents):
            rewards_for_plot[agent].append(episode_rewards[agent])
            fs[agent].append(env.fs[agent] / step)
            hs[agent].append(env.hs[agent] / step)
            
            framerates_to_print.append([round(float(fs[agent][-1]), 3), round(float(hs[agent][-1]), 3)])
            
            env.fs[agent] = 0
            env.hs[agent] = 0
        
        for agent in agents:
            agent.update_epsilon()
                    
        print(f"Episode: {episode+1}/{num_episodes} - rewards: {episode_rewards} - framerates: {framerates_to_print}")

    plot_rewards(rewards_for_plot)
    plot_local_framerate(fs)
    plot_offloading_framerate(hs)

def plot_rewards(rewards):
    
    # print(rewards)
    
    window = 10
    plt.suptitle("Multi-agent : rewards")
    plt.title(f"P_i = {power_idle}, P_f = {power_max}, fps = {proc_rate}")
    
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    
    for i in range(0, env._num_agents):
        # print(rewards[i])
        plt.plot(range(window - 1, len(rewards[i])), np.convolve(rewards[i], np.ones(window)/window, mode='valid'), label = f"smooth {batteries[i]}Wh", alpha = 1.0)
        plt.plot(rewards[i], label = f"raw {batteries[i]}Wh", alpha = 0.3)
    
    plt.grid()    
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
    plt.tight_layout()
    plt.savefig(f"rewards_plot.pdf")
    plt.close()
    
def plot_local_framerate(fs):
    window = 10
    plt.suptitle("Multi-agent : local average framerate")
    plt.title(f"P_i = {power_idle}, P_f = {power_max}, fps = {proc_rate}")
    
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    
    for i in range(0, env._num_agents):
        # print(rewards[i])
        plt.plot(range(window - 1, len(fs[i])), np.convolve(fs[i], np.ones(window)/window, mode='valid'), label = f"smooth {batteries[i]}Wh", alpha = 1.0)
        plt.plot(fs[i], label = f"raw {batteries[i]}Wh", alpha = 0.3)
    
    plt.grid()
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
    plt.tight_layout()
    plt.savefig(f"local_framerate_plot.pdf")
    plt.close()
    
def plot_offloading_framerate(fs):
    window = 10
    plt.suptitle("Multi-agent : offloading average framerate")
    plt.title(f"P_i = {power_idle}, P_f = {power_max}, fps = {proc_rate}")
    
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    
    for i in range(0, env._num_agents):
        # print(rewards[i])
        plt.plot(range(window - 1, len(fs[i])), np.convolve(fs[i], np.ones(window)/window, mode='valid'), label = f"smooth {batteries[i]}Wh", alpha = 1.0)
        plt.plot(fs[i], label = f"raw {batteries[i]}Wh", alpha = 0.3)
    
    plt.grid()
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
    plt.tight_layout()
    plt.savefig(f"offloading_framerate_plot.pdf")
    plt.close()
    
if __name__ == "__main__":
    
    # env params
    delta_time = 15 * 60
    proc_interval = 1 * 60
    
    proc_rate = 20
    arrival_rate = 15
    
    num_agents = 2
    num_episodes = 2000
    
    power_idle = 2.4
    power_max = 6.0
    
    batteries = [25, 100, 50]
    panel_surfaces = [1.0, 0.5, 0.75]
    
    # irradiance datafiles
    # irradiance_datapaths = [
    #     '../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv',
    #     '../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv',
    #     '../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv'
    # ]
    
    irradiance_datapaths = [
        '../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv',
        '../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv',
    ]
    
    # env constructor
    env = CustomEnvironment(num_agents, irradiance_datapaths, delta_time, proc_interval, proc_rate, arrival_rate, batteries, panel_surfaces, power_idle, power_max)
    test_policy(env, num_episodes)
    