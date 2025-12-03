import numpy as np
from custom_environment import CustomEnvironment

# function for initial testing of environment
def test_policy(env, num_episodes):
    final_rewards = []
    
    for episode in range(0, num_episodes):
        observations, _ = env.reset()
        episode_rewards = {agent: 0 for agent in env.possible_agents}
        step = 0
        
        while env.agents:
            print(f"\n === Step: {step} ===")
            for elem in observations:
                print(f"observation {elem}: {observations[elem]}")
            
            actions = {
                agent: [0, 0, 0, 0]
                # agent: env.action_space(agent).sample()
                for agent in env.agents
            }
            
            print(f"actions: {actions}")

            observations, rewards, terminations, truncations, infos = env.step(actions)
            # print(f"Observations: {observations}\nRewards: {rewards}\nTerminations:{terminations}\nTruncations: {truncations}\nInfos: {infos}\n")
            
            
            for agent in rewards:
                episode_rewards[agent] += rewards[agent]
                
            step += 1
            input("Press enter to continue...")
        
        total_reward = sum(episode_rewards.values())
        final_rewards.append(total_reward)
        
        print(f"Episode: {episode+1}/{num_episodes} - step: {step} - rewards: {episode_rewards} - total: {total_reward}")

if __name__ == "__main__":
    
    # env params
    delta_time = 15 * 60
    proc_interval = 1 * 60
    
    proc_rate = 20
    arrival_rate = 15
    
    num_agents = 2
    num_episodes = 2
    
    power_idle = 2.4
    power_max = 6.0
    
    batteries = [25, 100]
    panel_surfaces = [1.0, 0.5]
    
    # irradiance datafiles
    irradiance_datapaths = [
        '../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2023.csv',
        '../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv'
    ]
    
    # env constructor
    env = CustomEnvironment(num_agents, irradiance_datapaths, delta_time, proc_interval, proc_rate, arrival_rate, batteries, panel_surfaces, power_idle, power_max)
    test_policy(env, num_episodes)
    