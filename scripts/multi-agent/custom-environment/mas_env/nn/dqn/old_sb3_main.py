from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure

from custom_environment import CustomEnvironment
from env_wrapper import EnvWrapper

import numpy as np
import matplotlib.pyplot as plt

irradiance_datapaths = [
    '../../../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv',
    '../../../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv',
    '../../../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv'
    ]
delta_time = 15 * 60
proc_interval = 1 * 60
proc_rate = 20
arrival_rate = 15

eps_init = 1.0
eps_fin = 0.05
eps_dec = 0.9975

num_agents = 2
battery_capacities = [25, 100]
panel_surfaces = [1.0, 0.5]

# num_agents = 3
# battery_capacities = [25, 100, 50]
# panel_surfaces = [1.0, 0.5, 0.75]

power_idle = 2.6
power_max = 6.0

w = 1.0

num_episodes = 2001
train_freq = 8

def plot_rewards(rewards):
    
    # print(rewards)
    
    window = 10
    plt.suptitle("Multi-agent : rewards")
    plt.title(f"P_i = {power_idle}, P_f = {power_max}, fps = {proc_rate}, interval: {proc_interval}s")
    
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    
    for i in range(0, env._num_agents):
        # print(rewards[i])
        plt.plot(range(window - 1, len(rewards[i])), np.convolve(rewards[i], np.ones(window)/window, mode='valid'), label = f"smooth {battery_capacities[i]}Wh", alpha = 1.0)
        plt.plot(rewards[i], label = f"raw {battery_capacities[i]}Wh", alpha = 0.3)
    
    plt.grid()
    # plt.ylim(-10, 500)
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig(f"rewards_plot_{num_episodes - 1}_{env.episode}_{proc_interval}_{w}_{num_agents}agents.pdf")
    plt.close()

def plot_battery_levels(levels):
    window = 10
    plt.suptitle("Multi-agent : battery levels")
    plt.title(f"P_i = {power_idle}, P_f = {power_max}, fps = {proc_rate}, interval: {proc_interval}s")
    
    plt.xlabel("Episodes")
    plt.ylabel("Battery")
    
    for i in range(0, env._num_agents):    
        # print(rewards[i])
        plt.plot(range(window - 1, len(levels[i])), np.convolve(levels[i], np.ones(window)/window, mode='valid'), label = f"smooth {battery_capacities[i]}Wh", alpha = 1.0)
        plt.plot(levels[i], label = f"raw {battery_capacities[i]}Wh", alpha = 0.3)
    
    plt.grid()
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig(f"avg_battery_plot_{num_episodes-1}_{env.episode}_{proc_interval}_{w}_{num_agents}agents.pdf")
    plt.close()


def plot_battery_levels_individual(levels):
    window = 10
    # plt.suptitle("Multi-agent : battery levels")
    # plt.title(f"P_i = {power_idle}, P_f = {power_max}, fps = {proc_rate}, interval: {proc_interval}s")
    
    # plt.xlabel("Episodes")
    # plt.ylabel("Battery")
    
    for i in range(0, env._num_agents):
        plt.suptitle("Multi-agent : battery levels")
        plt.title(f"P_i = {power_idle}, P_f = {power_max}, fps = {proc_rate}, interval: {proc_interval}s")
        
        plt.xlabel("Episodes")
        plt.ylabel("Battery")
    
        # print(rewards[i])
        plt.plot(range(window - 1, len(levels[i])), np.convolve(levels[i], np.ones(window)/window, mode='valid'), label = f"smooth {battery_capacities[i]}Wh", alpha = 1.0)
        plt.plot(levels[i], label = f"raw {battery_capacities[i]}Wh", alpha = 0.3)
    
        plt.grid()
        plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"avg_battery_plot_{battery_capacities[i]}Wh_{num_episodes-1}_{env.episode}_{proc_interval}_{w}_{num_agents}agents.pdf")
        plt.close()

def plot_backlogs(backlogs):
    window = 10
    plt.suptitle("Multi-agent : average backlog")
    plt.title(f"P_i = {power_idle}, P_f = {power_max}, fps = {proc_rate}, interval: {proc_interval}s")
    
    plt.xlabel("Timesteps")
    plt.ylabel("Rewards")
    
    for i in range(0, env._num_agents):
        # print(rewards[i])
        plt.plot(range(window - 1, len(backlogs[i])), np.convolve(backlogs[i], np.ones(window)/window, mode='valid'), label = f"smooth {battery_capacities[i]}Wh", alpha = 1.0)
        plt.plot(backlogs[i], label = f"raw {battery_capacities[i]}Wh", alpha = 0.3)
    
    plt.grid()
    # plt.legend()
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig(f"backlog_plot_{num_episodes-1}_{env.episode}_{proc_interval}_{w}_{num_agents}agents.pdf")
    plt.close()

def plot_battery_daily(data):    
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
        plt.savefig(f"battery_{int(env.battery_capacities[elem] / 3600)}Wh_{num_episodes-1}_{env.episode}_{proc_interval}_{w}_{num_agents}agents.pdf")
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
        plt.savefig(f"backlog_{int(env.battery_capacities[elem] / 3600)}Wh_{num_episodes-1}_{env.episode}_{proc_interval}_{w}_{num_agents}agents.pdf")
        plt.close()        

def plot_framerate(fs):
    window = 10
    plt.suptitle("Multi-agent : average framerate")
    plt.title(f"P_i = {power_idle}, P_f = {power_max}, fps = {proc_rate}, interval: {proc_interval}s")
    
    plt.xlabel("Episodes")
    plt.ylabel("Framerate")
    
    for i in range(0, env._num_agents):
        # print(rewards[i])
        plt.plot(range(window - 1, len(fs[i])), np.convolve(fs[i], np.ones(window)/window, mode='valid'), label = f"smooth {battery_capacities[i]}Wh", alpha = 1.0)
        plt.plot(fs[i], label = f"raw {battery_capacities[i]}Wh", alpha = 0.3)
    
    plt.grid()
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig(f"framerate_plot_{num_episodes-1}_{env.episode}_{proc_interval}_{w}.pdf")
    plt.close()
    
def plot_local_framerate(fs):
    window = 10
    plt.suptitle("Multi-agent : local average framerate")
    plt.title(f"P_i = {power_idle}, P_f = {power_max}, fps = {proc_rate}, interval: {proc_interval}s")
    
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    
    for i in range(0, env._num_agents):
        # print(rewards[i])
        plt.plot(range(window - 1, len(fs[i])), np.convolve(fs[i], np.ones(window)/window, mode='valid'), label = f"smooth {battery_capacities[i]}Wh", alpha = 1.0)
        plt.plot(fs[i], label = f"raw {battery_capacities[i]}Wh", alpha = 0.3)
    
    plt.grid()
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig(f"local_framerate_plot_{num_episodes-1}_{env.episode}_{proc_interval}_{w}.pdf")
    plt.close()
    
def plot_offloading_framerate(fs):
    window = 10
    plt.suptitle("Multi-agent : offloading average framerate")
    plt.title(f"P_i = {power_idle}, P_f = {power_max}, fps = {proc_rate}, interval: {proc_interval}s")
    
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    
    for i in range(0, env._num_agents):
        # print(rewards[i])
        plt.plot(range(window - 1, len(fs[i])), np.convolve(fs[i], np.ones(window)/window, mode='valid'), label = f"smooth {battery_capacities[i]}Wh", alpha = 1.0)
        plt.plot(fs[i], label = f"raw {battery_capacities[i]}Wh", alpha = 0.3)
    
    plt.grid()
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    plt.tight_layout()
    plt.savefig(f"offloading_framerate_plot_{num_episodes-1}_{env.episode}_{proc_interval}_{w}.pdf")
    plt.close()

def decode(encoded_action):
    fti = int(encoded_action / (3 * num_agents * (proc_rate + 1)))
    r = int(encoded_action % (3 * num_agents * (proc_rate + 1)))
    
    xti = int(r / (num_agents * (proc_rate + 1)))
    r = int(r % ((num_agents * (proc_rate + 1))))
    
    gti = int(r / (proc_rate + 1))
    r = int(r % (proc_rate + 1))
        
    hti = r

    # print([fti, xti, gti, hti])
    
    return [fti, xti, gti, hti]

def get_epsilon(eps, eps_fin, eps_dec):
    
    if(eps <= eps_fin):
        return eps_fin
    else:
        return eps*eps_dec
    
env = CustomEnvironment(
        num_agents,
        irradiance_datapaths,
        delta_time,
        proc_interval,
        proc_rate,
        arrival_rate,
        battery_capacities,
        panel_surfaces,
        power_idle,
        power_max,
        w)

# env_wrapper = EnvWrapper(env)

models = {i : DQN(
                policy="MlpPolicy",
                env=EnvWrapper(env, i),
                learning_rate=0.0001,
                buffer_size=100000,
                learning_starts=500,
                batch_size=64,
                tau=1.0,
                gamma=0.99,
                train_freq=train_freq,
                gradient_steps=1,
                replay_buffer_class=None,
                replay_buffer_kwargs=None,
                optimize_memory_usage=False,
                n_steps=1,
                target_update_interval=((3600 / (proc_interval * 60))),
                exploration_fraction=1.0,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                max_grad_norm=1,
                stats_window_size=100,
                tensorboard_log=None,
                policy_kwargs=None,
                verbose=0,
                seed=None,
                device='auto',
                _init_setup_model=True
            )
          for i in range(0, num_agents)}

for i in range(num_agents):
    models[i].set_logger(configure(None, ["stdout"]))
    # models[i]._current_progress_remaining = 1.0 

rewards_plot = [[] for agent in range(0, num_agents)]    
batteries = [[] for agent in range(0, num_agents)]
batteries_local = [0 for i in range(0, num_agents)]
battery_daily = [[] for agent in range(0, num_agents)]

backlogs = [[] for agent in range(0, num_agents)]
backlogs_local = [0 for i in range(0, num_agents)]
backlogs_daily = [[] for agent in range(0, num_agents)]

fs = [[] for i in range(0, num_agents)]
hs = [[] for i in range(0, num_agents)]
framerates = [[] for i in range(0, num_agents)]
    
eps = eps_init

for i in range(0, num_episodes):
    
    # if i < 100:
    #     env._arrival_rate = 8
    # elif i < 1500:
    #     env._arrival_rate = 10
    # elif i < 2500:
    #     env._arrival_rate = 12
    # else:
    #     env._arrival_rate = 15
    
    obs = env.reset()

    rewards_episode = {agent: 0 for agent in range(num_agents)}
    obs = obs[0]
    
    battery_daily_temp = [[] for agent in range(0, num_agents)]
    backlog_daily_temp = [[] for agent in range(0, num_agents)]
    
    for agent_id in range(0, num_agents):
        batteries_local[agent_id] = 0
        backlogs_local[agent_id] = 0

    if(eps > eps_fin):
        eps *= eps_dec
        
    step = 0
    # print("obs prima di while: ", obs[0])
    total_timesteps = num_episodes * env.max_steps
        
    while env.agents:
        actions_encoded = {}
        actions = {}
        current_timestep = i * env.max_steps + step
        progress_remaining = 1.0 - (current_timestep / total_timesteps)
        
        # print(obs)
        
        for agent_id in range(0, num_agents):
            # print(obs)
            # print(f"obs agente: {agent_id}", obs[agent_id][agent_id])
            models[agent_id]._current_progress_remaining = progress_remaining

            action = 0
            
            if(np.random.random() < eps):
                action = models[agent_id].action_space.sample()
            else:
                action, _ = models[agent_id].predict(obs[agent_id], deterministic=False)

            # action = models[agent_id].action_space.sample()            
            # action, _ = models[agent_id].predict(obs[agent_id], deterministic=False)
            
            actions_encoded[agent_id] = action
            actions[agent_id] = decode(action)
            
        # input(actions)
            
        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        # print(f"Step: {step}")

        for agent_id in range(0, num_agents):
            done = terminations[agent_id] or truncations[agent_id]
            rewards_episode[agent_id] += rewards[agent_id]
            action_encoded = actions_encoded[agent_id]
            
            models[agent_id].replay_buffer.add(
                obs = obs[agent_id],
                next_obs = next_obs[agent_id],
                action = np.array([action_encoded]),
                reward = np.array(rewards[agent_id]),
                done = np.array([done]),
                infos = [{}]
            )
            
            # print(f" Agent {agent_id} -> [{actions[agent_id]} - {action_encoded}] , {rewards[agent_id]}")
            # 
            models[agent_id].num_timesteps += 1
            # models[agent_id].train(gradient_steps=1, batch_size=32)
            
            batteries_local[agent_id] += next_obs[agent_id][agent_id*3]
            # input(f"{next_obs[agent_id][0]} - {next_obs[agent_id][agent_id*3]} - {next_obs}")
            
            backlogs_local[agent_id] += env.backlogs[agent_id]
            
            if (models[agent_id].num_timesteps > models[agent_id].learning_starts and
                models[agent_id].num_timesteps % train_freq == 0):
                models[agent_id].train(gradient_steps=1, batch_size=64)
            
            if(i % int(num_episodes/10) == 0):
                battery_daily_temp[agent_id].append(env.battery_energies[agent_id]/env.battery_capacities[agent_id])
                backlog_daily_temp[agent_id].append(env.backlogs[agent_id])
            
        # for agent_id in range(0, num_agents):
        #     models[agent_id].train()
            # print(f"{agent_id} -> fs: {env.fs[agent_id]}")
            # print(f"{agent_id} -> hs: {env.hs[agent_id]}")
        # print(f"epsilon: {eps}")
    
        # input()
        obs = next_obs
        step += 1
    
        
    
    for agent_id in range(0, env._num_agents):

        fs[agent_id].append(env.fs[agent_id] / env.max_steps)
        # input(fs)
        if(env.hs_counter[agent_id] > 0):
            hs[agent_id].append(env.hs[agent_id] / env.hs_counter[agent_id])
        else:
            hs[agent_id].append(0.0)

        env.fs[agent_id] = 0
        env.hs[agent_id] = 0
        env.hs_counter[agent_id] = 0
        
        # input(f"{framerates[agent_id]} {fs[agent_id][-1]} {hs[agent_id][-1]}")
        framerates[agent_id].append(fs[agent_id][-1] + hs[agent_id][-1])
        
    print(f"Episode {i + 1}/{num_episodes} - rewards: {rewards_episode} - epsilon: {eps}")

    for agent_id in range(0, num_agents):
        rewards_plot[agent_id].append(rewards_episode[agent_id]) 
        batteries[agent_id].append(batteries_local[agent_id] / env.max_steps)        
        backlogs[agent_id].append(backlogs_local[agent_id] / env.max_steps)            

        if(i % int(num_episodes / 10) == 0):
            battery_daily[agent_id].append(battery_daily_temp[agent_id])
            backlogs_daily[agent_id].append(backlog_daily_temp[agent_id])
        
        
plot_rewards(rewards_plot)  
plot_backlogs(backlogs)
plot_battery_levels(batteries)
plot_backlog_daily(backlogs_daily)
plot_battery_daily(battery_daily)
plot_framerate(framerates)
plot_local_framerate(fs)
plot_offloading_framerate(hs)
        