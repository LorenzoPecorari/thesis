import gymnasium
from gymnasium.spaces import Box, Discrete, Dict
import numpy as np

class EnvWrapper(gymnasium.Env):
    def __init__(self, env, agent_id):
        self.env = env
        self.agent_id = agent_id
        
        # print(env._observation_spaces)
        
        # for elem in env._observation_spaces:
        #     print(env._observation_spaces[elem], type(env._observation_spaces[elem]))
        
        self.observation_space = env._observation_spaces[agent_id]
        
        # mixed radix representation !!!
        self.action_space = Discrete((env._processing_rate + 1) * 3 * env._num_agents * (env._processing_rate + 1))
        
        # self.state_shape = (env._num_agents * 3)
        # self.action_shape = (env._processing_rate + 1) * 3 * env._num_agents * (env._processing_rate + 1)

        # self.action_shape = (env._processing_rate + 1, 3, env._num_agents, env._processing_rate + 1)
        
        
    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=None, options=None)
        return obs[self.agent_id], infos[self.agent_id]
    
    def step(self, action):
        actions = {self.agent_id: action}
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        
        return (obs[self.agent_id], 
                rewards[self.agent_id], 
                terminations[self.agent_id], 
                truncations[self.agent_id], 
                infos[self.agent_id])
    
    def render(self):
        pass
    
    def close(self):
        pass