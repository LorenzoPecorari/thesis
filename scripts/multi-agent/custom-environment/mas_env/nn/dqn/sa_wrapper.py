import gymnasium as gym
from gymnasium import spaces

class SingleAgentWrapper(gym.Env):
    
    def __init__(self, ma_env, agent_id, other_agents):
        super().__init__()
        
        self.env = ma_env
        self.agent_id = agent_id
        self.other_agents = other_agents
        
        self.observation_space = ma_env._observation_spaces[agent_id]
        self.action_space = ma_env._action_spaces[agent_id]
        
    def reset(self):
        obs_dict, infos_dict = self.env.reset()
        
        agent_obs = obs_dict[self.agent_id]
        agent_infos = infos_dict[self.agent_id]
        
        return agent_obs, agent_infos
    
    def step(self, action):
        actions = {self.agent_id: action}
        
        for other_agent in self.env.possible_agents:
            if(other_agent == self.agent_id):
                continue
            
            if(other_agent in self.other_agents):
                pass