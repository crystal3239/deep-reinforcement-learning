import numpy as np
import ddpg_agent
import importlib
importlib.reload(ddpg_agent)
import random

class MADDPG:
    
    def __init__(self, state_size, action_size, agent_num, random_seed=2):
        self.state_size = state_size
        self.action_size = action_size
        self.agent_num = agent_num
        self.seed = random.seed(random_seed)
        
        self.memory = ddpg_agent.ReplayBuffer(action_size, ddpg_agent.BUFFER_SIZE, ddpg_agent.BATCH_SIZE, random_seed)
        self.agents = [ddpg_agent.Agent(state_size, action_size, agent_num, random_seed) for _ in range(agent_num)]
        self.losses = (0., 0.)
        
    def reset(self):
        for agent in self.agents: agent.reset()
            
    def act(self, states, add_noise = True):
        return [agent.act(state, add_noise) for agent, state in zip(self.agents, states)]
    
    def step(self, states, actions, rewards, next_states, dones):
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) < ddpg_agent.BATCH_SIZE: return
        critic_losses = []
        actor_losses = []
        for agent in self.agents:
            experiences = self.memory.sample()
            critic_loss, actor_loss = agent.learn(experiences, ddpg_agent.GAMMA)
            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)
        self.losses = (np.mean(critic_losses), np.mean(actor_losses))