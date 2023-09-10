import model.network as Net
import torch.optim as optim
import torch
import numpy as np
import torch.distributions as dist
import gym
from torch.utils.tensorboard import SummaryWriter

class A2C_Continuous(object):
    def __init__(self, device: str, state_dim: int, action_dim: int, 
                 gamma: float, lr_policy: float, lr_critic: float,
                 layer_size: int, hidden_size: int) -> None:
        super(A2C_Continuous, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr_policy = lr_policy
        self.lr_critic = lr_critic
        self.layer_size = layer_size
        self.hidden_size = hidden_size
        self.device = device
        self.time_discount = 1
        
        self.actor = Net.Policy(obs_dim=state_dim,
                                action_dim=action_dim,
                                layer_size=layer_size,
                                hidden_size=hidden_size,
                                is_continuous=True).to(device)
        
        self.critic = Net.VValue(obs_dim=state_dim,
                                 layer_size=layer_size,
                                 hidden_size=hidden_size).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_policy)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
    
    def __train(self):
        self.actor.train()
        self.critic.train()
        
    def update(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray,
               reward: np.ndarray, terminal: np.ndarray, log_prob: float):
        
        state = torch.FloatTensor(state).to(self.device).reshape(-1, self.state_dim)
        action = torch.FloatTensor(action).to(self.device).reshape(-1, self.action_dim)
        next_state = torch.FloatTensor(next_state).to(self.device).reshape(-1, self.state_dim)
        reward = torch.FloatTensor(reward).to(self.device).reshape(-1, 1)
        terminal = torch.FloatTensor(terminal).to(self.device).reshape(-1, 1)
        log_prob = torch.FloatTensor(log_prob).to(self.device)
        
        self.__train()
        
        # compute necessary value
        v_s = self.critic(state)
        
        with torch.no_grad():
            v_s_ = self.critic(next_state)
            td_target = reward + self.gamma * v_s_ * (1 - terminal)
        
        # 更新actor 计算actor损失 
        # actor_loss = - advantage * \log \pi(a|s) * time_discount
        # actor_loss = - (r + \gamma V(s') - V(s)) * \log \pi(a|s)
        actor_loss = - (td_target - v_s).detach() * log_prob * self.time_discount    # .detach()用于分离梯度
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新critic 计算critic损失
        critic_loss = (td_target - v_s) ** 2
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新time_discount 沿trajectory衰减  
        self.time_discount *= self.gamma
    
    def select_action(self, state: np.ndarray) -> (np.ndarray, float):
        # 返回action 和 log_prob
        state = torch.FloatTensor(state).to(self.device).reshape(-1, self.state_dim)
        with torch.no_grad():
            mean, logvar = self.actor(state)
        action_dist = dist.Normal(mean, logvar)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.cpu().numpy(), log_prob.cpu()
    
    def evaluation(self, env: gym.Env, writter: SummaryWriter, steps=None):
        episode_reward = 0
        state = env.reset() # 此时state是numpy
        done = False
        while not done:
            action, _ = self.select_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        if steps:
            writter.add_scalar("evaluation/return", episode_reward, steps)
        return episode_reward
    
    def save_model(self, path: str, steps: int):
        # 此处path传入目录 以 /结尾
        torch.save(self.actor.state_dict(), path + "actor_" + str(steps) + ".pth")
        torch.save(self.critic.state_dict(), path + "critic_" + str(steps) + ".pth")
        
            
    def train(self, env: gym.Env, env_test: gym.Env, writter: SummaryWriter, max_train_steps: int, 
              save_interval: int, log_interval: int, saving_path: str):
        cur_step = 0
        train_episodes = 0
        while cur_step < max_train_steps:
            train_episodes += 1
            state = env.reset()
            done = False
            self.time_discount = 1  #每次episode完毕后重置
            while not done:
                cur_step += 1
                
                action, log_prob = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                
                self.update(state, action, next_state, reward, done, log_prob)
                state = next_state
            
            if train_episodes % log_interval == 0:
                self.evaluation(env, writter, cur_step)
            
            if train_episodes % save_interval == 0:
                self.save_model(saving_path, cur_step)