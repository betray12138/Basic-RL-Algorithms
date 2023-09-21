import model.network as Net
import torch.optim as optim
import torch
import numpy as np
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from util.reward_norm import RewardScaling, Normalization
from replay.replay import ReplayBuffer

class Reinforce(object):
    def __init__(self, device: str, state_dim: int, action_dim: int,
                 gamma: float, lr_policy: float, lr_critic: float,
                 layer_size: int, hidden_size: int, max_grad_norm: float, 
                 max_replay_size: int) -> None:
        super(Reinforce, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr_policy = lr_policy
        self.lr_critic = lr_critic
        self.layer_size = layer_size
        self.hidden_size = hidden_size
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.replay = ReplayBuffer(state_dim, 1, max_replay_size)   # 离散空间动作被reshape成1
        
        self.actor = Net.Policy(obs_dim=state_dim,
                                action_dim=action_dim,
                                max_action = None,
                                layer_size=layer_size,
                                hidden_size=hidden_size,
                                is_continuous=False).to(device)
    
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_policy)
    
    def __train(self):
        self.actor.train()
    
    def __eval(self):
        self.actor.eval()
        
    def update(self):
        state, action, reward, next_state, terminal = self.replay.sample_all_to_tensor(self.device)
        self.__train()
        
        # compute G_t and update policy network
        g_t = torch.zeros_like(reward).to(self.device)
        for i in reversed(range(g_t.shape[0])):
            g_t[i] = (0 if i >= g_t.shape[0] - 1 else g_t[i + 1]) * self.gamma + reward[i]
        
        # 更新actor 计算actor损失 
        # actor_loss = - G_t * \log \pi(a|s)
        _, actor_dist = self.actor(state)
        log_prob = actor_dist.log_prob(action)
        actor_loss = - (g_t * log_prob).mean()  
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        
        self.actor_optimizer.step()
        
    
    def select_action(self, state: np.ndarray, is_evaluation=False) -> (np.ndarray, torch.FloatTensor):
        # 返回action 和 log_prob
        # 此处log_prob需要保持梯度，否则无法回传
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).reshape(-1, self.state_dim)
            mean, action_dist = self.actor(state)
            action = action_dist.sample()
            if is_evaluation:
                return np.array([np.argmax(mean.detach().cpu().numpy())])
            return action.cpu().numpy()
    
    def evaluation(self, env, writter: SummaryWriter, steps=None):
        self.__eval()
        episode_reward = 0
        state = env.reset() # 此时state是numpy
        done = False
        truncated = False
        while not done and not truncated:
            action = self.select_action(state, True)
            next_state, reward, done, truncated = env.step(action)
            episode_reward += reward
            state = next_state.flatten()
        if steps:
            writter.add_scalar("evaluation/return", episode_reward, steps)
        return episode_reward
    
    def save_model(self, path: str, steps: int):
        # 此处path传入目录 以 /结尾
        torch.save(self.actor.state_dict(), path + "actor_" + str(steps) + ".pth")
        
            
    def train(self, env, env_test, writter: SummaryWriter, max_train_steps: int, 
              save_interval: int, log_interval: int, saving_path: str):
        cur_step = 0
        train_episodes = 0
        while cur_step < max_train_steps:
            train_episodes += 1
            state = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            while not done and not truncated:
                cur_step += 1
                action = self.select_action(state)
                next_state, reward, done, truncated = env.step(action)
                episode_reward += reward
                self.replay.store(s=state,
                                  a=action,
                                  r=np.array([reward]),
                                  s_=next_state,
                                  dw=np.array([done]))
                state = next_state.flatten()
                
            print("Episode: " + str(train_episodes) + " training return: " + str(episode_reward))
            writter.add_scalar("training/return", episode_reward, cur_step)
            
            self.update()
            self.replay.clear()
            
            if train_episodes % log_interval == 0:
                self.evaluation(env_test, writter, cur_step)
            
            if train_episodes % save_interval == 0:
                self.save_model(saving_path, cur_step)
  