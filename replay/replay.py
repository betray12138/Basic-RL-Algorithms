import numpy as np
import torch

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clear()

    def store(self, s: np.ndarray, a: np.ndarray, r: np.ndarray, s_: np.ndarray, dw: np.ndarray):
        self.state[self.count] = s.flatten()
        self.action[self.count] = a.flatten()
        self.reward[self.count] = r.flatten()
        self.next_state[self.count] = s_.flatten()
        self.terminal[self.count] = dw.flatten()
        self.count = (self.count + 1) % self.max_size  
        self.size = min(self.size + 1, self.max_size)  

    def sample_to_tensor(self, batch_size: int, device: str):
        # 一般用于off-policy算法
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
        batch_s = torch.FloatTensor(self.state[index]).to(device)
        batch_a = torch.FloatTensor(self.action[index]).to(device)
        batch_r = torch.FloatTensor(self.reward[index]).to(device)
        batch_s_ = torch.FloatTensor(self.next_state[index]).to(device)
        batch_dw = torch.FloatTensor(self.terminal[index]).to(device)

        return batch_s, batch_a, batch_r, batch_s_, batch_dw
    
    def sample_all_to_tensor(self, device: str):
        # 一般用于on-policy算法 诸如A2C
        index = np.array([i for i in range(self.size)])
        batch_s = torch.FloatTensor(self.state[index]).to(device)
        batch_a = torch.FloatTensor(self.action[index]).to(device)
        batch_r = torch.FloatTensor(self.reward[index]).to(device)
        batch_s_ = torch.FloatTensor(self.next_state[index]).to(device)
        batch_dw = torch.FloatTensor(self.terminal[index]).to(device)

        return batch_s, batch_a, batch_r, batch_s_, batch_dw
    
    def clear(self):
        self.count = 0
        self.size = 0
        self.state = np.zeros((self.max_size, self.state_dim))
        self.action = np.zeros((self.max_size, self.action_dim))
        self.reward = np.zeros((self.max_size, 1))
        self.next_state = np.zeros((self.max_size, self.state_dim))
        self.terminal = np.zeros((self.max_size, 1))