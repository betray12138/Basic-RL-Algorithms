import numpy as np
import torch
from util.segment_tree import SumTree

class ReplayBuffer(object):
    def __init__(self, state_dim: int, action_dim: int, max_size: int):
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
        
    def sample_with_index_to_tensor(self, index: np.ndarray, device: str):
        batch_s = torch.FloatTensor(self.state[index]).to(device)
        batch_a = torch.FloatTensor(self.action[index]).to(device)
        batch_r = torch.FloatTensor(self.reward[index]).to(device)
        batch_s_ = torch.FloatTensor(self.next_state[index]).to(device)
        batch_dw = torch.FloatTensor(self.terminal[index]).to(device)
        return batch_s, batch_a, batch_r, batch_s_, batch_dw

    def sample_to_tensor(self, batch_size: int, device: str):
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
        return self.__sample_with_index_to_tensor(index, device)
    
    def sample_all_to_tensor(self, device: str):
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
        
        
class PER(ReplayBuffer):
    def __init__(self, state_dim: int, action_dim: int, max_size: int,
                 alpha: float, beta: float):
        super(PER, self).__init__(state_dim, action_dim, max_size)
        self.sum_tree = SumTree(self.max_size)
        
        # PER weight
        self.alpha = alpha
        self.beta = beta

    def store(self, s: np.ndarray, a: np.ndarray, r: np.ndarray, s_: np.ndarray, dw: np.ndarray):
        # for the first transiton, the priority would be set to 1.0; 
        # otherwise be set to max priority in the sum_tree to ensure the transition can be sampled
        priority = 1.0 if self.size == 0 else self.sum_tree.priority_max
        self.sum_tree.update_priority(buffer_index=self.count, priority=priority)
        
        super().store(s, a, r, s_, dw)
    
    def sample_to_tensor(self, batch_size: int, device: str):
        index, IS_weight = self.sum_tree.prioritized_sample(N=self.size,
                                                            batch_size=batch_size,
                                                            beta=self.beta)
        # index does not need to compute on gpu, but the IS_weight need
        batch_s, batch_a, batch_r, batch_s_, batch_dw = self.sample_with_index_to_tensor(index, device)
        return batch_s, batch_a, batch_r, batch_s_, batch_dw, index, torch.FloatTensor(IS_weight).reshape(batch_size, -1).to(device)
    
    def update_priority_batch(self, batch_index: np.ndarray, td_error: torch.Tensor):
        priorities_in_batch = (np.abs(td_error.detach().cpu().numpy()) + 1e-6) ** self.alpha
        # Note that the `batch_index` and  `priorities_in_batch` must have the same dimension
        for idx, priority in zip(batch_index, priorities_in_batch.flatten()):
            self.sum_tree.update_priority(idx, priority)
        
    def adjust_beta(self, beta: float):
        self.beta = beta
