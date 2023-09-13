import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class Policy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, max_action: float, layer_size: int, hidden_size: int, is_continuous) -> None:
        super(Policy, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.layer_size = layer_size
        self.hidden_size = hidden_size
        self.max_action = max_action
        self.is_continuous = is_continuous  
        self.policy = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_size)
        )
        self.net_block = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        for _ in range(self.layer_size - 1):
            self.policy.extend(self.net_block)
        
        self.mean = nn.Linear(hidden_size, self.action_dim)
        if self.is_continuous:
            self.log_var = nn.Parameter(torch.zeros(1, action_dim), requires_grad=True)
        
        
    def forward(self, state: torch.FloatTensor) -> (torch.Tensor, torch.Tensor):
        # state is tensor 
        # the dimension is batch_size * state_dim
        policy_output = self.policy(state)
        mean = self.mean(policy_output)
        if self.is_continuous:
            action_mean = torch.tanh(mean) * self.max_action
            prob_dist = dist.Normal(action_mean, self.log_var.expand_as(mean).exp().sqrt())
            return action_mean, prob_dist
        else:
            action_mean = F.softmax(mean, dim = 1)
            prob_dist = dist.Categorical(action_mean)  
            return action_mean, prob_dist
        
        
        
class QValue(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, layer_size: int, hidden_size: int):
        super(QValue, self).__init__()
        self.input_dim = obs_dim + action_dim
        self.out_dim = 1
        self.layer_size = layer_size
        self.hidden_size = hidden_size
        self.net_block = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.qf = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size)
        )
        for _ in range(self.layer_size - 1):
            self.qf.extend(self.net_block)
        self.qf.append(nn.Linear(hidden_size, 1))
        
    def forward(self, state: torch.FloatTensor, action: torch.FloatTensor) -> torch.Tensor:
        input = torch.cat((state, action), dim=-1)
        return self.qf(input)
    
    
class VValue(nn.Module):
    def __init__(self, obs_dim: int, layer_size: int, hidden_size: int):
        super(VValue, self).__init__()
        self.input_dim = obs_dim
        self.out_dim = 1
        self.layer_size = layer_size
        self.hidden_size = hidden_size
        self.net_block = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.vf = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size)
        )
        for _ in range(self.layer_size - 1):
            self.vf.extend(self.net_block)
        self.vf.append(nn.Linear(hidden_size, 1))
        
    def forward(self, state: torch.FloatTensor) -> torch.Tensor:
        return self.vf(state)