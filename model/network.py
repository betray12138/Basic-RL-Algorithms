import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np

class Policy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, max_action: float, 
                 layer_size: int, hidden_size: int, is_continuous: bool) -> None:
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
        
        
class DQNPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, layer_size: int, hidden_size: int, use_noisy: bool):
        super(DQNPolicy, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.layer_size = layer_size
        self.hidden_size = hidden_size
        self.policy = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_size)
        )
        self.net_block = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        for _ in range(self.layer_size - 1):
            self.policy.extend(self.net_block)
        
        # 5. Noisy-DQN
        if use_noisy:
            self.qf = NoisyLinearNet(hidden_size, self.action_dim)
        else:
            self.qf = nn.Linear(hidden_size, self.action_dim)
        
        
    def forward(self, state: torch.FloatTensor) -> torch.Tensor:
        # state is tensor 
        # the dimension is batch_size * state_dim
        policy_output = self.policy(state)
        return self.qf(policy_output)
    
        
class DuelingPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, layer_size: int, hidden_size: int, use_noisy: bool):
        super(DuelingPolicy, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.layer_size = layer_size
        self.hidden_size = hidden_size
        self.policy = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_size)
        )
        self.net_block = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        for _ in range(self.layer_size - 1):
            self.policy.extend(self.net_block)
        
        if use_noisy:
            self.vs = NoisyLinearNet(hidden_size, 1)
            self.adv = NoisyLinearNet(hidden_size, self.action_dim)
        else:
            self.vs = nn.Linear(hidden_size, 1)
            self.adv = nn.Linear(hidden_size, self.action_dim)
        
        
    def forward(self, state: torch.FloatTensor) -> torch.Tensor:
        # state is tensor 
        # the dimension is batch_size * state_dim
        policy_output = self.policy(state)
        
        # expand used to extend the dimension into batch_size * self.action_dim
        vs = self.vs(policy_output).expand(state.shape[0], self.action_dim) 
        adv = self.adv(policy_output)
        
        # the dimension of adv is batch_size * self.action_dim
        # adv.mean(1) is used to compute the mean of dimension 1, and this will cause the dimension decreasement
        # use unsqueeze(1) to add the new dimension before dimension 2
        action_val = vs + adv - adv.mean(1).unsqueeze(1).expand(state.shape[0], self.action_dim)
        return action_val   
        
        
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
    

class NoisyLinearNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, sigma_init: float = 0.4) -> None:
        super(NoisyLinearNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma_init = sigma_init
        
        # here we choose to implement [independent gaussian noise]
        self.w_mu = nn.Parameter(torch.FloatTensor(self.out_dim, self.in_dim))
        self.w_sigma = nn.Parameter(torch.FloatTensor(self.out_dim, self.in_dim))
        self.w_noise = nn.Parameter(torch.FloatTensor(self.out_dim, self.in_dim), requires_grad=False)
        
        self.b_mu = nn.Parameter(torch.FloatTensor(out_dim))
        self.b_sigma = nn.Parameter(torch.FloatTensor(out_dim))
        self.b_noise = nn.Parameter(torch.FloatTensor(out_dim), requires_grad=False)
        
        self.set_parameters()   # parameters only be set once
    
    def set_parameters(self):
        # set mu parameter
        mu_range = np.sqrt(3 / self.in_dim)
        self.w_mu.data.uniform_(-mu_range, mu_range)
        self.b_mu.data.uniform_(-mu_range, mu_range)
        
        # set sigma parameter
        self.w_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_dim))
        self.b_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_dim))
    
    def __compute_noise_generate_func(self, size: int):
        # use sgn(x)\sqrt{|x|}, where x is subjected to standard gaussian distribution
        x = torch.randn(size)   # generate the col vector, which is subjected to standard gaussian distribution
        return x.sign().mul(x.abs().sqrt())
        
    def reset_noise(self):
        # call every training steps
        epsilon_i = self.__compute_noise_generate_func(self.in_dim)
        epsilon_j = self.__compute_noise_generate_func(self.out_dim)
        
        # use torch.ger to implement dot mul
        self.w_noise.copy_(torch.ger(epsilon_j, epsilon_i))
        self.b_noise.copy_(epsilon_j)
        
    def forward(self, x: torch.FloatTensor):
        # the noise cannot be used during evaluation:
        if self.training:
            self.reset_noise()
            weight = self.w_mu + self.w_sigma.mul(self.w_noise)
            bias = self.b_mu + self.b_sigma.mul(self.b_noise)
        else:
            weight = self.w_mu
            bias = self.b_mu
        return F.linear(x, weight, bias)