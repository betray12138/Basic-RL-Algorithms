import model.network as Net
import torch.optim as optim
import torch
import numpy as np
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from util.reward_norm import RewardScaling
from replay.replay import ReplayBuffer
import copy
import torch.nn as nn

class TD3(object):
	def __init__(self, device: str, state_dim: int, action_dim: int, max_action: float,
				 gamma: float, lr_policy: float, lr_critic: float,
				 layer_size: int, hidden_size: int, max_grad_norm: float, 
				 max_replay_size: int, batch_size:int, target_update_coee: float, action_noise_std: float, 
				 actor_update_freq: int, target_policy_noise: float, target_noise_clip: float) -> None:
		super(TD3, self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.max_action = max_action
		self.gamma = gamma
		self.lr_policy = lr_policy
		self.lr_critic = lr_critic
		self.layer_size = layer_size
		self.hidden_size = hidden_size
		self.device = device
		self.max_grad_norm = max_grad_norm
		self.batch_size = batch_size
		self.target_update_coee = target_update_coee	# denotes the update brought by source network
		self.action_noise_std = action_noise_std
  
		# TD3 tricks
		self.actor_update_freq = actor_update_freq
		self.target_policy_noise = target_policy_noise
		self.target_noise_clip = target_noise_clip
		self.critic_update_times = 0
  
		self.replay = ReplayBuffer(state_dim, action_dim, max_replay_size)
		
		self.actor = Net.Policy(obs_dim=state_dim,
								action_dim=action_dim,
								max_action = max_action,
								layer_size=layer_size,
								hidden_size=hidden_size,
								is_continuous=True).to(device)
		self.actor_target = copy.deepcopy(self.actor).to(device)
		
		self.critic1 = Net.QValue(obs_dim=state_dim,
								 action_dim=action_dim,
								 layer_size=layer_size,
								 hidden_size=hidden_size).to(device)
		self.critic2 = Net.QValue(obs_dim=state_dim,
								 action_dim=action_dim,
								 layer_size=layer_size,
								 hidden_size=hidden_size).to(device)
		self.critic_target1 = copy.deepcopy(self.critic1).to(device)
		self.critic_target2 = copy.deepcopy(self.critic2).to(device)
		
		self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_policy)
		self.critic_optimizer = optim.Adam([{"params": self.critic1.parameters()}, {"params": self.critic2.parameters()}], lr=self.lr_critic)
	
	def __train(self):
		self.actor.train()
		self.critic1.train()
		self.critic2.train()
	
	def __eval(self):
		self.actor.eval()
		self.critic1.eval()
		self.critic2.eval()
		
	def update(self):
		state, action, reward, next_state, terminal = self.replay.sample_to_tensor(self.batch_size, self.device)
		self.__train()
		
		# critic_loss:
		#	noise = gaussian() * target_policy_noise
		#	noise_a = \pi_target(s') + noise
		#   (r + \gamma min (Q1_target(s', noise_a), Q2_target(s', noise_a)) - Q(s,a)) ** 2
		# actor_loss:
		#   -Q1(s, \pi(s))   # noticed that Q do not update but \pi need to update, so need to freeze the parameter of Q1
		
		# update the critic
		with torch.no_grad():
			# trick1: target policy smoothing regularization
			noise = (torch.randn_like(action) * self.target_policy_noise).clamp(self.target_noise_clip, self.target_noise_clip).to(self.device)
			pi_target_a_, _ = self.actor_target(next_state)
			pi_target_a_ = (pi_target_a_ + noise).clamp(-self.max_action, self.max_action)
			# trick2: double Q architecture
			q_target_s_a_ = torch.min(self.critic_target1(next_state, pi_target_a_), self.critic_target2(next_state, pi_target_a_))  
			td_target = reward + self.gamma * q_target_s_a_ * (1 - terminal)
		qsa1, qsa2 = self.critic1(state, action), self.critic2(state, action)
		critic_loss = ((td_target - qsa1) ** 2).mean() + ((td_target - qsa2) ** 2).mean()
		
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		clip_grad_norm_(self.critic1.parameters(), self.max_grad_norm)
		clip_grad_norm_(self.critic2.parameters(), self.max_grad_norm)
		self.critic_optimizer.step()
  
		self.critic_update_times += 1
  
		# update the actor
		# trick3: delay actor update
		if self.critic_update_times % self.actor_update_freq == 0:
			for param in self.critic1.parameters():
				param.requires_grad = False
			
			pi_a, _ = self.actor(state)
			actor_loss = - self.critic1(state, pi_a).mean()
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
			self.actor_optimizer.step()
			
			for param in self.critic1.parameters():
				param.requires_grad = True
    
			# target update
			for param, target_param in zip(self.critic1.parameters(), self.critic_target1.parameters()):
				target_param.data.copy_(self.target_update_coee * param + (1 - self.target_update_coee) * target_param)
			for param, target_param in zip(self.critic2.parameters(), self.critic_target2.parameters()):
				target_param.data.copy_(self.target_update_coee * param + (1 - self.target_update_coee) * target_param)
			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.target_update_coee * param + (1 - self.target_update_coee) * target_param)
		
	def select_action(self, state: np.ndarray, is_evaluation=False) -> (np.ndarray, torch.FloatTensor):
		with torch.no_grad():
			state = torch.FloatTensor(state).to(self.device).reshape(-1, self.state_dim)
			action, _ = self.actor(state)	# deterministic policy
			return action.cpu().numpy()
	
	def evaluation(self, env, writter: SummaryWriter, steps=None):
		self.__eval()
		episode_reward = 0
		state = env.reset()
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
		# path denotes the directory location end with /
		torch.save(self.actor.state_dict(), path + "actor_" + str(steps) + ".pth")
		torch.save(self.critic1.state_dict(), path + "critic1_" + str(steps) + ".pth")
		torch.save(self.critic2.state_dict(), path + "critic2_" + str(steps) + ".pth")
		
			
	def train(self, env, env_test, writter: SummaryWriter, max_train_steps: int, random_steps: int,
			  save_interval: int, log_interval: int, saving_path: str):
		# random steps used to collect the initial dataset
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
				if cur_step <= random_steps:
					action = env.action_space.sample()	# 随机采样动作
				else:
					action = self.select_action(state)
					# DDPG use action noise to ensure exploration
					action = (action + np.random.normal(0, self.action_noise_std, size=self.action_dim)).clip(-self.max_action, self.max_action)
				next_state, reward, done, truncated = env.step(action)
				episode_reward += reward
				self.replay.store(s=state,
								  a=action,
								  r=np.array([reward]),
								  s_=next_state,
								  dw=np.array([done]))
				state = next_state.flatten()
	
				if cur_step > random_steps:
					# begin to update the network
					# of course, you can choose to update the model multiple times
					self.update()
	
			print("Episode: " + str(train_episodes) + " training return: " + str(episode_reward))
			writter.add_scalar("training/return", episode_reward, cur_step)
			
			if train_episodes % log_interval == 0:
				self.evaluation(env_test, writter, cur_step)
			
			if train_episodes % save_interval == 0:
				self.save_model(saving_path, cur_step)
  