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

class DDPG(object):
	def __init__(self, device: str, state_dim: int, action_dim: int, max_action: float,
				 gamma: float, lr_policy: float, lr_critic: float,
				 layer_size: int, hidden_size: int, max_grad_norm: float, 
				 max_replay_size: int, batch_size:int, target_update_coee: float, action_noise_std: float,
	 			 rewardScaling: RewardScaling = None) -> None:
		super(DDPG, self).__init__()
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
		self.replay = ReplayBuffer(state_dim, action_dim, max_replay_size)
		self.rewardScale = rewardScaling
		
		self.actor = Net.Policy(obs_dim=state_dim,
								action_dim=action_dim,
								max_action = max_action,
								layer_size=layer_size,
								hidden_size=hidden_size,
								is_continuous=True).to(device)
		self.actor_target = copy.deepcopy(self.actor).to(device)
		
		self.critic = Net.QValue(obs_dim=state_dim,
								 action_dim=action_dim,
								 layer_size=layer_size,
								 hidden_size=hidden_size).to(device)
		self.critic_target = copy.deepcopy(self.critic).to(device)
		
		self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_policy)
		self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
	
	def __train(self):
		self.actor.train()
		self.critic.train()
	
	def __eval(self):
		self.actor.eval()
		self.critic.eval()
		
	def update(self):
		state, action, reward, next_state, terminal = self.replay.sample_to_tensor(self.batch_size, self.device)
		self.__train()
		
		# critic_loss:
		#   (r + \gamma Q_target(s', \pi_target(s')) - Q(s,a)) ** 2
		# actor_loss:
		#   -Q(s, \pi(s))   # noticed that Q do not update but \pi need to update, so need to freeze the parameter of Q
		
		# update the actor first, or the updated critic will influence the actor gradient 
		# though this does not make big difference
		for param in self.critic.parameters():
			param.requires_grad = False
		
		pi_a, _ = self.actor(state)
		actor_loss = - self.critic(state, pi_a).mean()
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
		self.actor_optimizer.step()
		
		for param in self.critic.parameters():
			param.requires_grad = True
		
		# update the critic
		with torch.no_grad():
			pi_target_a_, _ = self.actor_target(next_state)
			q_target_s_a_ = self.critic_target(next_state, pi_target_a_)
			td_target = reward + self.gamma * q_target_s_a_ * (1 - terminal)
		qsa = self.critic(state, action)
		critic_loss = ((td_target - qsa) ** 2).mean()
		
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

		self.critic_optimizer.step()
		
		# target update
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
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
		torch.save(self.critic.state_dict(), path + "critic_" + str(steps) + ".pth")
		
			
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
			if self.rewardScale:
				self.rewardScale.reset()
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
				if self.rewardScale:
					reward = self.rewardScaling.get_and_update(reward)
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
  