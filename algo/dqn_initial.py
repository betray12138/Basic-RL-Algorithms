import model.network as Net
import torch.optim as optim
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from util.reward_norm import RewardScaling
from replay.replay import ReplayBuffer
import copy
import torch.nn as nn

class DQN(object):
	def __init__(self, device: str, state_dim: int, action_dim: int, gamma: float, 
				 lr_policy: float, layer_size: int, hidden_size: int, max_grad_norm: float, 
				 max_replay_size: int, batch_size:int, target_update_coee: float, target_interval: int,
     			 epsilon_decay: float, epsilon_max: float, epsilon_min: float) -> None:
		super(DQN, self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.gamma = gamma
		self.lr_policy = lr_policy
		self.layer_size = layer_size
		self.hidden_size = hidden_size
		self.device = device
		self.max_grad_norm = max_grad_norm
		self.batch_size = batch_size
		self.target_update_coee = target_update_coee	# denotes the update brought by source network
		self.replay = ReplayBuffer(state_dim, 1, max_replay_size)
		self.epsilon_decay = epsilon_decay
		self.epsilon_max = epsilon_max
		self.epsilon_min = epsilon_min
		self.target_interval = target_interval
		self.update_step = 0
		self.epsilon = epsilon_max
		
		# Due to DQN needs to output \max_a Q(s', a'), we module this as multiple outputs corresponding to action dimensions.
		self.critic = Net.Policy(obs_dim=state_dim,
								 action_dim=action_dim,
								 max_action=None,
								 layer_size=layer_size,
								 hidden_size=hidden_size,
								 is_continuous=False).to(device)
		self.critic_target = copy.deepcopy(self.critic).to(device)
		
		self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_policy)
	
	def __train(self):
		self.critic.train()
		self.update_step += 1
	
	def __eval(self):
		self.critic.eval()
		
	def update(self):
		state, action, reward, next_state, terminal = self.replay.sample_to_tensor(self.batch_size, self.device)
		self.__train()
		
		# loss: td-target
		# (r_i + \gamma * \max_a Q_{target}(s', a) - Q(s,a)) ^ 2
		with torch.no_grad():
			# max will return a tuple with two element, for which we need the first element
			q_target = self.critic_target(next_state)[0].max(dim=1, keepdim=True)[0]
			td_target = reward + self.gamma * q_target * (1. - terminal)

		# gather used to collect the value according to the action
		critic_loss = ((td_target - self.critic(state)[0].gather(1, action.to(torch.int64))) ** 2).mean()

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

		self.critic_optimizer.step()
		
		# target update
		if self.update_step % self.target_interval == 0:
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.target_update_coee * param + (1 - self.target_update_coee) * target_param)
		
	def select_action(self, state: np.ndarray, is_evaluation=False) -> (np.ndarray, torch.FloatTensor):
		with torch.no_grad():
			state = torch.FloatTensor(state).to(self.device).reshape(-1, self.state_dim)
			action_mean, _ = self.critic(state)	
			return np.array([np.argmax(action_mean.detach().cpu().numpy())]) 
	
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
			while not done and not truncated:
				cur_step += 1
				# use episode-greedy to ensure exploration / policy improvement guarantee / linear regret bound
				if cur_step <= random_steps or self.epsilon > np.random.random():
					action = np.array([env.action_space.sample()])	# 随机采样动作
				else:
					action = self.select_action(state)
	 
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

					self.epsilon = max(self.epsilon_min, self.epsilon - (self.epsilon_max - self.epsilon_min) * self.epsilon_decay)

	
			print("Episode: " + str(train_episodes) + " training return: " + str(episode_reward))
			writter.add_scalar("training/return", episode_reward, cur_step)
			
			if train_episodes % log_interval == 0:
				self.evaluation(env_test, writter, cur_step)
			
			if train_episodes % save_interval == 0:
				self.save_model(saving_path, cur_step)
  