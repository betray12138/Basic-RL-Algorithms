import gymnasium as gym
from tensorboardX import SummaryWriter
import argparse
import os
import torch
import numpy as np
import datetime
from util.reward_norm import RewardScaling

from env.env_wrapper import GymNasiumWrapper
from algo.a2c_continuous import A2C_Continuous


parser = argparse.ArgumentParser(description='PyTorch A2C Continuous')
parser.add_argument('--gamma', type=float, default=0.9, metavar='G',
					help='discount factor (default: 0.9)')
parser.add_argument('--seed', type=int, default=0, metavar='N',
					help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
					help='interval between training status logs (default: 10)')
parser.add_argument('--save-interval', type=int, default=100, metavar='N',
					help='interval between saving the model (default: 100)')
parser.add_argument('--lr-policy', type=float, default=4e-4, metavar='N',
					help='the learning rate for training the policy')
parser.add_argument('--lr-critic', type=float, default=4e-3, metavar='N',
					help='the learning rate for training the critic')
parser.add_argument('--layer-size', type=float, default=2, metavar='N',
					help='the layer size of network')
parser.add_argument('--hidden-size', type=float, default=64, metavar='N',
					help='the hidden size of network')
parser.add_argument('--env-name', type=str, default="Pendulum-v1", metavar='N',
					help='the env name')
parser.add_argument('--max-train-steps', type=int, default=1000000, metavar='N',
					help='the total training steps')
parser.add_argument('--max-grad-norm', type=float, default=0.5, metavar='N',
					help='gradient norm used to prevent gradient explosion')
parser.add_argument('--max-replay-size', type=float, default=32, metavar='N',
					help='maximum replay size to store the transition')
args = parser.parse_args()

env_unwrapped = gym.make(args.env_name)
env_unwrapped_test = gym.make(args.env_name)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

env = GymNasiumWrapper(env_unwrapped)
env_test = GymNasiumWrapper(env_unwrapped_test)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

rewardScaling = RewardScaling(args.gamma)

agent = A2C_Continuous(device=device,
                       state_dim=env.observation_space.shape[0],
                       action_dim=env.action_space.shape[0],
                       max_action=float(env.action_space.high[0]),
                       gamma=args.gamma,
                       lr_policy=args.lr_policy,
                       lr_critic=args.lr_critic,
                       layer_size=args.layer_size,
                       hidden_size=args.hidden_size,
                       max_grad_norm=args.max_grad_norm,
                       max_replay_size=args.max_replay_size)

common_path = "./logs/" + agent.__class__.__name__ + "/" + args.env_name + "/seed_" + str(args.seed) + "_" + \
    datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + "/"
save_model_path = common_path + "model/"
log_path = common_path

if not os.path.exists(log_path):    
	os.makedirs(log_path)
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)
writer = SummaryWriter(log_path)

agent.train(env, env_test, writer, args.max_train_steps, args.save_interval, args.log_interval, save_model_path)