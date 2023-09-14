import gymnasium as gym
from tensorboardX import SummaryWriter
import argparse
import os
import torch
import numpy as np
import datetime

from env.env_wrapper import GymNasiumWrapper
from algo.dqn_rainbow import RainBow


parser = argparse.ArgumentParser(description='PyTorch DQN initial')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
					help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=123, metavar='N',
					help='random seed (default: 121)')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
					help='interval between training status logs (default: 10)')
parser.add_argument('--save-interval', type=int, default=100, metavar='N',
					help='interval between saving the model (default: 100)')
parser.add_argument('--lr-policy', type=float, default=1e-4, metavar='N',
					help='the learning rate for training the policy')
parser.add_argument('--layer-size', type=float, default=3, metavar='N',
					help='the layer size of network')
parser.add_argument('--hidden-size', type=float, default=256, metavar='N',
					help='the hidden size of network')
parser.add_argument('--env-name', type=str, default="CartPole-v1", metavar='N',
					help='the env name')
parser.add_argument('--max-train-steps', type=int, default=5000000, metavar='N',
					help='the total training steps')
parser.add_argument('--max-grad-norm', type=float, default=10.0, metavar='N',
					help='gradient norm used to prevent gradient explosion')
parser.add_argument('--max-replay-size', type=float, default=50000, metavar='N',
					help='maximum replay size to store the transition')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
					help='batch size used to train the network')
parser.add_argument('--target-update-coefficient', type=float, default=0.005, metavar='N',
					help='coefficient used to soft update the target network')
parser.add_argument('--random-steps', type=float, default=10000, metavar='N',
					help='the steps used to collect the initial dataset')
parser.add_argument('--target-interval', type=int, default=1, metavar='N',
					help='the frequency of updating the target network')
parser.add_argument('--epsilon-decay', type=float, default=5e-4, metavar='N',
					help='the frequency of updating the target network')
parser.add_argument('--epsilon-max', type=int, default=0.5, metavar='N',
					help='the frequency of updating the target network')
parser.add_argument('--epsilon-min', type=int, default=0.1, metavar='N',
					help='the frequency of updating the target network')


# RainBow Tricks
# 1. double DQN
parser.add_argument('--use-double', type=bool, default=True, metavar='N',
					help='whether to use double network tricks')

# 2. PER
parser.add_argument('--use-per', type=bool, default=False, metavar='N',
					help='whether to use PER')
parser.add_argument('--prop-alpha', type=float, default=0.6, metavar='N',
					help='the power of computing the p(i)')
parser.add_argument('--weight-beta', type=float, default=0.4, metavar='N',
					help='the initial weight of beta')
parser.add_argument('--gain-beta-steps', type=int, default=5e5, metavar='N',
					help='the steps for beta changes to 1')

# 3. Dueling DQN
parser.add_argument('--use-duel', type=bool, default=False, metavar='N',
					help='whether to use dueling network tricks')


args = parser.parse_args()

env_unwrapped = gym.make(args.env_name, args.seed)
env_unwrapped_test = gym.make(args.env_name, args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

env = GymNasiumWrapper(env_unwrapped, False)
env_test = GymNasiumWrapper(env_unwrapped_test, False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

agent = RainBow(device=device,
             state_dim=env.observation_space.shape[0],
             action_dim=env.action_space.n,
             gamma=args.gamma,
             lr_policy=args.lr_policy,
             layer_size=args.layer_size,
             hidden_size=args.hidden_size,
             max_grad_norm=args.max_grad_norm,
             max_replay_size=args.max_replay_size,
             batch_size=args.batch_size,
             target_update_coee=args.target_update_coefficient,
             target_interval=args.target_interval,
             epsilon_decay=args.epsilon_decay,
             epsilon_max=args.epsilon_max,
             epsilon_min=args.epsilon_min, 
             use_double=args.use_double,
             use_per=args.use_per,
             prop_alpha=args.prop_alpha,
             weight_beta=args.weight_beta,
             gain_beta_steps=args.gain_beta_steps,
             use_duel=args.use_duel)

common_path = "./logs/" + agent.__class__.__name__ + "/" + args.env_name + "/seed_" + str(args.seed) + "_" + \
    datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + "/"
save_model_path = common_path + "model/"
log_path = common_path

if not os.path.exists(log_path):    
	os.makedirs(log_path)
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)
writer = SummaryWriter(log_path)

agent.train(env, env_test, writer, args.max_train_steps, args.random_steps,
            args.save_interval, args.log_interval, save_model_path)