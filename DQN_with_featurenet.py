#!/usr/bin/env python
import gym, sys, copy, argparse
import torch
import torch.nn as nn
import nel
import gym
import random
import os
import numpy as np
import pdb
import copy
from Networks import FeatureNet, weights_init
import torch.nn.functional as F


random.seed(10)
torch.manual_seed(10)
np.random.seed(10)

class QNetwork(nn.Module):
	def __init__(self, args):
		super(QNetwork,self).__init__()
		self.network = nn.Sequential(
			nn.Linear(51, 25),
			nn.ReLU(inplace=True),
			nn.Linear(25, 15),
			nn.ReLU(inplace=True),
			nn.Linear(15, 3)
			)
		self.network.apply(weights_init)
		self.sm = nn.Softmax(dim=0)

	def forward(self, features):
		out = self.network(features)
		out = self.sm(out)
		return out

	def save_model_weights(self, suffix):
		# Helper function to save your model / weights.
		os.makedirs(suffix, exist_ok=True)
		save_path = os.path.join(suffix, 'model.pth')
		print(save_path)
		torch.save(self, save_path)
		print("Model is saved in '{}'...".format(save_path))

	def load_model(self, model_file):
		# Helper function to load an existing model.
		model = torch.load(model_file)
		print("Model is loaded from '{}'...".format(model_file))
		return model

	def load_model_weights(self,weight_file):
		# Helper funciton to load model weights.
		pass


class Replay_Memory():
	def __init__(self, memory_size=50000, burn_in=10000):
		self.memory=[]
		self.memory_size = memory_size
		self.burn_in = burn_in

	def sample_batch(self, batch_size=32):

		sampled_batch = random.sample(self.memory, batch_size)
		return sampled_batch

	def append(self, transition):
		if len(self.memory) == self.memory_size:
			self.memory.pop(0)
			self.memory.append(transition)
		else:
			self.memory.append(transition)

	def len(self):
		return len(self.memory)


class DQN_Agent():
	def __init__(self, environment_name, args, render=False):
		self.env = gym.make(environment_name)
		self.env.seed(10)
		if render:
			self.env.render()


		self.gamma = 0.99
		self.epsilon_decay = 0.995
		lr = 0.001
		hidden_layer_size = 128

		self.network = QNetwork(args).cuda()
		self.store_network = copy.deepcopy(self.network).cuda()
		self.feature_net = FeatureNet(args)
		params = list(self.network.parameters()) + list(self.feature_net.parameters())
		self.epsilon = 0.5
		self.memory = Replay_Memory()
		self.burn_in_memory()
		self.episodes = 100000
		self.time_steps = 300
		self.loss = nn.SmoothL1Loss()
		self.optimizer = torch.optim.Adam(params, lr=lr)
		self.args = args
		self.env_name = environment_name

	def epsilon_greedy_policy(self, q_values):
		if np.random.random() < self.epsilon:
			action = np.random.randint(self.env.action_space.n)#.sample()
		else:
			action = np.argmax(q_values)
		self.epsilon *= self.epsilon_decay
		return action

	def greedy_policy(self, q_values):
		return np.argmax(q_values)

	def burn_in_memory(self):
		burn_in = self.memory.burn_in
		state = self.env.reset()
		prev_state = copy.deepcopy(state)
		while(len(self.memory.memory)<burn_in):
			action = np.random.randint(self.env.action_space.n) #self.env.action_space.sample()
			new_state, rew, _, info = self.env.step(action)
			prev_state = state
			state = new_state
			self.memory.append((prev_state, state, action, rew))
			if(len(self.memory.memory)==burn_in):
				break

	def create_predictions_labels(self, batch):
		prev_states = [batch[i][0] for i in range(len(batch))]
		states = [batch[i][1] for i in range(len(batch))]
		stats = list(zip(prev_states, states))
		feats = [self.feature_net(stats[i]).unsqueeze(0) for i in range(len(stats))]
		feats = torch.cat(feats, 0)
		print('FEATS')
		print(feats.shape)
		# print(len(states))
		# print('VISION SHAPE')
		# print(len(prev_states))
		# print(prev_visions.shape)
		# print(states[0][0]['vision'].shape)
		# visions = np.vstack([np.expand_dims(state[0]['vision'], axis=0) for state in states])
		# scents = np.vstack([np.expand_dims(state[0]['scent'], axis=0) for state in states])
		# moveds = np.vstack([np.expand_dims(state[0]['scent'], axis=0) for state in states])
		actions = np.vstack([batch[i][2] for i in range(len(batch))])
		print(actions.shape)
		rewards = np.vstack([batch[i][3] for i in range(len(batch))])
		# print('LAST VISIONS')
		# print(prev_visions.shape)
		# print(visions.shape)
		print(torch.cat((feats, torch.from_numpy(actions).float()), 1).shape)
		print(self.network.forward(torch.cat((feats, torch.from_numpy(actions).float()), 1).cuda()).shape)
		predictions = self.network.forward(torch.cat((feats, torch.from_numpy(actions).float()), 1).cuda()).gather(1, torch.from_numpy(actions).long().cuda())
		#if self.args.network_type == 'double' or 'two_step':
		values, actions_max = self.network.forward(torch.cat((feats, torch.from_numpy(actions).float()), 1).cuda()).detach().max(1)
		actions_max = actions_max.unsqueeze(1)
		Q = self.store_network.forward(torch.cat((feats, torch.from_numpy(actions).float()), 1).cuda()).detach().gather(1, actions_max)
		# elif self.args.network_type == 'simple' or 'two_step':
		#   Q = self.store_network.forward(n_states).detach().max(1)[0].unsqueeze(1).float()
		rewards = torch.from_numpy(rewards).float()
		targets = torch.autograd.Variable(rewards.cuda() + self.gamma * Q.cuda(), requires_grad=False)
		return predictions, targets


	def train(self, batch_size):
		train_steps = 0
		avg_reward = []
		rewards = []
		avg_reward_two_step = []
		losses = []
		scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = 0.99)
		self.state = self.env.reset()
		self.state_t_1 = self.state
		action = np.random.randint(self.env.action_space.n)
		print(type(action))
		for i in range(self.episodes):
			features = self.feature_net((self.state_t_1, self.state))
			# print(features.shape)
			# print(torch.tensor(action).unsqueeze(0).shape)
			# print(type(action))
			input = torch.cat((features, torch.tensor(int(action)).unsqueeze(0).float()), 0)
			# print(input.shape)
			# print(type(input))
			values = self.network.forward(input.cuda())
			values = values.cpu().data.numpy()
			action = self.epsilon_greedy_policy(values)
			# print(type(action))
			next_state, reward, is_final, info = self.env.step(action)
			rewards.append(reward)
			if is_final==True:
				is_final = 0
			else:
				is_final = 1
			self.memory.append((self.state_t_1, self.state, action, reward))
			self.state_t_1 = self.state
			self.state = next_state
			print('average reward')
			print(np.mean(rewards[-100:]))
			if i%100 == 0:
				train_steps+=1
				batch = self.memory.sample_batch(batch_size)
				logits, ys = self.create_predictions_labels(batch)
				# print(logits.shape)
				# print(ys.shape)
				loss = self.loss(logits, ys)
				losses.append(loss.cpu().data.numpy())
				print(loss)
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				self.store_network = copy.deepcopy(self.network)
				scheduler.step()
				# av_rew_two_step = self.two_step_lookahead(0.05)
				# av_rew = self.test(0.05)
				# avg_reward_two_step.append(av_rew_two_step)
				# avg_reward.append(av_rew)
				# print(self.env_name+" Episode #: "+str(i)+", Average Reward: "+str(av_rew)+", Average Loss: "+str(np.mean(losses)))
		torch.save(self.feature_net.state_dict(), args.model_path+'featureNet')
		torch.save(self.network.state_dict(), args.model_path+'network')
		# np.save("./Save_files/"+self.env_name+" Average Reward Two Step - "+self.args.network_type, avg_reward_two_step)
		self.env.close()

	def test(self, epsilon, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory.
		total_rew = 0
		#FOR TESTING (WITH GREEDY POLICY)
		if epsilon == 0:
			for episode in range(100):
				state = self.env.reset()
				while True:
					self.env.render()
					actions = self.network.forward(state)
					action = self.greedy_policy(actions)
					next_state, reward, is_final, info = self.env.step(action)
					total_rew += reward
					state = next_state
					if is_final:
						break
		#FOR TESTING (WITH EPSILON GREEDY POLICY WITH A GIVEN EPSILON)
		else:
			for episode in range(100):
				state = self.env.reset()
				while True:
					#self.env.render()
					actions = self.network.forward(state)

					#if random.random() < epsilon:
					#	action = self.env.action_space.sample()
					#else:
					action = np.argmax(actions.data.numpy())
					next_state, reward, is_final, info = self.env.step(action)
					total_rew += reward
					state = next_state
					if is_final:
						break
		return total_rew/100

def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str, default='NEL-v0')
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--model',dest='model_file',type=str)
	parser.add_argument('--batch_size',dest='batch_size',type=int, default = 32)
	parser.add_argument('--network_type',dest='network_type',type=str, default="double")
	return parser.parse_args()

def main(args):

	args = parse_arguments()
	environment_name = args.env
	print(type(environment_name))
	print(args.model_file)

	agent = DQN_Agent(environment_name, args)
	agent.train(args.batch_size)

if __name__ == '__main__':
	main(sys.argv)
