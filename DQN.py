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

random.seed(10)
torch.manual_seed(10)
np.random.seed(10)

class QNetwork(nn.Module):
	def __init__(self, environment_name, gamma, lr, hidden_size1):
		super(QNetwork, self).__init__()

		env1 = gym.make(environment_name)
		self.num_actions = env1.action_space.n

		self.conv1 = torch.nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1)
		self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

		self.ff1 = nn.Linear(3872,1000)#hidden_size1)
		torch.nn.init.xavier_uniform_(self.ff1.weight)
		self.relu = nn.ReLU()

		self.ff2 = nn.Linear(1000,100)#(hidden_size1,2*hidden_size1)
		torch.nn.init.xavier_uniform_(self.ff2.weight)

		self.ff3 = nn.Linear(106,32)#(2*hidden_size1,hidden_size1)
		torch.nn.init.xavier_uniform_(self.ff3.weight)

		self.final = nn.Linear(32,self.num_actions)#(hidden_size1, self.num_actions)
		torch.nn.init.xavier_uniform_(self.final.weight)

	def forward(self, states):
		(prev_visions, prev_scents, prev_moveds), (visions, scents, moveds) = states
		# print(prev_visions.shape)
		# print(prev_scents.shape)

		inp = torch.cat((torch.from_numpy(prev_visions).float().cuda(), torch.from_numpy(visions).float().cuda()), dim=3)
		# print(inp.shape)
		inp = inp.view(inp.shape[0], inp.shape[3], inp.shape[2], inp.shape[1])
		# print(inp.shape)
		conv1 = self.relu(self.conv1(inp))
		# print(conv1.shape)
		conv2 = self.relu(self.conv2(conv1))
		conv2 = conv2.view(conv2.shape[0], -1)
		# print(conv2.shape)
		ff1 = self.relu(self.ff1(conv2))
		ff2 = self.relu(self.ff2(ff1))
		# print(ff2.shape)
		ff2_input = torch.cat((ff2, torch.from_numpy(np.asarray(prev_scents)).float().cuda(), torch.from_numpy(np.asarray(scents)).float().cuda()), 1)

		ff3 = self.relu(self.ff3(ff2_input))
		output = self.final(ff3)
		return output

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

		self.network = QNetwork(environment_name, self.gamma, lr, hidden_layer_size).cuda()
		self.store_network = copy.deepcopy(self.network).cuda()
		self.epsilon = 0.5
		self.memory = Replay_Memory()
		self.episodes = 10000
		self.time_steps = 300
		self.loss = nn.MSELoss()
		self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
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
		while(len(self.memory.memory)<burn_in):
			self.state = self.env.reset()
			self.state = np.reshape(self.state, (1, self.state.shape[0]))
			for j in range(self.time_steps):
				action = np.random.randint(self.env.action_space.n) #self.env.action_space.sample()
				next_state, reward, is_final, info = self.env.step(action)
				if is_final==True:
					is_final = 0
				else:
					is_final = 1
				self.memory.append([self.state, action, reward, next_state, is_final])
				self.state = next_state
				if is_final:
					break
				if(len(self.memory.memory)==burn_in):
					break

	def create_predictions_labels(self, batch):
		prev_states = np.vstack([batch[i][0] for i in range(len(batch))])
		prev_visions = np.vstack([np.expand_dims(state[0]['vision'], axis=0) for state in prev_states])
		prev_scents = np.vstack([np.expand_dims(state[0]['scent'], axis=0) for state in prev_states])
		prev_moveds = np.vstack([np.expand_dims(state[0]['moved'], axis=0) for state in prev_states])
		states = np.vstack([batch[i][1] for i in range(len(batch))])
		# print(len(states))
		# print('VISION SHAPE')
		# print(len(prev_states))
		# print(prev_visions.shape)
		# print(states[0][0]['vision'].shape)
		visions = np.vstack([np.expand_dims(state[0]['vision'], axis=0) for state in states])
		scents = np.vstack([np.expand_dims(state[0]['scent'], axis=0) for state in states])
		moveds = np.vstack([np.expand_dims(state[0]['scent'], axis=0) for state in states])
		actions = torch.from_numpy(np.vstack([batch[i][2] for i in range(len(batch))]))
		rewards = np.vstack([batch[i][3] for i in range(len(batch))])
		n_states = np.vstack([batch[i][4] for i in range(len(batch))])
		n_visions = np.vstack([np.expand_dims(state[0]['vision'], axis=0) for state in n_states])
		n_scents = np.vstack([np.expand_dims(state[0]['scent'], axis=0) for state in n_states])
		n_moveds = np.vstack([np.expand_dims(state[0]['scent'], axis=0) for state in n_states])
		is_finals = torch.tensor([batch[i][5] for i in range(len(batch))]).float().unsqueeze(1)
		# print('LAST VISIONS')
		# print(prev_visions.shape)
		# print(visions.shape)
		predictions = self.network.forward([(prev_visions, prev_scents, prev_moveds), (visions, scents, moveds)]).gather(1, actions.long())
		if self.args.network_type == 'double' or 'two_step':
		  values, actions_max = self.network.forward([(visions, scents, moveds), (n_visions, n_scents, n_moveds)]).detach().max(1)
		  actions_max = actions_max.unsqueeze(1)
		  Q = self.store_network.forward([(visions, scents, moveds), (n_visions, n_scents, n_moveds)]).detach().gather(1, actions_max)
		elif self.args.network_type == 'simple' or 'two_step':
		  Q = self.store_network.forward(n_states).detach().max(1)[0].unsqueeze(1).float()
		rewards = torch.from_numpy(rewards).float()
		targets = torch.autograd.Variable(rewards + self.gamma * Q * is_finals, requires_grad=True)
		return predictions, targets


	def train(self, batch_size):
		train_steps = 0
		avg_reward = []
		avg_reward_two_step = []
		losses = []
		scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = 0.99)
		for i in range(self.episodes):
			self.state = self.env.reset()
			prev_vision = np.expand_dims(self.state['vision'], axis=0)
			prev_scent = np.expand_dims(self.state['scent'], axis=0)
			prev_moved = np.expand_dims(self.state['moved'], axis=0)
			self.state_t_1 = self.state
			vision = np.expand_dims(self.state['vision'], axis = 0)
			scent = np.expand_dims(self.state['scent'], axis = 0)
			moved = np.expand_dims(self.state['moved'], axis=0)
			while True:
				values = self.network.forward([(prev_vision, prev_scent, prev_moved), (vision, scent, moved)])
				values = values.cpu().data.numpy()
				action = self.epsilon_greedy_policy(values)
				next_state, reward, is_final, info = self.env.step(action)
				if is_final==True:
					is_final = 0
				else:
					is_final = 1
				self.memory.append((self.state_t_1, self.state, action, reward, next_state, is_final))
				self.state_t_1 = self.state
				self.state = next_state

				if is_final==0:
					print(i)
					print(loss)
					break
			if len(self.memory.memory)>batch_size:
				train_steps+=1
				batch = self.memory.sample_batch(batch_size)
				logits, ys = self.create_predictions_labels(batch)
				loss = self.loss(ys, logits)
				losses.append(loss.data.numpy())
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
			self.store_network = copy.deepcopy(self.network)

			if i % 100 == 0:
				scheduler.step()
				# av_rew_two_step = self.two_step_lookahead(0.05)
				av_rew = self.test(0.05)
				# avg_reward_two_step.append(av_rew_two_step)
				avg_reward.append(av_rew)
				print(self.env_name+" Episode #: "+str(i)+", Average Reward: "+str(av_rew)+", Average Loss: "+str(np.mean(losses)))
		np.save("./Save_files/"+self.env_name+" Average Reward - "+self.args.network_type, avg_reward)
		np.save("./Save_files/"+self.env_name+" Loss - "+self.args.network_type, losses)
		# np.save("./Save_files/"+self.env_name+" Average Reward Two Step - "+self.args.network_type, avg_reward_two_step)
		self.env.close()

	def two_step_lookahead(self, epsilon):
		total_rew = 0
		for episode in range(20):
			s11 = self.env.reset()

			while True:
				a1_ = self.network.forward(s11)
				a11 = a1_.data.numpy()[0]
				a12 = a1_.data.numpy()[1]
				temp = copy.deepcopy(self.env)

				s21, r11, is_final21, _ = self.env.step(0)
				a21 = self.network.forward(s21)
				a21 = np.max(a21.data.numpy())
				self.env = copy.deepcopy(temp)

				s22, r12, is_final22, _ = self.env.step(1)
				a22 = self.network.forward(s22)
				a22 = np.max(a22.data.numpy())
				self.env = copy.deepcopy(temp)

				if random.random() < epsilon:
					final_action = self.env.action_space.sample()
				else:
					final_action = np.argmax([a11+self.gamma*a21, a12+self.gamma*a22])

				next_state, reward, is_final, info = self.env.step(final_action)
				total_rew += reward
				s11 = next_state
				if is_final:
					break
		return total_rew/20

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
	parser.add_argument('--env',dest='env',type=str, default='NEL-render-v0')
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
