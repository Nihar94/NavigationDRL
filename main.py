#!/usr/bin/env python
import gym, sys, copy, argparse
import torch
import torch.nn as nn
import nel
import pickle
import numpy as np
import pdb
import copy
from Networks import *
from ExpertTraining import *

torch.manual_seed(10)
np.random.seed(10)

'''
def trainRewardNet(args):
	env = gym.make(args.env)
	featureNet = FeatureNet(args)
	actor = Actor()
	critic = Critic()
	prev_state = env.reset()
	curr_state = copy.deepcopy(prev_state)
	curr_expert_state = copy.deepcopy(prev_state)
	prev_expert_state = copy.deepcopy(prev_state)
	actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
'''

def play(args):
	env = gym.make('NEL-render-v0')
	featureNet = FeatureNet(args)
	featureNet.load_state_dict(torch.load(args.model_path+'featureNet'))
	tongNet = TongNet(args)
	tongNet.load_state_dict(torch.load(args.model_path+'tongNet'))
	prev_state = env.reset()
	curr_state = copy.deepcopy(prev_state)
	avg_reward = 0
	for t in range(1000):
		env.render()
		features = featureNet((prev_state, curr_state))
		prev_state = curr_state
		actions = tongNet(features)
		print(actions)
		action = torch.argmax(actions)
		curr_state, reward, _, info = env.step(action)
		avg_reward += reward
		print(avg_reward/(t+1))

def trainTongNet(args):
	env = gym.make(args.env)
	with open('ExpertTrajectories.data', 'rb') as filehandle:
		expertTrajectories = pickle.load(filehandle)
	featureNet = FeatureNet(args)
	tongNet = TongNet(args)
	try:
		featureNet.load_state_dict(torch.load(args.model_path+'featureNet'))
		tongNet.load_state_dict(torch.load(args.model_path+'tongNet'))
	except(Exception):
		print('No model save files')
	prev_state = env.reset()
	curr_state = copy.deepcopy(prev_state)
	curr_expert_state = copy.deepcopy(prev_state)
	prev_expert_state = copy.deepcopy(prev_state)
	params = list(featureNet.parameters()) + list(tongNet.parameters())
	optimizer = torch.optim.Adam(params, lr=args.lr)
	for e in range(args.epochs):
		for i in range(len(expertTrajectories)):
			for t in range(len(expertTrajectories[i])):
				expert_features = featureNet((prev_expert_state, curr_expert_state))
				prev_expert_state = curr_expert_state
				novice_features = featureNet((prev_state, curr_state))		
				prev_state = curr_state
				expert_action = expertTrajectories[i][t][1]
				novice_actions = tongNet(novice_features)
				loss = -torch.log(novice_actions[expert_action])*10
				#print(loss)
				# pdb.set_trace()
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				curr_expert_state = expertTrajectories[i][t][0]
				novice_action = torch.argmax(novice_actions)
				curr_state, _, _, _ = env.step(novice_action)
			print('Epoch #:'+str(e)+'Trajectory #:'+str(i))
	torch.save(featureNet.state_dict(), args.model_path+'featureNet')
	torch.save(tongNet.state_dict(), args.model_path+'tongNet')
			
def parse_arguments():
	parser = argparse.ArgumentParser(description='Maze Navigator Argument Parser')
	parser.add_argument('--env',dest='env',type=str, default='NEL-v0')
	parser.add_argument('--render',dest='render',type=bool,default=False)
	parser.add_argument('--train',dest='train',type=int)
	parser.add_argument('--generate',dest='generate',type=int)
	parser.add_argument('--test',dest='test',type=int)
	parser.add_argument('--lr',dest='lr',type=float,default=1e-3)
	parser.add_argument('--model_path',dest='model_path',type=str, default='/home/nihar/Desktop/DeepRL/Project/models/')
	parser.add_argument('--N',dest='N',type=int, default = 1000)
	parser.add_argument('--epochs',dest='epochs',type=int, default = 5)
	return parser.parse_args()

def main(args):
	args = parse_arguments()
	environment_name = args.env
	if(args.test==1):
		play(args)
	if(args.train==1):
		trainTongNet(args)
	if(args.generate==1):
		generateTrajectories()
	
if __name__ == '__main__':
	main(sys.argv)
