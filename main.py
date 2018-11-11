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

torch.manual_seed(10)
np.random.seed(10)

def trainTongNet(args):
	env = gym.make(args.env)
	with open('ExpertTrajectories.data', 'rb') as filehandle:
		expertTrajectories = pickle.load(filehandle)
	featureNet = FeatureNet(args)
	tongNet = TongNet(args)
	prev_state = env.reset()
	current_state = copy.deepcopy(prev_state)
	current_expert_state = copy.deepcopy(prev_state)
	prev_expert_state = copy.deepcopy(prev_state)
	for i in range(len(expertTrajectories)):
		for t in range(len(expertTrajectories[i])):
			expert_features = featureNet((prev_expert_state, current_expert_state))
			current_expert_state = expertTrajectories[i][t][0]
			novice_features = featureNet((prev_state, current_state))		
			pdb.set_trace()
			expert_action = expertTrajectories[i][0]
			novice_actions = tongNet(features)
			loss = -np.log(novice_actions[expert_action])
			loss.backward()

def parse_arguments():
	parser = argparse.ArgumentParser(description='Maze Navigator Argument Parser')
	parser.add_argument('--env',dest='env',type=str, default='NEL-v0')
	parser.add_argument('--render',dest='render',type=bool,default=False)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--model',dest='model_file',type=str, default='')
	parser.add_argument('--batch_size',dest='batch_size',type=int, default = 32)
	parser.add_argument('--search_size',dest='search_size',type=int, default = 100)
	parser.add_argument('--network_type',dest='network_type',type=str, default="double")
	return parser.parse_args()

def main(args):
	args = parse_arguments()
	environment_name = args.env
	print(type(environment_name))
	print(args.model_file)
	trainTongNet(args)

if __name__ == '__main__':
	main(sys.argv)
