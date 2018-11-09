#!/usr/bin/env python
import gym, sys, copy, argparse
import torch
import torch.nn as nn
import numpy as np
import pdb
import copy


def parse_arguments():
	parser = argparse.ArgumentParser(description='Maze Navigator Argument Parser')
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
