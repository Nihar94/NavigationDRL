import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import gym

class FeatureNet(nn.module):
	def __init__(self, args):
		super(Main, self).__init__()
		self.env = args.env
		self.batch_size = args.batch_size
		# Fusion multiplier for Visual Features
		self.alpha = Variable(torch.randn(1), requires_grad=True)
		# Fusion multiplier for Scent
		self.beta = Variable(torch.randn(1), requires_grad=True)
		self.features = nn.Sequential(
				nn.Conv2d(1, 10, 3),
				nn.ReLU(),
				nn.Conv2d(10, 25, 3),
				nn.ReLU(),
				nn.Conv2d(25, 20, 1),
				nn.ReLU(),
			)

	def forward(self, states):
		(prev_vision, prev_scent, prev_moved), (vision, scent, moved) = states
		vision_features = self.alpha * self.features(vision)
		scent = self.beta * scent
		combined_features = torch.cat((vision_features, scent), 0)
		return combined_features

class 