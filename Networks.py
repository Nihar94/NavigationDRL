import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import gym
import pdb

torch.manual_seed(10)
np.random.seed(10)

class FeatureNet(nn.Module):
	def __init__(self, args):
		super(FeatureNet, self).__init__()
		# Fusion multiplier for Visual Features
		self.alpha = Variable(torch.randn(1), requires_grad=True)*0+1
		
		# Fusion multiplier for Scent
		self.beta = Variable(torch.randn(1), requires_grad=True)*0+1
		
		# Alexnet features with frozen weights
		self.alexnet = models.alexnet(pretrained=True).features
		for param in self.alexnet.parameters():
			param.requires_grad = False

		# Learnable classifier1
		self.vision_features = nn.Sequential(
			nn.Linear(256 * 6 * 6, 300),
			nn.ReLU(inplace=True)
			)

		# Learnable classifier2
		self.combined_features = nn.Sequential(
			nn.Linear(608, 200)
			)

	def forward(self, states):

		prev_scent = torch.from_numpy(states[0]['scent'])
		curr_scent = torch.from_numpy(states[1]['scent'])
		
		prev_vision = torch.from_numpy(states[0]['vision']).permute(2,0,1).unsqueeze(0)
		curr_vision = torch.from_numpy(states[1]['vision']).permute(2,0,1).unsqueeze(0)
		
		prev_moved = int(states[0]['moved'] == True)
		curr_moved = int(states[1]['moved'] == True)
		
		Uprev_vision = nn.functional.interpolate(prev_vision, size=(224,224)) #self.upsample(prev_vision)
		Ucurr_vision = nn.functional.interpolate(curr_vision, size=(224,224)) #self.upsample(curr_vision)
			
		vision_features = torch.cat((Uprev_vision, Ucurr_vision), 0)
		
		vision_features = self.alexnet(vision_features)
		vision_features = vision_features.view(vision_features.size(0), 256*6*6)
		vision_features = self.alpha * self.vision_features(vision_features).view(-1)
		
		scent = torch.cat((prev_scent, curr_scent), 0)
		scent = self.beta * scent

		movement = torch.tensor([prev_moved, curr_moved]).float()
		movement.requires_grad=True
		
		combined_features = torch.cat((vision_features, scent, movement), 0)
		combined_features = self.combined_features(combined_features)

		return combined_features

class TongNet(nn.Module):
	def __init__(self, args):
		super(TongNet,self).__init__()
		self.network = nn.Sequential(
			nn.Linear(200, 200),
			nn.ReLU(inplace=True),
			nn.Linear(200, 100),
			nn.ReLU(inplace=True),
			nn.Linear(100, 20),
			nn.ReLU(inplace=True),
			nn.Linear(20, 3)
			)
		self.sm = nn.Softmax(dim=0)

	def forward(self, features):
		out = self.network(features)
		out = self.sm(out)
		return out

class RewardNet(nn.Module):
	def __init__(self, args):
		super(RewardNet, self).__init__()
		self.network = nn.Sequential(
			nn.Linear(200, 200),
			nn.ReLU(inplace=True),
			nn.Linear(200, 100),
			nn.ReLU(inplace=True),
			nn.Linear(100, 20),
			nn.ReLU(inplace=True),
			nn.Linear(20, 3)
			)
		self.sm = nn.Softmax(dim=0)
		
	def forward(self, features):
		out = self.network(features)
		out = self.sm(out)
		return out