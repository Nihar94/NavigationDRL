import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import gym
import pdb

torch.manual_seed(10)
np.random.seed(10)

def weights_init(m):
    classname = m.__class__.__name__
    if(classname=='Linear'):
	    torch.nn.init.xavier_uniform_(m.weight)

class FeatureNet(nn.Module):
	def __init__(self, args):
		super(FeatureNet, self).__init__()
		# Fusion multiplier for Visual Features
		self.alpha = Variable(torch.randn(1), requires_grad=True)*0+1
		
		# Fusion multiplier for Scent
		self.beta = Variable(torch.randn(1), requires_grad=True)*0+1
		
		# Alexnet features with frozen weights
		#self.alexnet = models.alexnet(pretrained=True).features
		#for param in self.alexnet.parameters():
		#	param.requires_grad = False

		self.cnns = nn.Sequential(
			nn.Conv2d(3, 5, 3, padding=1),
			nn.LeakyReLU(inplace=True),
			nn.MaxPool2d(3,stride=2),
			nn.Conv2d(5,3,3,padding=1),
			nn.LeakyReLU(inplace=True),
			nn.MaxPool2d(3,stride=2)
			)
		# Learnable classifier1
		self.vision_features = nn.Sequential(
			nn.Linear(3*2*2, 20),
			nn.LeakyReLU(inplace=True),
			nn.Linear(20, 10),
			nn.LeakyReLU(inplace=True)
			)
		self.vision_features.apply(weights_init)
		# Learnable classifier2
		self.combined_features = nn.Sequential(
			nn.Linear(28, 28),
			nn.LeakyReLU(inplace=True),
			nn.Linear(28, 10),
			nn.LeakyReLU(inplace=True)
			)
		self.combined_features.apply(weights_init)
		
	def forward(self, states):

		prev_scent = torch.from_numpy(states[0]['scent'])
		curr_scent = torch.from_numpy(states[1]['scent'])
		
		prev_vision = torch.from_numpy(states[0]['vision']).permute(2,0,1).unsqueeze(0)
		curr_vision = torch.from_numpy(states[1]['vision']).permute(2,0,1).unsqueeze(0)
		# pdb.set_trace()
		prev_moved = int(states[0]['moved'] == True)*10
		curr_moved = int(states[1]['moved'] == True)*10
		
		# Uprev_vision = nn.functional.interpolate(prev_vision, size=(224,224)) #self.upsample(prev_vision)
		# Ucurr_vision = nn.functional.interpolate(curr_vision, size=(224,224)) #self.upsample(curr_vision)
		
		vision_features = torch.cat((prev_vision, curr_vision), 0)
		
		vision_features = self.cnns(vision_features)
		#vision_features = self.alexnet(vision_features)
		vision_features = vision_features.view(vision_features.size(0), 3*2*2)
		
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
			nn.Linear(50, 25),
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

# Value Model

class Critic(nn.Module):
	def __init__(self):
		super(Critic, self).__init__()
		self.network = nn.Sequential(
			nn.Linear(50, 25),
			nn.ReLU(inplace=True),
			nn.Linear(25, 15),
			nn.ReLU(inplace=True),
			nn.Linear(15, 3)
			)
		self.network.apply(weights_init)

	def forward(self, features):
		out = self.network(features)
		return out

# Policy Model

class Actor(nn.Module):
	def __init__(self):
		super(Actor, self).__init__()
		self.network = nn.Sequential(
			nn.Linear(50, 25),
			nn.ReLU(inplace=True),
			nn.Linear(25, 15),
			nn.ReLU(inplace=True),
			nn.Linear(15, 3)
			)
		self.network.apply(weights_init)

	def forward(self, features):
		out = self.network(features)
		return out