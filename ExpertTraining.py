import gym
import nel
import pdb
from getch import getch, pause 
import numpy as np
import pickle

env = gym.make('NEL-render-v0')

# Tong = scent[0]
# Jelly Bean = scent[2]
# Diamond = scent[1]

key = {'w':0,'a':1,'d':2}

try:
  with open('ExpertTrajectories.data', 'rb') as filehandle:
    trajectories = pickle.load(filehandle)
except(Exception):
  print('No file found, create new trajectories')
  trajectories = []

trajectory = []
env.reset()
avg_reward = 0
for t in range(10000):
  env.render()
  action = getch()
  if(action=='q'):
    trajectories.append(trajectory)
    with open('ExpertTrajectories.data', 'wb') as filehandle:
      pickle.dump(trajectories, filehandle)
    print(len(trajectories))
    break
  observation, reward, _, info = env.step(key[action])
  trajectory.append((observation, key[action], reward))
  avg_reward += reward
  print(avg_reward/(t+1))
