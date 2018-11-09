import gym
import nel
import pdb

# Use 'NEL-render-v0' to include rendering support.
# Otherwise, use 'NEL-v0', which should be much faster.
env = gym.make('NEL-render-v0')

# The created environment can then be used as any other 
# OpenAI gym environment. For example:
#print(env.action_space)
print(env.step(0)[0])

# Tong = scent[0]
# Jelly Bean = scent[2]
# Diamond = scent[1]

'''
for t in range(10000):
  # Render the current environment.
  env.render()
  # Sample a random action.
  action = env.action_space.sample()
  print(action)
  # Run a simulation step using the sampled action.
  #action = int(input())
  observation, reward, _, _ = env.step(action)
  #a = env.step(action)
  #pdb.set_trace()
  #print(observation['scent'],reward)
'''