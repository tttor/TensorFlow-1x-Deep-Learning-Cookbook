import gym
from gym import envs
import numpy as np
import matplotlib.pyplot as plt

################################################################################
env_ids = [spec.id for spec in envs.registry.all()]
print("Total Number of environments are", len(env_ids))
for env_id in sorted(env_ids):
    print(env_id)

################################################################################
env_name = 'Breakout-v4'

env = gym.make(env_name)
obs = env.reset()
print("The observation space is ", obs.shape)

actions = env.action_space
print("The actions are", actions)
print("The number of possible actions are", env.action_space.n)

def random_agent(n):
    action = np.random.randint(0,n)
    return action

for step in range(1000):
    action = random_agent(env.action_space.n)
    obs, reward, done, info = env.step(action)
    env.render()

    if done:
        img = env.render(mode='rgb_array')
        plt.imshow(img)
        plt.axis('off')
        plt.savefig('game_last_stage.jpg')
        plt.show()
        print("The game is over in {} steps".format(step))
        break

env.close()
