'''
The agents in this recipe are **not** learning any policy;
they make their decision based on their initial set of weights (fixed policy).
'''
import os
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  ## To deactivate SSE Warnings

class RlAgent(object):
    def __init__(self, n_input_units, n_output_units, init=False, W=None, b=None ):
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._X = tf.placeholder(tf.float32,shape=(1,n_input_units))

            if init==False:
                self._W = tf.Variable(tf.random_normal([n_input_units,n_output_units]), trainable=False)
                self._bias = tf.Variable(tf.random_normal([1,n_output_units]),trainable=False)
            else:
                self._W = W
                self._bias = b

            ## tf.multinomial, to make the decision of the possible action to take.
            ## The function returns the action based on the sigmoidal values of the nine output neurons of our network.
            output = tf.nn.sigmoid(tf.matmul(self._X,self._W)+ self._bias)
            self._result = tf.multinomial(output,1)

            ##
            self._sess = tf.Session()

            ##
            init = tf.global_variables_initializer()
            self._sess.run(init)

    def predict(self, X):
        action = self._sess.run(self._result, feed_dict= {self._X: X})
        return action

    def get_weights(self):
        W, b = self._sess.run([self._W, self._bias])
        return W, b

def preprocess_image(img):
  '''
  preprocessing we do is
    convert it to grayscale,
    increase the contrast, and
    reshape it into a row vector
  '''
  img = img.mean(axis =2) # to grayscale
  img[img==150] = 0
  img = (img - 128)/128 - 1 # Normalize image from -1 to 1
  m,n = img.shape
  return img.reshape(1,m*n)

def play_one_episode(env, agent, render=False):
  obs = env.reset()
  img_pre = preprocess_image(obs)
  done = False
  t = 0

  while not done and t < 10000:
    if render: env.render()
    t += 1

    action = agent.predict(img_pre)
    #print(t,action)
    obs, reward, done, info = env.step(action)
    img_pre = preprocess_image(obs)
    if done:
      break

  return t

def play_multiple_episodes(env, n_episodes, init, W=None, b=None):
  obs = env.reset()
  img_pre = preprocess_image(obs) #img_pre.shape= (1, 33600), a row vector

  if init == False:
    agent = RlAgent(img_pre.shape[1], env.action_space.n)
  else:
    agent = RlAgent(img_pre.shape[1], env.action_space.n, init, W, b)

  episode_lengths = np.empty(n_episodes)
  for i in range(n_episodes):
    episode_lengths[i] = play_one_episode(env, agent)

  avg_length = episode_lengths.mean()
  print("avg length:", avg_length)
  if init == False:
      W, b = agent.get_weights()

  return avg_length, W, b

def random_search(env, n_agents, n_episodes_per_agent):
  '''
  invokes play_multiple_episodes();
  each time play_multiple_episodes() is called, a new agent is instantiated with
  a new set of random weights and biases.
  One of these randomly created NN agents will outperform others, and
  this will be the agent that we finally select:
  '''
  episode_lengths = []
  longest_length = 0
  for t in range(n_agents):
    print("Agent {} reporting".format(t))
    avg_length, wts, bias = play_multiple_episodes(env, n_episodes_per_agent, init=False)
    episode_lengths.append(avg_length)

    if avg_length > longest_length:
      longest_length = avg_length
      best_wt = wts
      best_bias = bias

  # plt.plot(episode_lengths)
  # plt.show()
  return best_wt, best_bias

if __name__ == '__main__':
    #env_name = 'Breakout-v0'
    env_name = 'MsPacman-v0'
    env = gym.make(env_name)

    ## train
    print('train: begin')
    n_agents = 1
    n_episodes_per_agent = 1
    W, b = random_search(env, n_agents, n_episodes_per_agent)
    print('train: end')

    ## test
    print('test: begin')
    print("Final Run with best Agent")
    play_multiple_episodes(env, n_episodes=1, init=True, W=W, b=b)
    print('test: end')

    ## closure
    env.close()
