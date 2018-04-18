'''
A **simplistic** NN implementation of Q learning.
the neural networks are employed as a function approximator to predict the value function (Q).
NN has output nodes equal to the number of possible actions, and
their output signifies the value function of the corresponding action.
'''
import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

class NeuralNetwork:
    def __init__(self, D):
        self.W = tf.Variable(tf.random_normal(shape=(D, 1)), name='w')
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')

        # make prediction and cost
        Y_hat = tf.reshape(tf.matmul(self.X, self.W), [-1])
        err = self.Y - Y_hat
        cost = tf.reduce_sum(tf.pow(err,2))

        # ops we want to call later
        eta = 0.1
        self.train_op = tf.train.GradientDescentOptimizer(eta).minimize(cost)
        self.predict_op = Y_hat

        # start the session and initialize params
        init_op = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init_op)

    def train(self, X, Y):
        self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})

    def predict(self, X):
        return self.session.run(self.predict_op, feed_dict={self.X: X})

class FeatureTransformer:
  '''
  to quantize the continuous-valued observation states
  step:
  * first generates 20,000 random samples of observation space examples.
    The FeatureTransformer class is instantiated with the random observation space examples,
    which are used to train the RBFSampler using the fit_transform function method.
  * The randomly generated observation space examples are standardized using the scikit StandardScaler class.
  * Then scikit's RBFSampler is employed with different variances to
    cover different parts of the observation space.
  '''
  def __init__(self, env):
    #obs_examples = np.array([env.observation_space.sample() for x in range(20000)])

    obs_examples = np.random.random((20000, 4))
    # print(obs_examples.shape)
    scaler = StandardScaler()
    scaler.fit(obs_examples)

    # Used to converte a state to a featurizes represenation.
    # We use RBF kernels with different variances to cover different parts of the space
    featurizer = FeatureUnion([
            ("cart_position", RBFSampler(gamma=0.02, n_components=500)),
            ("cart_velocity", RBFSampler(gamma=1.0, n_components=500)),
            ("pole_angle", RBFSampler(gamma=0.5, n_components=500)),
            ("pole_velocity", RBFSampler(gamma=0.1, n_components=500))
            ])
    feature_examples = featurizer.fit_transform(scaler.transform(obs_examples))
    # print(feature_examples.shape)

    self.dimensions = feature_examples.shape[1]
    self.scaler = scaler
    self.featurizer = featurizer

  def transform(self, observations):
    scaled = self.scaler.transform(observations)
    return self.featurizer.transform(scaled)

class Agent:
  def __init__(self, env, feature_transformer):
    self.env = env
    self.feature_transformer = feature_transformer

    ## we need n different neural network objects to get the predicted state action value
    self.agent = []
    for i in range(env.action_space.n):
      nn = NeuralNetwork(feature_transformer.dimensions)
      self.agent.append(nn)

  def predict(self, s):
    X = self.feature_transformer.transform([s])
    return np.array([m.predict(X)[0] for m in self.agent])

  def update(self, s, a, G):
    X = self.feature_transformer.transform([s])
    self.agent[a].train(X, [G])

  def sample_action(self, s, epsilon):
    if np.random.random() < epsilon:
      return self.env.action_space.sample()
    else:
      return np.argmax(self.predict(s))

def run_one_episode(env, agent, epsilon, gamma):
  obs = env.reset()
  done = False
  totalreward = 0
  step_idx = 0
  max_n_steps = 2000
  while (not done) and (step_idx < max_n_steps):
    action = agent.sample_action(obs, epsilon)
    prev_obs = obs

    obs, reward, done, info = env.step(action)
    env.render()

    ## penalty
    if done:
      reward = -400

    ## update the model of action-value network via Q learning
    ## G = r + \gamma max_{a'} Q(s',a')
    Q_prime = agent.predict(obs)
    assert(len(Q_prime.shape) == 1) # Q_prime.shape= (n_actions,)

    G = reward + gamma*np.max(Q_prime) # as the Q target
    agent.update(prev_obs, action, G)

    ## why not include reward==-400?
    if reward == 1:
      totalreward += reward

    step_idx += 1

  return totalreward

if __name__ == '__main__':
    '''
    https://gym.openai.com/envs/CartPole-v0/
    A reward of +1 is provided for every timestep that the pole remains upright.
    The episode ends
        * when the pole is more than 15 degrees from vertical, or
        * the cart moves more than 2.4 units from the center.
    CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials.
    '''
    env_name = 'CartPole-v0'
    env = gym.make(env_name)

    ft = FeatureTransformer(env)
    agent = Agent(env, ft)
    gamma = 0.97

    ## training ################################################################
    n_episodes = 1000
    totalrewards = np.empty(n_episodes)
    running_avg = np.empty(n_episodes)
    for n in range(n_episodes):
        ## epsilon here is used in Epsilon Greedy Policy for exploration during the training phase
        ## epsilon is annealed during the training process so that,
        ## initially, the agent takes lots of random actions (exploration) but
        ## as training progresses, the actions with maximum Q value are taken (exploitation).
        epsilon = 1.0 / np.sqrt(n + 1)

        totalreward = run_one_episode(env, agent, epsilon, gamma)
        totalrewards[n] = totalreward
        running_avg[n] = totalrewards[max(0, n - 100):(n + 1)].mean()

        if (n % 100) == 0:
            print("episode: {0}, total reward: {1} epsilon: {2} avg reward (last 100): {3}". \
                  format(n, totalreward, epsilon, running_avg[n]))

    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", totalrewards.sum())

    plt.plot(totalrewards)
    plt.xlabel('episodes')
    plt.ylabel('Total Rewards')
    plt.show()

    plt.plot(running_avg)
    plt.xlabel('episodes')
    plt.ylabel('Running Average')
    plt.show()

    env.close()
