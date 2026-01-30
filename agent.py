import yaml
import torch
import itertools
import gymnasium
from dqn import DQN
import flappy_bird_gymnasium
from experience_replay import ReplayMemory

device = "mps" if torch.backends.mps.is_available() else "cpu"

class Agent:
  def __init__(self, hyperparameter_set="hyperparameters.yaml"):
    with open(hyperparameter_set, 'r') as file:
      all_hyperparameter_sets = yaml.safe_load(file)
      hyperparameters = all_hyperparameter_sets[hyperparameter_set]

    self.replay_memory_size = hyperparameters['replay_memory_size']
    self.mini_batch_size = hyperparameters['mini_batch_size']
    self.epsilon_init = hyperparameters['epsilon_init']
    self.epsilon_decay = hyperparameters['epsilon_decay']
    self.epsilon_min = hyperparameters['epsilon_min']

  def run(self, is_training=True, render=False):
    # env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
    env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)

    num_actions = env.action_space.n
    num_states = env.observation_space.shape[0]

    rewards_per_episode = []

    policy_dqn = DQN(input_dim=num_states, 
                     hidden_dim=128, 
                     output_dim=num_actions).to(device)
    
    if is_training:
       memory = ReplayMemory(self.replay_memory_size)

    for epoch in itertools.count():
      # Reset the environment
      state, _ = env.reset()
      terminated = False
      episode_reward = 0

      # Loop through 1 episode
      while not terminated:
          # Next action:
          # (feed the observation to your agent here)
          action = env.action_space.sample()

          # Processing:
          new_state, reward, terminated, _, info = env.step(action)

          # Accumulate reward
          episode_reward += reward

          if is_training:
            memory.append((state, action, new_state, reward, terminated))

          # Move to new state
          state = new_state

      rewards_per_episode.append(episode_reward)

    env.close()