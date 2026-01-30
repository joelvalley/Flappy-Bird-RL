import os
import yaml
import torch
import random
import argparse
import itertools
import gymnasium
from dqn import DQN
from torch import nn
import flappy_bird_gymnasium
from datetime import datetime, timedelta
from experience_replay import ReplayMemory

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# Device agnostic code
device = "mps" if torch.backends.mps.is_available() else "cpu"

class Agent:
  def __init__(self, hyperparameter_set):
    with open("hyperparameters.yaml", 'r') as file:
      all_hyperparameter_sets = yaml.safe_load(file)
      hyperparameters = all_hyperparameter_sets[hyperparameter_set]

    self.replay_memory_size = hyperparameters['replay_memory_size']
    self.mini_batch_size = hyperparameters['mini_batch_size']
    self.epsilon_init = hyperparameters['epsilon_init']
    self.epsilon_decay = hyperparameters['epsilon_decay']
    self.epsilon_min = hyperparameters['epsilon_min']
    self.network_sync_rate = hyperparameters['network_sync_rate']
    self.learning_rate_a = hyperparameters['learning_rate_a']
    self.discount_factor_g = hyperparameters['discount_factor_g']
    self.stop_on_reward = hyperparameters['stop_on_reward']
    self.hidden_dim = hyperparameters['hidden_dim']
    self.use_lidar = hyperparameters['use_lidar']

    self.loss_fn = nn.MSELoss()
    self.optimizer = None

    self.LOG_FILE = os.path.join(RUNS_DIR, f"{hyperparameter_set}.log")
    self.MODEL_FILE = os.path.join(RUNS_DIR, f"{hyperparameter_set}.pt")


  def run(self, is_training=True, render=False):
    env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=self.use_lidar)

    num_actions = env.action_space.n
    num_states = env.observation_space.shape[0]

    rewards_per_episode = []
    epsilon_history = []

    policy_dqn = DQN(input_dim=num_states, 
                     hidden_dim=self.hidden_dim, 
                     output_dim=num_actions).to(device)
    
    if is_training:
       memory = ReplayMemory(self.replay_memory_size)

       epsilon = self.epsilon_init

       target_dqn = DQN(input_dim=num_states, 
                        hidden_dim=self.hidden_dim, 
                        output_dim=num_actions).to(device)
       target_dqn.load_state_dict(policy_dqn.state_dict())

       step_count = 0

       self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

       epsilon_history = []

       best_reward = -99999999
    else:
      policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

      policy_dqn.eval()

    for episode in itertools.count():
      # Reset the environment
      state, _ = env.reset()
      state = torch.tensor(state, dtype=torch.float32, device=device)

      terminated = False
      episode_reward = 0

      # Loop through 1 episode
      while not terminated:
          
          if is_training and random.random() < epsilon:
            action = torch.tensor(env.action_space.sample(), dtype=torch.int64, device=device)

            step_count += 1
          else:
            with torch.inference_mode():
              action = policy_dqn(state.unsqueeze(0)).squeeze(0).argmax()

          # Processing:
          new_state, reward, terminated, _, info = env.step(action.item())

          # Accumulate reward
          episode_reward += reward

          new_state = torch.tensor(new_state, dtype=torch.float32, device=device)
          reward = torch.tensor(reward, dtype=torch.float32, device=device)
          terminated = torch.tensor(terminated, dtype=torch.bool, device=device)

          if is_training:
            memory.append((state, action, new_state, reward, terminated))

            step_count += 1

          # Move to new state
          state = new_state

      rewards_per_episode.append(episode_reward)

      if is_training:
        if episode_reward > best_reward:
          log_message = f"{datetime.now().strftime(DATE_FORMAT)} - New best reward: {episode_reward} in episode {episode}"
          print(log_message)
          with open(self.LOG_FILE, 'a') as file:
            file.write(log_message + "\n")

          torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
          best_reward = episode_reward

      # Epsilon greedy
      epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
      epsilon_history.append(epsilon)

      # Training & syncing the networks
      if len(memory) > self.mini_batch_size:
        mini_batch = memory.sample(self.mini_batch_size)

        self.optimize(mini_batch, policy_dqn, target_dqn)

        if step_count > self.network_sync_rate:
          target_dqn.load_state_dict(policy_dqn.state_dict())
          step_count = 0

  def optimize(self, mini_batch, policy_dqn, target_dqn):
    states, actions, new_states, rewards, terminations = zip(*mini_batch)

    states = torch.stack(states)

    actions = torch.stack(actions)

    new_states = torch.stack(new_states)

    rewards = torch.stack(rewards)
    terminations = torch.stack(terminations).float().to(device)
    
    with torch.no_grad():
      target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

    current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

    # Compute loss
    loss = self.loss_fn(current_q, target_q)

    # Optimize model (zero gradients, backpropagate, gradient descent)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--hyperparameters', '-hp', default='flappybird', help='')
  parser.add_argument('--train', action='store_true', help='Training mode')
  args = parser.parse_args()

  agent = Agent(hyperparameter_set=args.hyperparameters)

  if args.train:
    agent.run(is_training=True)
  else:
    agent.run(is_training=False, render=True)