import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

class ValueNetwork(nn.Module):
    def __init__(self, input_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Error occurs here
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

def setup(self):
    if self.train or not os.path.isfile("ppo_model.pth"):
        self.logger.info("Setting up model from scratch.")
        self.policy_network = PolicyNetwork(input_size=198, output_size=len(ACTIONS))
        self.value_network = ValueNetwork(input_size=198)
        self.optimizer_policy = optim.Adam(self.policy_network.parameters(), lr=0.001)
        self.optimizer_value = optim.Adam(self.value_network.parameters(), lr=0.001)
    else:
        self.logger.info("Loading model from saved state.")
        self.policy_network = torch.load("ppo_policy_model.pth")
        self.value_network = torch.load("ppo_value_model.pth")

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.state_size = 198  # Modify based on your state representation
    self.action_size = len(ACTIONS)

    # Hyperparameters for PPO
    self.gamma = 0.99
    self.eps_clip = 0.2
    self.policy_net = PolicyNetwork(self.state_size, self.action_size)
    self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.01)
    self.policy_net.train()

    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.policy_net.load_state_dict(torch.load(file))
        self.policy_net.eval()


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    state = state_to_features(game_state)
    state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)

    # Get action probabilities from the policy network
    action_probs, _ = self.policy_net(state_tensor)

    # Choose action using the probability distribution
    m = distributions.Categorical(action_probs)
    action_index = m.sample()
    action = ACTIONS[action_index.item()]

    return action

def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to the input of your model, i.e., a feature vector.

    :param game_state: A dictionary describing the current game board.
    :return: np.array
    """

    # Game board
    board = game_state['field']

    # Convert board to 1 for crates, -1 for walls, 0 for free tiles
    board = np.where(board == 1, 1, 0)  # crates
    board = np.where(board == -1, -1, board)  # walls

    # Bombs
    bombs = np.zeros_like(board)
    for (x, y), countdown in game_state['bombs']:
        bombs[x, y] = countdown

    # Explosions
    explosion_map = game_state['explosion_map']

    # Coins
    coins = np.zeros_like(board)
    for (x, y) in game_state['coins']:
        coins[x, y] = 1

    # Agent information
    agent = np.zeros(board.shape)
    x, y = game_state['self'][3]
    agent[x, y] = 1

    # Concatenate all information into a single feature vector
    stacked_channels = np.stack([board, bombs, explosion_map, coins, agent])

    # Flatten and return as a vector
    return stacked_channels.reshape(-1)

