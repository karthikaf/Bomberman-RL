from collections import namedtuple, deque
from typing import List
import numpy as np
import torch
from torch import nn
import events as e
from .callbacks import state_to_features

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Hyperparameters
TRANSITION_HISTORY_SIZE = 1000
GAMMA = 0.99
BATCH_SIZE = 32
EPS_CLIP = 0.2
NUM_EPOCHS = 5
Critic_Loss = nn.MSELoss()


def calculate_advantage(discounted_rewards, values):
    advantage = discounted_rewards - values
    return advantage


def train_policy_network(policy_net, transitions, optimizer):
    states = torch.stack([transition.state for transition in transitions])
    actions = torch.tensor([ACTIONS.index(transition.action) for transition in transitions],
                           dtype=torch.long).unsqueeze(1)
    old_action_probs = torch.stack([transition.action_probs for transition in transitions])
    returns = torch.tensor([transition.reward for transition in transitions], dtype=torch.float).unsqueeze(1)

    _, values = policy_net(states)
    advantage = calculate_advantage(returns, values)

    for _ in range(NUM_EPOCHS):
        action_probs, new_values = policy_net(states)
        new_action_probs = action_probs.gather(1, actions)
        ratio = new_action_probs / old_action_probs

        surrogate1 = ratio * advantage
        surrogate2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantage
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        value_loss = Critic_Loss(values, returns)

        optimizer.zero_grad()
        loss = policy_loss + value_loss
        loss.backward()
        optimizer.step()


def setup_training(self):
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.total_reward = 0
    self.episode_rewards = []
    self.episode_steps = []


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    reward = reward_from_events(self, events)
    self.total_reward += reward
    self.transitions.append(
        Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    reward = reward_from_events(self, events)
    self.total_reward += reward
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward))

    # Train the policy network at the end of the round
    train_policy_network(self.policy_net, self.transitions, self.optimizer)

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        torch.save(self.policy_net.state_dict(), file)

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.BOMB_DROPPED: 5,
        e.KILLED_SELF: -10,
        e.SURVIVED_ROUND: 10,
        e.MOVED_DOWN: 0.5,
        e.MOVED_UP: 0.5,
        e.MOVED_RIGHT: 0.5,
        e.MOVED_LEFT: 0.5,
        e.WAITED: 0,
        e.INVALID_ACTION: -6
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
