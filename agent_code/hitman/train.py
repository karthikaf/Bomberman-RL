import pickle
import numpy as np
import events as e
from typing import List
from pathlib import Path
from .callbacks import ACTIONS
from collections import namedtuple, deque
from .callbacks import game_state_to_feature

compass_mapping = {
    "N": {"90": "E", "180": "S", "270": "W"},
    "S": {"90": "W", "180": "N", "270": "E"},
    "W": {"90": "N", "180": "E", "270": "S"},
    "E": {"90": "S", "180": "W", "270": "N"},
}

action_mapping = {
    "UP": {"90": "RIGHT", "180": "DOWN", "270": "LEFT"},
    "DOWN": {"90": "LEFT", "180": "UP", "270": "RIGHT"},
    "LEFT": {"90": "UP", "180": "RIGHT", "270": "DOWN"},
    "RIGHT": {"90": "DOWN", "180": "LEFT", "270": "UP"},
    "WAIT": {"90": "WAIT", "180": "WAIT", "270": "WAIT"},
    "BOMB": {"90": "BOMB", "180": "BOMB", "270": "BOMB"},
}

# Manually defined custom events
PERFECT_CRATE_OCCURRENCE = "PERFECT CRATE"
BLOCKED_PASSAGE_BOMB_EVENT = "BLOCKED PASSAGE BOMB EVENT"
FOLLOWED_COMPASS_DIRECTION = "FOLLOWED COMPASS DIRECTION"
ADJACENT_ENEMY_EVENT = "ADJACENT ENEMY EVENT"

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# should be greater than or equal to 1, since it defines temporal difference of N
RECENT_HISTORY_SIZE = 4


def setup_training(self):
    self.transitions = deque(maxlen=RECENT_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(
        'Encountered game event(s) {} in step {}'.format(", ".join(map(repr, events)), new_game_state["step"]))

    old_agent_state, new_agent_state = game_state_to_feature(self, old_game_state), game_state_to_feature(self,
                                                                                                          new_game_state)

    action_to_compass_direction = {"UP": "N", "DOWN": "S", "LEFT": "W", "RIGHT": "E"}

    if old_agent_state is not None and self_action in action_to_compass_direction and \
            self.state_space.get_state(old_agent_state)["compass"] == action_to_compass_direction[self_action]:
        events.append(FOLLOWED_COMPASS_DIRECTION)

    if self_action == "BOMB" and old_agent_state is not None:

        compass_mode = self.state_space.get_state(old_agent_state)["compass_mode"]

        if compass_mode == "crate" and self.state_space.get_state(old_agent_state)["compass"] == "NP":
            events.append(PERFECT_CRATE_OCCURRENCE)

        if compass_mode in ["coin", "attack"]:
            compass_directions = ["N", "S", "E", "W"]
            shifts = [[0, -1], [0, 1], [1, 0], [-1, 0]]
            for compass_direction, shift in zip(compass_directions, shifts):
                x, y = np.array(old_game_state["self"][3]) + np.array(shift)
                if self.state_space.get_state(old_agent_state)["compass"] == compass_direction and \
                        old_game_state["field"][x, y] == 1:
                    events.append(BLOCKED_PASSAGE_BOMB_EVENT)

        if compass_mode == "coin" and self.state_space.get_state(old_agent_state)["compass"] == "NP":
            events.append(BLOCKED_PASSAGE_BOMB_EVENT)

        if compass_mode == "attack" and self.state_space.get_state(old_agent_state)["compass"] == "NP":
            events.append(ADJACENT_ENEMY_EVENT)

    self.transitions.append(Transition(old_agent_state, self_action, new_agent_state, reward_from_events(self, events)))

    if len(self.transitions) == self.transitions.maxlen:
        rotational_update_q_table(self)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    last_agent_state = game_state_to_feature(self, last_game_state)
    self.transitions.append(Transition(last_agent_state, last_action, None, reward_from_events(self, events)))

    while self.transitions:
        rotational_update_q_table(self)

    with open("trained-model.pt", "wb") as file:
        pickle.dump(self.q_table, file)

    # Checkpoint is created for every 1000 rounds to avoid redundancy
    if last_game_state["round"] > 0 and last_game_state["round"] % 1000 == 0:
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_rounds = getattr(self, "checkpoint_rounds", 0)
        filename = f"checkpoint-{last_game_state['round'] + checkpoint_rounds}.pt"
        with open(checkpoint_dir / filename, "wb") as file:
            pickle.dump(self.q_table, file)


def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.MOVED_UP: -0.1,
        e.MOVED_DOWN: -0.1,
        e.MOVED_RIGHT: -0.1,
        e.MOVED_LEFT: -0.1,
        e.INVALID_ACTION: -0.6,
        e.BOMB_DROPPED: -0.5,
        e.WAITED: -0.4,
        e.CRATE_DESTROYED: 1.5,
        e.GOT_KILLED: -5,
        e.KILLED_SELF: -3,
        e.SURVIVED_ROUND: 1.5,
        e.KILLED_OPPONENT: 5,
        PERFECT_CRATE_OCCURRENCE: 2.0,
        BLOCKED_PASSAGE_BOMB_EVENT: 1.0,
        FOLLOWED_COMPASS_DIRECTION: 0.4,
        ADJACENT_ENEMY_EVENT: 1.0
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def rotational_update_q_table(self):
    origin_state_index, action = self.transitions[0][:2]

    if origin_state_index is None:
        self.transitions.popleft()
        return

    origin_state = self.state_space.get_state(origin_state_index)
    rotations = [0, 90, 180, 270]

    for rotation in rotations:
        rotated_state = dict(origin_state)

        for _ in range(rotation // 90):
            rotated_state['tile_up'], rotated_state['tile_down'], rotated_state['tile_left'], rotated_state['tile_right'] = \
                rotated_state['tile_left'], rotated_state['tile_right'], rotated_state['tile_down'], rotated_state['tile_up']

        # Update compass and action based on rotation
        compass_rotation = str(rotation % 360)
        rotated_state['compass'] = compass_mapping.get(origin_state['compass'], {}).get(compass_rotation, "NP")
        rotated_action = action_mapping.get(action, {}).get(compass_rotation, "BOMB")

        rotated_state_index = self.state_space.get_index(rotated_state)
        update_q_table(self, rotated_state_index, rotated_action)

    self.transitions.popleft()


def update_q_table(self, origin_state_index, action):
    alpha = 0.08
    gamma = 0.9
    end_state = self.transitions[-1][2]

    # Collect rewards and discount them by gamma
    discounted_rewards = sum([gamma ** i * s[-1] for i, s in enumerate(self.transitions)])

    old_q_value = self.q_table[origin_state_index, ACTIONS.index(action)]

    if end_state is None:
        q_remainder_estimate = 0.0
    else:
        q_remainder_estimate = self.q_table[end_state].max()

    updated_q_value = old_q_value + alpha * (discounted_rewards
                                             + gamma ** len(self.transitions) * q_remainder_estimate
                                             - old_q_value)

    self.q_table[origin_state_index, ACTIONS.index(action)] = updated_q_value
