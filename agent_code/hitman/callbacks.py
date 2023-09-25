import os
import pickle
import random
import settings
import numpy as np
from pathlib import Path
from .config import GameState
from typing import List, Tuple

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']


def setup(self):
    # Game state features are manually defined below
    state_features = {
        "tile_up": ['free_tile', 'coin', 'invalid', 'adjacent', 'explosion', 'ticking'],
        "tile_down": ['free_tile', 'coin', 'invalid', 'adjacent', 'explosion', 'ticking'],
        "tile_left": ['free_tile', 'coin', 'invalid', 'adjacent', 'explosion', 'ticking'],
        "tile_right": ['free_tile', 'coin', 'invalid', 'adjacent', 'explosion', 'ticking'],
        "compass": ["N", "S", "W", "E", "NP"],
        "compass_mode": ["coin", "crate", "escape", "attack"]
    }

    # Create the state space object
    self.state_space = GameState(state_features)

    # Define checkpoint filename and load Q-table if available
    checkpoint_filename = "checkpoint-50000.pt"
    saved_model_filename = "trained-model.pt"

    if self.train and os.path.isfile(checkpoint_filename):
        print("Loading the trained-model from checkpoint")
        self.logger.info(f"Loading the trained-model from {checkpoint_filename}")
        self.checkpoint_rounds = int(Path(checkpoint_filename).stem.split("_")[-1])
        with open(checkpoint_filename, "rb") as file:
            self.q_table = pickle.load(file)
    elif self.train or not os.path.isfile(saved_model_filename):
        print("Training the agent from the beginning")
        self.logger.info("Training the agent from the beginning")
        self.q_table = np.zeros((len(self.state_space), len(ACTIONS)))
    else:
        print("Loading from the trained-model.")
        self.logger.info("Loading from the trained-model.")
        with open(saved_model_filename, "rb") as file:
            self.q_table = pickle.load(file)


def act(self, game_state: dict) -> str:
    agent_state = game_state_to_feature(self, game_state)

    # Define epsilon parameters
    epsilon_initial = 1.0
    epsilon_decay = 0.999
    epsilon_min = 0.01

    # Calculate epsilon based on the current round and training status
    if self.train:
        checkpoint_rounds = self.checkpoint_rounds if hasattr(self, "checkpoint_rounds") else 0
        epsilon = max(epsilon_min, epsilon_initial * epsilon_decay ** (game_state["round"] - 1 + checkpoint_rounds))
    else:
        epsilon = max(epsilon_min, epsilon_initial * epsilon_decay ** (game_state["round"] - 1))

    # Explore or exploit based on epsilon-greedy strategy
    if self.train and random.random() < epsilon:
        chosen_action = np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
        self.logger.debug(
            f"[{game_state['round']:06d}|{game_state['step']:03d}] Agent state: {self.state_space.get_state(agent_state)}")
        self.logger.info(f"[{game_state['round']:06d}|{game_state['step']:03d}] Exploring {chosen_action}")
    else:
        chosen_action = ACTIONS[np.argmax(self.q_table[agent_state])]
        self.logger.debug(
            f"[{game_state['round']:06d}|{game_state['step']:03d}] Agent state: {self.state_space.get_state(agent_state)}")
        self.logger.debug(f"[{game_state['round']:06d}|{game_state['step']:03d}] QTable: {self.q_table[agent_state]}")
        self.logger.debug(f"[{game_state['round']:06d}|{game_state['step']:03d}] Action {chosen_action}")
    return chosen_action


def game_state_to_feature(self, game_state):
    if game_state is None:
        return None

    agent_state = dict()

    x, y = game_state['self'][3]
    arena = game_state['field']
    bombs = game_state['bombs']
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    explosion_map = game_state['explosion_map']
    bomb_map = compute_bomb_map(arena, bombs)

    # Defining hyperparameters for the reach
    crate_schedule = np.array([1, 20, 60, 100, 160, 220, 300])
    crate_reach = np.max(np.where(game_state["step"] >= crate_schedule)) + 1
    coin_reach = min((game_state["step"] // 50) + 5, 7)

    # ticking bomb feature computes the compass toward the bomb
    if bomb_map[x, y] != 5:
        # Set compass mode to escape
        agent_state["compass_mode"] = "escape"
        my_position = (x, y)

        # the agent will run away from the bomb, because the feature points to the bomb
        if my_position in [b[0] for b in bombs]:
            free_space = arena == 0
            for o in others:
                free_space[o] = False

            safe_space = free_space & (bomb_map == 5)
            safe_space_indices = list(map(tuple, np.transpose(np.nonzero(safe_space))))

            best_direction, _, _ = breadth_first_search(free_space, (x, y), safe_space_indices)

            if best_direction == (x, y - 1):
                agent_state["compass"] = "S"
            elif best_direction == (x, y + 1):
                agent_state["compass"] = "N"
            elif best_direction == (x - 1, y):
                agent_state["compass"] = "E"
            elif best_direction == (x + 1, y):
                agent_state["compass"] = "W"
            else:
                agent_state["compass"] = "NP"

        else:
            ticking_bomb_direction = compute_ticking_bomb_feature(bombs, my_position)
            agent_state["compass"] = ticking_bomb_direction

    elif (
            (len(coins) > 0) and
            ((np.max(np.abs(np.array(coins) - (x, y)), axis=1) <= coin_reach).any() or
             (len(others) == 0 and (np.count_nonzero(arena == 1) == 0))
            )
    ):

        # Set compass mode to coin collecting
        agent_state["compass_mode"] = "coin"

        # Look for the shortest distance between the agent and coins
        free_space = arena == 0
        for o in others:
            free_space[o] = False
        targets = coins

        best_direction, _, _ = breadth_first_search(free_space, (x, y), targets)

        if best_direction == (x, y - 1):
            agent_state["compass"] = "N"
        elif best_direction == (x, y + 1):
            agent_state["compass"] = "S"
        elif best_direction == (x - 1, y):
            agent_state["compass"] = "W"
        elif best_direction == (x + 1, y):
            agent_state["compass"] = "E"
        else:
            agent_state["compass"] = "NP"

    elif len(others) > 0:
        # attack feature is computed below
        agent_state["compass_mode"] = "attack"
        best_enemy_direction = compute_attack_feature(others, (x, y))
        agent_state["compass"] = best_enemy_direction

    else:
        # Set compass mode to coin collecting
        agent_state["compass_mode"] = "crate"

        best_crate_direction = compute_crate_feature(arena, bomb_map, others, (x, y), crate_reach)
        agent_state["compass"] = best_crate_direction

    # Check if there are free tiles in the adjacent tile, x & y are the current position
    directions = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]

    feature_keys = ["tile_up", "tile_down", "tile_left", "tile_right"]
    for (key, d) in zip(feature_keys, directions):

        if (
                agent_state["compass_mode"] == "escape" and
                bomb_map[d] != 5 and
                arena[d] == 1
        ):
            agent_state[key] = "adjacent"

        # This tile is in danger of explosion, or explosion is ongoing
        elif explosion_map[d] != 0:
            agent_state[key] = "explosion"

        elif bomb_map[d] != 5:
            agent_state[key] = "ticking"

        elif d in coins:
            agent_state[key] = "coin"

        # enemy agents with bombs are in adjacent
        elif d in others:
            agent_state[key] = "adjacent"

        elif arena[d] == 0:
            agent_state[key] = "free_tile"

        # That's a crate
        elif arena[d] == 1:
            agent_state[key] = "adjacent"

        elif arena[d] == -1:
            agent_state[key] = "invalid"

    # agent_state_index will contain the index of present state
    agent_state_index = self.state_space.get_index(agent_state)

    return agent_state_index


def compute_bomb_map(arena, bombs):
    bomb_map = np.full(arena.shape, 5, dtype=np.int32)
    for (x_bomb, y_bomb), t in bombs:
        x_blast, y_blast = compute_blast(arena, (x_bomb, y_bomb), settings.BOMB_POWER)
        bomb_map[x_blast, y_bomb] = t
        bomb_map[x_bomb, y_blast] = t
    return bomb_map


def compute_blast(arena, bomb_position, bomb_power):
    x_bomb, y_bomb = bomb_position

    l_spread = -bomb_power
    r_spread = bomb_power + 1
    if arena[x_bomb - 1, y_bomb] == -1:  # Wall to the left
        l_spread = 0
    if arena[x_bomb + 1, y_bomb] == -1:  # Wall to the right
        r_spread = 1

    u_spread = -bomb_power
    d_spread = bomb_power + 1
    if arena[x_bomb, y_bomb - 1] == -1:  # Wall above
        u_spread = 0
    if arena[x_bomb, y_bomb + 1] == -1:  # Wall below
        d_spread = 1

    x_blast_spread = np.expand_dims(np.arange(l_spread, r_spread), axis=0)
    y_blast_spread = np.expand_dims(np.arange(u_spread, d_spread), axis=0)

    x_blast = np.array([[x_bomb]]) + x_blast_spread
    x_blast = x_blast[(x_blast > 0) & (x_blast < arena.shape[0] - 1)]

    y_blast = np.array([[y_bomb]]) + y_blast_spread
    y_blast = y_blast[(y_blast > 0) & (y_blast < arena.shape[1] - 1)]

    return x_blast, y_blast


def breadth_first_search(free_space, start: Tuple[int, int], targets: List[Tuple[int, int]]):

    if len(targets) == 0:
        return None, None, 0

    # Initializing the queue
    queue = [start]
    explored = {start: None}

    # Find the first target using BFS
    closest_node = None
    while len(queue) > 0:
        current_node = queue.pop(0)

        if current_node in targets:
            closest_node = current_node
            break

        x, y = current_node
        neighbor_nodes = [(x, y + 1), (x, y - 1), (x - 1, y), (x + 1, y)]
        walkable_neighbour_nodes = [n for n in neighbor_nodes if free_space[n]]

        for neighbor_node in walkable_neighbour_nodes:
            if neighbor_node not in explored.keys():
                queue.append(neighbor_node)
                explored[neighbor_node] = current_node

    # If no target can be reached, agent returns from point x to y direction and distance to the closest target.
    if closest_node is None:
        dists = np.sum(np.abs((np.array(targets) - start)), axis=1)
        closest_node_index = np.argmin(dists)
        closest_node = targets[closest_node_index]
        dist = dists[closest_node_index]

        if closest_node[1] > start[1]:
            direction_node = (start[0], start[1] + 1)
        elif closest_node[1] < start[1]:
            direction_node = (start[0], start[1] - 1)
        elif closest_node[0] > start[0]:
            direction_node = (start[0] + 1, start[1])
        elif closest_node[0] < start[0]:
            direction_node = (start[0] - 1, start[1])
        else:
            direction_node = None
            closest_node = None
            dist = 0

    # Occurrence of agent standing on a coin
    elif closest_node == start:
        dist = 0
        direction_node = start

    # Backtracking the target if it can be reached
    else:
        dist = 1
        direction_node = closest_node
        while explored[direction_node] != start:
            direction_node = explored[direction_node]
            dist += 1

    return direction_node, closest_node, dist


def compute_num_crates_exploded(arena, bomb_location, bomb_power):
    x_bomb, y_bomb = bomb_location
    x_blast, y_blast = compute_blast(arena, (x_bomb, y_bomb), bomb_power)

    crate_map = (arena == 1).astype(int)
    crates_exploded = np.sum(crate_map[x_blast, y_bomb]) + np.sum(crate_map[x_bomb, y_blast])

    return crates_exploded


def compute_crate_feature(arena, bomb_map, others, my_location, reach):
    possible_locations = (arena == 0) & (bomb_map == 5)
    for o in others:
        possible_locations[o] = False

    # checking the nearby area within reach
    x, y = my_location
    possible_locations[min(x + reach + 1, possible_locations.shape[0]):, :] = False
    possible_locations[:max(x - reach, 0), :] = False
    possible_locations[:, min(y + reach + 1, possible_locations.shape[1]):] = False
    possible_locations[:, :max(y - reach, 0)] = False

    optimal_location = (-1, -1)
    max_crates_exploded = 0
    for px, py in np.transpose(possible_locations.nonzero()):

        # Consider only positions directly next to crates
        if (
                arena[px - 1, py] == 1 or arena[px + 1, py] == 1
                or arena[px, py - 1] == 1 or arena[px, py + 1] == 1
        ):
            crates_exploded = compute_num_crates_exploded(arena, (px, py), settings.BOMB_POWER)

            if crates_exploded > max_crates_exploded:
                optimal_location = (px, py)
                max_crates_exploded = crates_exploded

    # If there are no crates nearby, then we check globally for the closest crate.
    # Once we reach the closest crate, optimal crate can be found
    if max_crates_exploded == 0:
        crates = np.transpose((arena == 1).nonzero())

        if len(crates) == 0:
            return "NP"

        optimal_location = crates[
            np.argmin(np.sum(np.abs(crates - my_location), axis=1), axis=0)
        ]

    if optimal_location[1] > my_location[1]:
        return "S"
    elif optimal_location[1] < my_location[1]:
        return "N"

    if optimal_location[0] > my_location[0]:
        return "E"
    elif optimal_location[0] < my_location[0]:
        return "W"
    else:
        return "NP"


def compute_ticking_bomb_feature(bombs, my_position):
    if len(bombs) == 0:
        return "NP"

    reach = 3
    bomb_positions = np.array([b[0] for b in bombs])
    x_bomb_positions, y_bomb_positions = bomb_positions[:, 0], bomb_positions[:, 1]

    # Mask bombs that are in the same row or column as our agent
    mask_x = np.abs(x_bomb_positions - my_position[0]) == 0
    mask_y = np.abs(y_bomb_positions - my_position[1]) == 0
    dangerous_bomb_positions = bomb_positions[mask_x | mask_y, :]

    if len(dangerous_bomb_positions) == 0:
        return "NP"

    closest_dangerous_bomb = dangerous_bomb_positions[
        np.argmin(np.sum(np.abs(dangerous_bomb_positions - my_position), axis=1), axis=0)
    ]

    diff = closest_dangerous_bomb - my_position

    # We only care about bombs {reach} tiles away from us
    if np.max(np.abs(diff)) > reach:
        return "NP"

    # Compute in which direction the bomb is, relative to us
    if diff[0] < 0:  # x-axis
        return "W"
    elif diff[0] > 0:  # x-axis
        return "E"
    elif diff[1] < 0:  # y-axis
        return "N"
    elif diff[1] > 0:  # y-axis
        return "S"
    else:
        return "NP"  # We are standing on it


def compute_attack_feature(others, my_location):
    closest_enemy_index = np.argmin(np.sum(np.abs(np.array(others) - my_location), axis=1), 0)
    closest_enemy_location = np.array(others[closest_enemy_index])

    diff = closest_enemy_location - my_location

    if (
            (np.abs(diff) <= 2).all()  # We are 1 or 2 tiles away from the enemy
    ):
        return "NP"

    # Compute in which direction the bomb is, relative to us
    if diff[0] < 0:  # x-axis
        return "W"
    elif diff[0] > 0:  # x-axis
        return "E"
    elif diff[1] < 0:  # y-axis
        return "N"
    elif diff[1] > 0:  # y-axis
        return "S"
    else:
        return "NP"
