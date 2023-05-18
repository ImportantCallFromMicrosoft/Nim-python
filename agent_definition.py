"""
Code partially taken from:
https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/#sphx-glr-tutorials-training-agents-blackjack-tutorial-py
"""

import json
from copy import deepcopy
from collections import defaultdict

import numpy as np


class NimGameState:
    def __init__(self, slot1: int, slot2: int, slot3: int, slot4: int, slot5: int):
        self._state = [slot1, slot2, slot3, slot4, slot5]

    def __hash__(self):
        return (
            self._state[0]
            + 10 * self._state[1]
            + 100 * self._state[2]
            + 1000 * self._state[3]
            + 10000 * self._state[4]
        )

    def __getitem__(self, index):
        return self._state[index]

    def __setitem__(self, index, value):
        self._state[index] = value

    def __str__(self):
        res = ""
        for i, slot in enumerate(self._state):
            res += f"Slot {i+1}: {'#'*slot}\n"
        return res

    def __eq__(self, other):
        return hash(self) == hash(other)

    @staticmethod
    def from_hash(hash: int):
        slot1 = hash % 10
        hash = hash // 10
        slot2 = hash % 10
        hash = hash // 10
        slot3 = hash % 10
        hash = hash // 10
        slot4 = hash % 10
        hash = hash // 10
        slot5 = hash % 10
        return NimGameState(slot1, slot2, slot3, slot4, slot5)


class NimAction:
    NUM_ACTIONS = 25

    def __init__(self, slot: int, amount: int):
        self.slot = slot
        self.amount = amount

    def __hash__(self):
        return hash((self.slot, self.amount))

    def get_idx(self):
        return self.slot * 5 + self.amount

    @staticmethod
    def from_idx(idx: int):
        return NimAction(idx // 5, idx % 5)

    def __str__(self):
        return f"Action({self.slot}, {self.amount})"


class GameEnvironment:
    START_STATE = NimGameState(1, 2, 3, 4, 5)
    DEFEAT_STATE = NimGameState(0, 0, 0, 0, 0)
    PUNISHMENT_ILLEGAL_MOVE = -10
    REWARD_LEGAL_MOVE = 0
    PUNISHMENT_DEFEAT = -1

    def step(self, action: NimAction) -> tuple[NimGameState, int, bool, dict]:
        if (action.amount + 1) <= self.state[action.slot]:
            self.state[action.slot] = self.state[action.slot] - (action.amount + 1)
        else:
            return deepcopy(self.state), self.PUNISHMENT_ILLEGAL_MOVE, True, True, {}

        if self.state == self.DEFEAT_STATE:
            return deepcopy(self.state), self.PUNISHMENT_DEFEAT, True, False, {}

        return deepcopy(self.state), self.REWARD_LEGAL_MOVE, False, False, {}

    def reset(self):
        self.state = deepcopy(self.START_STATE)
        return deepcopy(self.state), {}

    def render(self, mode="human"):
        return


class NimAgent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.q_values = defaultdict(lambda: np.zeros(NimAction.NUM_ACTIONS))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error: list[float] = []

    def get_action(self, obs: int, opponent: bool = False) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon + ((1 - self.epsilon) / 2 * opponent):
            return np.random.randint(NimAction.NUM_ACTIONS)

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: int,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: int,
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward - self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    @staticmethod
    def load(filename):
        with open(filename) as f:
            data = json.load(f)
            q_values = defaultdict(lambda: np.zeros(NimAction.NUM_ACTIONS))
            for k, v in data["q_values"].items():
                q_values[int(k)] = np.array(v)
            del data["q_values"]
            agent = NimAgent(0.0, 0.0, 0.0, 0.0)
            agent.__dict__.update(data)
            agent.q_values = q_values
        return agent

    def save(self, filename):
        with open(filename, "w") as f:
            data = self.__dict__.copy()
            data["q_values"] = {k: v.tolist() for k, v in self.q_values.items()}
            json.dump(data, f)
