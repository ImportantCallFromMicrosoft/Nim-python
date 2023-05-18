import os

import tqdm
import matplotlib.pyplot as plt
import numpy as np

from agent_definition import NimAgent, GameEnvironment, NimAction, NimGameState


VERBOSE = False


def print_if_verbose(msg: str):
    if VERBOSE:
        print(msg)


def get_valid_state_from_opponent(
    env: GameEnvironment, agent: NimAgent, state: NimGameState
):
    print_if_verbose(env.state)
    truncated_other = True
    while truncated_other:
        action_other = NimAction.from_idx(agent.get_action(hash(state), opponent=True))
        print_if_verbose(action_other)
        next_state, reward, terminated, truncated_other, _ = env.step(action_other)
    if reward == env.REWARD_VICTORY:
        reward = -env.REWARD_VICTORY
    else:
        reward = 0
    return next_state, terminated, reward


# hyperparameters
LOAD_AGENT = True
learning_rate = 0.003
n_episodes = 1000000
start_epsilon = 0.5
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.001

agent = NimAgent(
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

if LOAD_AGENT and os.path.exists("agent.json"):
    agent = NimAgent.load("agent.json")

env = GameEnvironment()

episode_lengths = []
episode_rewards = []
episode_truncated = []

for episode in tqdm.tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # play one episode
    steps = 0
    # with probability 0.5, let the opponent start
    if np.random.rand() < 0.5:
        next_obs, other_terminated, other_reward = get_valid_state_from_opponent(
            env, agent, obs
        )
        obs = next_obs

    while not done:
        # Agent takes action
        print_if_verbose(env.state)
        action = NimAction.from_idx(agent.get_action(hash(obs)))
        print_if_verbose(action)
        next_obs, reward, done, truncated, _ = env.step(action)
        agent.update(hash(obs), action.get_idx(), reward, done, hash(next_obs))
        steps += 1

        # let opponent take action
        if not done:
            next_next_obs, done, other_reward = get_valid_state_from_opponent(
                env, agent, next_obs
            )
            reward += other_reward
            steps += 1
            obs = next_next_obs

    print_if_verbose(env.state)
    print_if_verbose("Episode finished")

    episode_lengths.append(steps)
    episode_rewards.append(reward)
    episode_truncated.append(truncated)
    agent.decay_epsilon()

# save the agent
agent.save("agent.json")

# plot the training error
rolling_length = 500
fig, axs = plt.subplots(ncols=4, figsize=(16, 5))

axs[0].set_title("Episode rewards")
# compute and assign a rolling average of the data to provide a smoother graph
reward_moving_average = (
    np.convolve(
        np.array(episode_rewards).flatten(), np.ones(rolling_length), mode="valid"
    )
    / rolling_length
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)

axs[1].set_title("Episode lengths")
length_moving_average = (
    np.convolve(
        np.array(episode_lengths).flatten(), np.ones(rolling_length), mode="valid"
    )
    / rolling_length
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)

axs[2].set_title("Ratio of truncated episodes to total episodes")
training_truncated_ratio = (
    np.convolve(np.array(episode_truncated), np.ones(rolling_length), mode="valid")
    / rolling_length
)
axs[2].plot(range(len(training_truncated_ratio)), training_truncated_ratio)

axs[3].set_title("Ratio of won episodes to total episodes")
training_win_ratio = (
    np.convolve(
        np.array([r > 0 for r in episode_rewards]),
        np.ones(rolling_length),
        mode="valid",
    )
    / rolling_length
)
axs[3].plot(range(len(training_win_ratio)), training_win_ratio)
plt.tight_layout()
plt.show()
