import os

import tqdm
import matplotlib.pyplot as plt
import numpy as np

from agent_definition import NimAgent, NimGameEnvironment, NimAction, NimGameState


VERBOSE = False


def print_if_verbose(msg: str):
    if VERBOSE:
        print(msg)


def get_valid_action_from_opponent(
    env: NimGameEnvironment, agent: NimAgent
):
    invalid = True
    while invalid:
        action = NimAction.from_idx(agent.get_action(hash(env.state), opponent=True))
        invalid = not env.action_valid(action)
    return action

def let_opponent_move(env: NimGameEnvironment, agent: NimAgent) -> NimGameState:
    action = get_valid_action_from_opponent(env, agent)
    obs, _, done, truncated, _ = env.step(action)
    assert not truncated
    return obs, done

# hyperparameters
LOAD_AGENT = True
LEARNING_RATE = 0.01
NUM_EPISODES = 100000
INITIAL_EPSILON = 0.5
EPSILON_DECAY = INITIAL_EPSILON / (NUM_EPISODES / 2)  # reduce the exploration over time
FINAL_EPSILON = 0.001

def main():
    agent = NimAgent(
        learning_rate=LEARNING_RATE,
        initial_epsilon=INITIAL_EPSILON,
        epsilon_decay=EPSILON_DECAY,
        final_epsilon=FINAL_EPSILON,
    )

    if LOAD_AGENT and os.path.exists("agent.json"):
        agent = NimAgent.load("agent.json")

    env = NimGameEnvironment()

    episode_lengths = []
    episode_rewards = []
    episode_truncated = []

    for _ in tqdm.tqdm(range(NUM_EPISODES)):
        obs, info = env.reset()
        done = False

        steps = 0
        
        # with probability 0.5, let the opponent start
        if np.random.rand() < 0.5:
            obs, _ = let_opponent_move(env, agent)
            steps += 1

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
                obs, done = let_opponent_move(env, agent)
                steps += 1

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
            np.array([r == 0 for r in episode_rewards]),
            np.ones(rolling_length),
            mode="valid",
        )
        / rolling_length
    )
    axs[3].plot(range(len(training_win_ratio)), training_win_ratio)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
