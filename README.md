# Training Q-Learner against itself to play Nim

## Introduction

Nim is a simple game, where two players draw a number of boxes from a pile. The player who draws the last box loses. The game is also known as Nim.

This projects demonstrates how to train a reinforcement learning agent to play Nim. The agent is trained against itself using Q-Learning.

## How to run

### Requirements

* Python 3.10 with pip

### Installation

Make sure you have added python 3.10 to your command line path.
Navigate into the repository root folder. Run the following command to install the required packages:

```bash
python -m pip install -r requirements.txt
```

### Training

To train the agent run the following command:

```bash
python -m train_agent.py
```

This creates a `agent.json` file in the root folder. This file contains the trained agents weights for the individual actions. The agent is trained for 1000000 episodes
which should last ~5 minutes on a modern CPU. The file is quite large (~100MB) because it contains the weights for all possible states, therefore I did not include it in the repository.


### Playing

To play against the trained agent run the following command:

```bash
python -m use_agent.py
```

This requires pygame. The controls are quite intuitive.
Can you beat the agent?
