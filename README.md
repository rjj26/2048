# Final Project CPSC474: 2048 Agent
**Objective: Beat 2048 using MCTS and Supervised Learning using a Neural Network**

**By Reese Johnson and Vinh Tran**

## Game Description: https://play2048.co/
(we implemented our own version to be able to take in our policies)

### Rules:
- single-player game on a 4x4 grid
- on each turn: the user decides how to slide the tiles and a 2 or 4 tile spawns randomly on an open space
- scoring is based on merging tiles with the same value, which accumulates 
- the objective is to obtain (or pass) the 2048 tile 
- the game terminates when there are no free spaces left and no tiles can be merged

## Code Description:
- we programmed various agents to try to achieve the 2048 tile (beat the game)
- we programmed 3 baseline agents (random, greedy, greedy w/ heuristics to compare to our optimal agents (MCTS and SL)

### Baseline Agents 
1. **Random:** makes random moves that are legal; this strategy actually performs quite well for a game like 2048
2. **Greedy Standard:** on every turn, agent seeks to maximize score
3. **Greedy Heuristics:** on every turn, agent chooses the move that will result in the best state according to the heuristics (mentioned below)

### MCTS
1. **Standard Implementation:** looks for best reward i.e. score
2. **Enhanced with Heuristics**
    - this uses various heuristics in order to determine the optimal state instead of just basing off the score reward because position matters a lot in a gmae like 2048
    - we used the following heurisitcs:
      1. monotonicity - measures increasing/decreasing tiles in one direction, penalty for non-monotonic boards
      2. smoothness - measures tiles of similar value that are adjacent, penalty on non-smooth boards
      3. merge - measures if adjacent tiles are equal so that they can be merged
      4. open tiles - measures number of open tiles on board
      5. move count - biasing agent to go for branch with most moves ~ equivalent to trying to obtain highest score
   
### Supervised Learning Convolutional Neural Network
note: we decided to train our CNN using expert data from one of the highest performing 2048 agents publically available

credit: C++ data used from running this established expectimax implementation: https://github.com/nneonneo/2048-ai.git 

1. **Classification Model**
    - Input: State
    - Output: Optimal (Predicted) Move/Action
      
2. **MCTS Enchanced by Classification model**
    - In simulation stage, instead of choosing a random move from the legal moves, we use the model to predict the optimal move instead
  
### Other Agents (Not Submitted)

1.**Regression Model**
   - SL model using Regression Convolutional Neural Network: doesn't perform well enough (time to train efficiently out of scope of project) but it is a cool idea
   - Input: State
   - Output: Predicted Score (not literal score, but the potential value of the current state ~ similar to heuristics)
   - Potential: Use similar to greedy agent or incorporate into an MCTS

### Test Results

**Specifications:** 100 iterations of the game, time allowed for mcts agents was 0.05s

**How to Run:** `python3 test.py` to see all our agents' performances


| AGENT | AVG SCORE (& MAX) | TILE DISTRIBUTION | STD DEV |
| --- | --- | --- | --- |
| **random moves** | 938.92 (2916) | { 64: 41.0, 128: 40.0, 256: 3.0 } |  458.04 |
| **greedy** | 2019.32 (7096) | { 64: 13.0, 128: 44.0, 256: 42.0, 512: 1.0 } | 965.33 |
| **greedy w/ heuristics** | 8938.76 (24928) | { 128: 2.0, 256: 14.0, 512: 56.0, 1024: 26.0, 2048: 2.0 } | 4291.32 |
| **mcts (standard)** | 6689.96 (16474) | { 128: 2.0, 256: 24.0, 512: 55.0, 1024: 19.0 } | 3193.49 |
| **mcts w/ heuristics** | 14191.0 (34380) | { 256: 8.0, 512: 22.0, 1024: 51.0, 2048: 19.0 } | 7399.27 |
| **mcts w/ neural network** | | | |
| **SL classification** | | | |


### Folder Structure:
- `\logs`: contains log files for what each member did for work for the project
- `\other_files`: contains files that were used to help tune heuristics, train models, initial versions of agents, etc.
- `\sl_models`: contains models for the supervised learning models
- `\sl_training_data`: contains a sample file of what we used to train our models (the real training model was too large)
- `\game_2048.py`: 2048 game that simulates game with given policy
- `\test.py`: our testing script that evaluates performance of the agents

