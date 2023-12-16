import numpy as np
import game_2048 as game
from random_agent import random_policy
from greedy_agent import greedy_policy
from mcts import mcts_avg_policy
from mcts_heuristics import mcts_heuristics_policy

""" testing script to compare all agents

    group members: Reese Johnson '24 and Vinh Tran '25

    game description: https://play2048.co/ (we implemented our own version to be able to take in our policies)
        - single-player game on a 4x4 grid
        - on each turn: the user decides how to slide the tiles and a 2 or 4 tile spawns randomly on an open space
        - scoring is based on merging tiles with the same value, which accumulates 
        - the objective is to obtain (or pass) the 2048 tile 
        - the game terminates when there are no free spaces left and no tiles can be merged

    code description:
        - we programmed various agents to try to achieve the 2048 
        - our primary agents are 
            (1) mcts (we set time per move at 0.05 for all mcts agents)
            (2) supervised learning model (CNN) that uses state action pairs
        - our results are stated below (on 100 iterations of the game):

    how to run: python3 test.py in terminal to see performance of all agents
        - may need to do pip3 install requirements.txt if you local computer doesn't have the packages

---------------------------------------------------------------------------------------------------------------------------------------------------
| AGENT                                     | AVG SCORE (& MAX SCORE) | TILE DISTRIBUTION                                               | STD DEV |
---------------------------------------------------------------------------------------------------------------------------------------------------
| random moves (baseline):                  | 938.92 (2916)           | { 64: 41.0, 128: 40.0, 256: 3.0 }                               | 458.04  |
---------------------------------------------------------------------------------------------------------------------------------------------------
| greedy (baseline):                        | 2019.32 (7096)          | { 64: 13.0, 128: 44.0, 256: 42.0, 512: 1.0 }                    | 965.33  |
---------------------------------------------------------------------------------------------------------------------------------------------------
| greedy w/ heuristics (baseline):          | 8938.76 (24928)         | { 128: 2.0, 256: 14.0, 512: 56.0, 1024: 26.0, 2048: 2.0 }       | 4291.32 |
---------------------------------------------------------------------------------------------------------------------------------------------------
| mcts (standard):                          | 6689.96 (16474)         | { 128: 2.0, 256: 24.0, 512: 55.0, 1024: 19.0 }                  | 3193.49 |
---------------------------------------------------------------------------------------------------------------------------------------------------
| mcts w/ heuristics:                       | 14191.0 (34380)         | { 256: 8.0, 512: 22.0, 1024: 51.0, 2048: 19.0 }                 | 7399.27 |
---------------------------------------------------------------------------------------------------------------------------------------------------
| mcts w/ neural network (determines move): |
---------------------------------------------------------------------------------------------------------------------------------------------------
| SL neural network model (classification): |
---------------------------------------------------------------------------------------------------------------------------------------------------

* a more descriptive description/analysis will be provided on our READme file if interested

"""

def calc_score(scores):
    avg = round(sum(scores) / len(scores), 2)
    max_score = round(max(scores), 2)

    return avg, max_score

def calc_tile(tiles):
    benchmark = [64, 128, 256, 512, 1024, 2048, 4096]
    res = {}
    length = len(tiles)

    for item in benchmark:
        count = 0
        for tile in tiles:
            if item == tile:
                count += 1
        res[item] = round((count / length) * 100, 1)

    return res

def calc_stddev(arr):
    std_dev = round(np.std(arr), 2)
    return std_dev

def display_policy(policy, name, iterations, count=False):
    average_score = []
    tiles = []

    print("--------------------")
    print(name)
    print("--------------------")

    for i in range(iterations):
        if not count:
            score, max_tile = game.simulate_game(policy, show_board=False, show_score=False)
        else:
            score, max_tile = game.simulate_count_moves(policy, show_board=False, show_score=False)
       
        # printing each iteration
        # if max_tile >= 2048:
        #     print(f"{i}: {score} (tile={max_tile})**")
        # else:
        #     print(f"{i}: {score} (tile={max_tile})")

        average_score.append(score)
        tiles.append(max_tile)

    res_score, res_max_score = calc_score(average_score)
    res_tile = calc_tile(tiles)
    res_score_var = calc_stddev(average_score)

    print(f"\n\nAverage Score: {res_score}")
    print(f"Max Score: {res_max_score}")
    print(f"Score Std Dev: {res_score_var}")
    print(f"Tile Distribution: {res_tile}\n")

if __name__ == "__main__":
    iterations = 100
    mcts_time = 0.05

    # instantiate objects of policy
    baseline_random = random_policy()
    baseline_greedy = greedy_policy(game, heuristics=False)
    baseline_greedy_heuristics = greedy_policy(game, heuristics=True)
    policy_mcts = mcts_avg_policy(mcts_time)
    policy_mcts_enhanced = mcts_heuristics_policy(mcts_time, [0.0025, 0.003, 0.003, 0.004, 0, 1.0])

    # display policy results in terminal
    # display_policy(baseline_random, "random agent", iterations)
    # display_policy(baseline_greedy, "greedy agent", iterations)
    # display_policy(baseline_greedy_heuristics, "greedy (heuristics) agent", iterations)
    display_policy(policy_mcts, "mcts standard", iterations)
    display_policy(policy_mcts_enhanced, "mcts heuristics", iterations, count=True)
