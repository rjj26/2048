import numpy as np
import game_2048 as game
from mcts import mcts_avg_policy
from mcts_heuristics import mcts_heuristics_policy

""" MCTS testing script to compare all mcts agents

    group members: Reese Johnson '24 and Vinh Tran '25

    game description: https://play2048.co/ (we implemented our own version to be able to take in our policies)
        - single-player game on a 4x4 grid
        - on each turn: the user decides how to slide the tiles and a 2 or 4 tile spawns randomly on an open space
        - scoring is based on merging tiles with the same value, which accumulates 
        - the objective is to obtain (or pass) the 2048 tile 
        - the game terminates when there are no free spaces left and no tiles can be merged

    code description:
        - mcts standard: uses score to determine optimal move
        - mcts heuristics: uses a variety of heuristics to determine move that results in best state 
        - mcts cnn: uses model to predict optimal move to play during simulation phase

    how to run: python3 test_mcts.py in terminal to see performance of all mcts  agents
        - may need to do pip3 install requirements.txt if you local computer doesn't have the packages

RESULTS: on 5 iterations because anythin past 0.5 takes a very, very long time (each game last 100s of moves), feel free to change iterations
---------------------------------------------------------------------------------------------------------------------------------------------------
| AGENT                                     | AVG SCORE (& MAX SCORE) | TILE DISTRIBUTION                                               | STD DEV |
---------------------------------------------------------------------------------------------------------------------------------------------------
| mcts (standard): 0.05                     | 6689.96 (16474)         | { 128: 2.0, 256: 24.0, 512: 55.0, 1024: 19.0 }                  | 3193.49 |
---------------------------------------------------------------------------------------------------------------------------------------------------
| mcts w/ heuristics:                       | 14191.0 (34380)         | { 256: 8.0, 512: 22.0, 1024: 51.0, 2048: 19.0 }                 | 7399.27 |
---------------------------------------------------------------------------------------------------------------------------------------------------
| mcts w/ neural network (determines move): |
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

    print(f"\nAverage Score: {res_score}")
    print(f"Max Score: {res_max_score}")
    print(f"Score Std Dev: {res_score_var}")
    print(f"Tile Distribution: {res_tile}\n")

if __name__ == "__main__":
    iterations = 5
    mcts_time1 = 0.05
    mcts_time2 = 0.5
    mcts_time3 = 1.0

    policy_mcts1 = mcts_avg_policy(mcts_time1)
    policy_mcts_enhanced1 = mcts_heuristics_policy(mcts_time1, [0.0025, 0.003, 0.003, 0.004, 0, 1.0])

    policy_mcts2 = mcts_avg_policy(mcts_time2)
    policy_mcts_enhanced2 = mcts_heuristics_policy(mcts_time2, [0.0025, 0.003, 0.003, 0.004, 0, 1.0])

    policy_mcts3 = mcts_avg_policy(mcts_time3)
    policy_mcts_enhanced3 = mcts_heuristics_policy(mcts_time3, [0.0025, 0.003, 0.003, 0.004, 0, 1.0])


    display_policy(policy_mcts1, "mcts, time=0.05", iterations)
    display_policy(policy_mcts_enhanced1, "heuristics, time=0.05", iterations, count=True)

    display_policy(policy_mcts2, "mcts, time=0.5", iterations)
    display_policy(policy_mcts_enhanced2, "heuristics, time=0.5", iterations, count=True)

    display_policy(policy_mcts3, "mcts, time=1.0", iterations)
    display_policy(policy_mcts_enhanced3, "heuristics, time=1.0", iterations, count=True)
