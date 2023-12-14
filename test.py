# test file for comparing policies down the line 
# for now, im just manually calling things

import game_2048 as game
from random_player import random_policy
from greedy import greedy_policy
from MCTS_avgscore import mcts_avg_policy
from MCTS_maxmoves import mcts_max_policy
from MCTS_heuristics import mcts_heuristics_policy

def calc_score(scores):
    score = sum(scores) / len(scores)
    return score

def calc_tile(tiles):
    benchmark = [256, 512, 1024, 2048, 4096]
    res = {}
    length = len(tiles)

    for item in benchmark:
        count = 0
        for tile in tiles:
            if item == tile:
                count += 1
        res[item] = (count / length) * 100

    return res

def display_policy(policy, name, iterations, count=False):
    average_score = []
    tiles = []

    print("--------------------")
    print(name)
    print("--------------------")

    for _ in range(iterations):
        score, max_tile = game.simulate_game(policy, show_board=False, show_score=False) if not count else game.simulate_count_moves(policy, show_board=False, show_score=False)
        print(f"{score} (tile={max_tile})")
        average_score.append(score)
        tiles.append(max_tile)

    res_score = calc_score(average_score)
    res_tile = calc_tile(tiles)

    print(f"\nAverage Score: {res_score}")
    print(f"Tile Percentage: {res_tile}\n")

if __name__ == "__main__":
    iterations = 100
    # baseline1 = random_policy()
    # baseline2 = greedy_policy(game)
    # policy1 = mcts_avg_policy(0.05)
    # policy2 = mcts_max_policy(0.05)
    policy3 = mcts_heuristics_policy(0.05)

    # display_policy(baseline1, "random", iterations)
    # display_policy(baseline2, "greedy", iterations)
    # display_policy(policy1, "mcts average", iterations)
    # display_policy(policy2, "mcts max depth", iterations, count=True)
    display_policy(policy3, "mcts heuristics", iterations, count=True)


"""
SCORING:
-----------------------

mcts on max depth: ~1hr, 100 iterations
-> Average Score: 12354.56
-> Tile Percentage: {512: 35.0, 1024: 46.0, 2048: 12.0, 4096: 0.0}
-> Max Score: 32488 (tile=2048)
"""
