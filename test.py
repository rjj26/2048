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

def calc_var(arr):
    # Step 1: Calculate the mean
    mean = sum(arr) / len(arr)

    # Step 2: Calculate deviations
    deviations = [x - mean for x in arr]

    # Step 3: Square each deviation
    squared_deviations = [x**2 for x in deviations]

    # Step 4: Calculate the mean of squared deviations
    variance = sum(squared_deviations) / len(squared_deviations)

    return variance

def display_policy(policy, name, iterations, count=False):
    average_score = []
    tiles = []

    print("--------------------")
    print(name)
    print("--------------------")

    for i in range(iterations):
        score, max_tile = game.simulate_game(policy, show_board=False, show_score=False) if not count else game.simulate_count_moves(policy, show_board=False, show_score=False)
       
        if max_tile >= 2048:
            print(f"{i}: {score} (tile={max_tile})**")
        else:
            print(f"{i}: {score} (tile={max_tile})")
        average_score.append(score)
        tiles.append(max_tile)

    res_score = calc_score(average_score)
    res_tile = calc_tile(tiles)
    res_score_var = calc_var(average_score)
    res_tile_var = calc_var(tiles)

    print(f"\nAverage Score: {res_score}")
    print(f"Tile Percentage: {res_tile}\n")
    print(f"Score Variance: {res_score_var}\n")
    print(f"Max Tile Variance: {res_tile_var}\n")

if __name__ == "__main__":
    iterations = 10
    # baseline1 = random_policy()
    # baseline2 = greedy_policy(game)
    # policy1 = mcts_avg_policy(0.05)
    # policy2 = mcts_max_policy(0.05)
    policy3 = mcts_heuristics_policy(0.05, [0.001, 0, 0, 0.001, 0, 1.0])
    policy4 = mcts_heuristics_policy(0.05, [0.0025, 0.003, 0.003, 0.004, 0, 1.0])
    #  policy6 = mcts_heuristics_policy(0.05, [0.0, 0, 0, 0.0, 0, 1.0])

    # display_policy(baseline1, "random", iterations)
    # display_policy(baseline2, "greedy", iterations)
    # display_policy(policy1, "mcts average", iterations)
    # display_policy(policy2, "mcts max depth", iterations, count=True)
    display_policy(policy3, "mcts heuristics1", iterations, count=True)
    display_policy(policy4, "mcts heuristics2", iterations, count=True)
    # display_policy(policy6, "mcts heuristics4", iterations, count=True)


"""
SCORING:
-----------------------

mcts on max depth: ~1hr, 100 iterations
-> Average Score: 12354.56
-> Tile Percentage: {512: 35.0, 1024: 46.0, 2048: 12.0, 4096: 0.0}
-> Max Score: 32488 (tile=2048)

mcts with heuristics (0.001, 0, 0, 0.001, 0, 1.0): 100 iterations
-> Average Score: 14657.72                                         
-> Tile Percentage: {256: 3.0, 512: 27.0, 1024: 53.0, 2048: 17.0, 4096: 0.0} 
-> Max Score: 35192 (tile=2048)                                                        
-> Score Variance: 54417994.32159997                               
-> Max Tile Variance: 266567.6799999999  

mcts with heuristics (0.0025, 0.003, 0.003, 0.004, 0, 1.0): 100 iterations
-> Average Score: 14191.0
-> Tile Percentage: {256: 8.0, 512: 22.0, 1024: 51.0, 2048: 19.0, 4096: 0.0}
-> Max Score: 34380 (tile=2048)** 
-> Score Variance: 54835496.12
-> Max Tile Variance: 303667.60959999973

"""
