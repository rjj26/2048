# test file for comparing policies down the line 
# for now, im just manually calling things

import game_2048 as game
from random_player import random_policy
from greedy import greedy_policy
from MCTS_avgscore import mcts_avg_policy
from MCTS_maxmoves import mcts_max_policy

if __name__ == "__main__":
    iterations = 10
    baseline = greedy_policy(game)
    policy1 = mcts_avg_policy(0.05)
    policy2 = mcts_max_policy(0.05)
    average_score = []

    # baseline greedy agent approach
    print("--------------------")
    print("Greedy Agent")
    print("--------------------")

    for i in range(iterations):
        score = game.simulate_game(baseline, show_board=False, show_score=False)
        average_score.append(score)
        print(score)

    res = sum(average_score) / len(average_score)

    print(f"Average Score: {res}\n")

    print("--------------------")
    print("MCTS Avg Score")
    print("--------------------")

    average_score.clear()

    for i in range(iterations):
        score = game.simulate_game(policy1, show_board=False, show_score=False)
        average_score.append(score)
        print(score)

    res = sum(average_score) / len(average_score)

    print(f"Average Score: {res}\n")

    print("--------------------")
    print("MCTS Max Moves")
    print("--------------------")

    average_score.clear()

    for i in range(iterations):
        score = game.simulate_count_moves(policy2, show_board=False, show_score=False)
        average_score.append(score)
        print(score)

    res = sum(average_score) / len(average_score)

    print(f"Average Score: {res}\n")
