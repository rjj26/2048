# test file for comparing policies down the line 
# for now, im just manually calling things

import game_2048 as game
from random_player import random_policy
from greedy import greedy_policy
from MCTS_avgscore import mcts_avg_policy
from MCTS_maxmoves import mcts_max_policy

def display_policy(policy, name, iterations, count=False):
    average_score = []

    print("--------------------")
    print(name)
    print("--------------------")

    for _ in range(iterations):
        score, max_tile = game.simulate_game(policy, show_board=False, show_score=False) if not count else game.simulate_count_moves(policy, show_board=False, show_score=False)
        print(f"{score} (tile={max_tile})")
        average_score.append(score)

    res = sum(average_score) / len(average_score)
    print(f"\nAverage Score: {res}\n")

if __name__ == "__main__":
    iterations = 2
    baseline1 = random_policy()
    baseline2 = greedy_policy(game)
    policy1 = mcts_avg_policy(0.05)
    policy2 = mcts_max_policy(0.05)

    display_policy(baseline1, "greedy", iterations)
    display_policy(baseline2, "greedy", iterations)
    display_policy(policy1, "mcts average", iterations)
    display_policy(policy2, "mcts max depth", iterations, count=True)
