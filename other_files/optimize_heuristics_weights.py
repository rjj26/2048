import random
import sys
sys.path.append('../')
import game_2048 as game
from mcts_heuristics import mcts_heuristics_policy

# global var define optimal ranges to test
ranges = [(0.0, 0.005), (0.0, 0.007), (0.0, 0.008), (0.0, 0.008), (0.0, 0.0), (1.0, 1.0)]

def random_search(iterations, top_n):
    best_weights = []
    best_scores = []
    best_tiles = []
    best_iterations = []

    for i in range(iterations):
        current_weights = [round(random.uniform(low, high), 4) for low, high in ranges]

        policy_function = mcts_heuristics_policy(0.05, current_weights)
        average_score, tile = game.simulate_count_moves(policy_function)

        weight_dict = {
            "mono": current_weights[0],
            "merge": current_weights[1],
            "open": current_weights[2],
            "smooth": current_weights[3],
            "score/tile": current_weights[4],
            "num moves": current_weights[5]
        }

        print("--------------------------")
        print("Iteration:", i + 1)
        print("Weights:", weight_dict)
        print("Score:", average_score)
        print("Tile:", tile)

        if len(best_weights) < top_n or average_score > min(best_scores):
            best_weights.append(current_weights)
            best_scores.append(average_score)
            best_tiles.append(tile)
            best_iterations.append(i + 1)

            # Sort the lists based on scores in descending order
            sorted_indices = sorted(range(len(best_scores)), key=lambda k: best_scores[k], reverse=True)
            best_weights = [best_weights[idx] for idx in sorted_indices]
            best_scores = [best_scores[idx] for idx in sorted_indices]
            best_tiles = [best_tiles[idx] for idx in sorted_indices]
            best_iterations = [best_iterations[idx] for idx in sorted_indices]

    return best_weights, best_scores, best_tiles, best_iterations

if __name__ == "__main__":
    num_iterations = 50
    top_n = 5

    # perform random search for weights
    best_weights_found, best_score_found, best_tile_found, best_iter_found = random_search(num_iterations, top_n)
    print("\n\n BEST VALUES\n")

    for i in range(top_n):
        weight_dict = {
            "mono": best_weights_found[i][0],
            "merge": best_weights_found[i][1],
            "open": best_weights_found[i][2],
            "smooth": best_weights_found[i][3],
            "score/tile": best_weights_found[i][4],
            "num moves": best_weights_found[i][5]
        }
    
        print("--------------------------")
        print("Iteration:", best_iter_found[i])
        print("Best weights found:", weight_dict)
        print("Best score found:", best_score_found[i])
        print("Best tile found:", best_tile_found[i])

"""
Best weights found: {'mono': 2.008, 'merge/open': 1.053, 'smooth': 0.43, 'score/tile': 1.837, 'num moves': 1.507}
Best score found: 23948                                             
Best tile found: 2048                                               
                                                     
Best weights found: {'mono': 1.867, 'merge/open': 1.54, 'smooth': 0.915, 'score/tile': 1.935, 'num moves': 1.176}
Best score found: 23424                                             
Best tile found: 2048                                               
                                                  
Best weights found: {'mono': 1.378, 'merge/open': 1.132, 'smooth': 0.852, 'score/tile': 1.217, 'num moves': 0.971}
Best score found: 15980                                             
Best tile found: 1024  

Best weights found: {'mono': 1.26, 'merge/open': 1.03, 'smooth': 0.24, 'score/tile': 2.06, 'num moves': 1.59}
Best score found: 15788
Best tile found: 1024

Best weights found: {'mono': 1.336, 'merge/open': 0.668, 'smooth': 0.882, 'score/tile': 1.724, 'num moves': 0.766}
Best score found: 15696                                             
Best tile found: 1024   

Best weights found: {'mono': 1.699, 'merge/open': 0.749, 'smooth': 1.151, 'score/tile': 1.743, 'num moves': 1.381}
Best score found: 14748                                             
Best tile found: 1024 

Best weights found: {'mono': 2.032, 'merge/open': 0.653, 'smooth': 0.847, 'score/tile': 1.139, 'num moves': 1.214}
Best score found: 14564
Best tile found: 1024

v1: ranges = [(1.2, 2.1), (0.6, 1.7), (0.1, 1.4), (0.8, 2.1), (0.7, 1.9)]
v2: ranges = [(1.450, 2.055), (0.690, 1.350), (0.430, 1.160), (1.450, 2.070), (0.850, 1.600)]
--------------------------------------------------------------------------------------------------------------------------------
"""
