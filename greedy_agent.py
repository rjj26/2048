import game_2048 as game
import math

################################################################
# Start of Heursitics
################################################################

def monotonicity_penalty(state):
    """ Count monotonicity penalty: 
        monotonicity are trends like increasing or decreasing
    """
    def calculate_sequence_monotonicity(sequence):
        monotonicity_left = 0
        monotonicity_right = 0

        for i in range(1, len(sequence)):
            if sequence[i - 1] > sequence[i]:
                monotonicity_left += sequence[i - 1] - sequence[i]
            else:
                monotonicity_right += sequence[i] - sequence[i - 1]

        return min(monotonicity_left, monotonicity_right)

    penalty = 0
    for row in state:
        penalty += calculate_sequence_monotonicity(row)

    for col in zip(*state):
        penalty += calculate_sequence_monotonicity(col)

    return -penalty  # negative for a penalty

def merges(state):
    """ Count merges heuristic: 
        number of potential merges and open spaces
    """
    merges = 0
    # open_spaces = 0
    for row in state:
        for i in range(len(row) - 1):
            if row[i] == row[i + 1]:
                merges += 1

    for col in zip(*state):
        for i in range(len(col) - 1):
            if col[i] == col[i + 1]:
                merges += 1

    return merges

def open_tiles(state):
    """ Count open tiles heuristic: number of free spaces
    """

    open_spaces = 0

    for row in state:
        open_spaces += row.count(0)

    for col in zip(*state):
        open_spaces += col.count(0)

    return open_spaces

def smoothness(state):
    """ Calculate smoothness heuristic: tiles closer in value that are closer to each other
        Output: smoothness heursitic score (as penatly if not smooth)
    """
            
    smoothness = 0
    for row in state:
        for i in range(len(row) - 1):
            if row[i] != 0 and row[i + 1] != 0:
                smoothness -= abs(math.log2(row[i]) - math.log2(row[i + 1]))

    for col in zip(*state):
        for i in range(len(col) - 1):
            if col[i] != 0 and col[i + 1] != 0:
                smoothness -= abs(math.log2(col[i]) - math.log2(col[i + 1]))
    return smoothness        

def evaluate_state(state, heuristics):
    # Adjust the weights based on experimentation
    weight_monotonicity, weight_merges, weight_open_spaces, weight_smoothness, weight_game_score = 0, 0, 0, 0, 0

    if heuristics:
        weight_monotonicity = 0.3
        weight_merges = 1.0
        weight_open_spaces = 1.0
        weight_smoothness = 0.5

    weight_game_score = 1.0

    monotonicity = monotonicity_penalty(state) * weight_monotonicity
    merge_score = merges(state) * weight_merges
    open_spaces = open_tiles(state) * weight_open_spaces
    smooth_penalty = smoothness(state) * weight_smoothness
    game_score = game.game_score(state) * weight_game_score

    # Combine the scores with weights
    total_score = monotonicity + merge_score + open_spaces + smooth_penalty + game_score

    return total_score

def greedy_policy(game, heuristics=False):
    """ baseline agent:
        greedy policy seeks to maximize score at every move
        by always picking the move that will increase the score 
        the most at any given game state
    """
     
    def greedy_player(matrix, moves):
        max_score = -1
        optimal_action = ""

        for move in moves:
            successor = game.get_successor_state(matrix, move)
            # score = game.game_score(successor)

            score = evaluate_state(successor, heuristics)
            
            if score > max_score:
                max_score = score
                optimal_action = move

        return optimal_action
    
    return greedy_player

if __name__ == "__main__":
    greedy = greedy_policy(game, heuristics=True)
    game.simulate_game(greedy, show_board=True, show_score=True)
