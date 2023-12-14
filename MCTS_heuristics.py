import math
import random
import time
import game_2048 as game

###############################################################################
# I am using the following heuristrics
# monotonicity
# smoothness
# game score
# movecount
###############################################################################

class Node:
    def __init__(self, state):
        self.state = state
        self.reward = 0
        self.visits = 0
        self.children = {}
        self.parent = None
        self.isFullyExpanded = game.is_game_over(self.state)
        
        
class MCTS_HEURISTICS:
    def __init__(self):
        pass

    def search(self, position, input_time, move_count):
        """ actual search function
            traverse -> expand -> simulate -> backpropogate
        """

        self.root = Node(position)
        start_time = time.time()
        best_action = None
        
        while time.time() - start_time < input_time: # go until time runs out
            node = self.traverse() 
            reward = self.simulate(node) + move_count
            self.back_propagation(node, reward)
    
        best_node = self.ucb(self.root, 0) # find best node in the populated tree

        # find best node in dictionary of children and return
        for action, node in self.root.children.items():
            if node is best_node:
                best_action = action
        
        return best_action

    def traverse(self, node=None):
        """ traverse tree by expanding children nodes
        """

        if node is None:
            node = self.root

        while not game.is_game_over(node.state):
            if not node.isFullyExpanded:
                return self.expand(node)

            node = self.ucb(node) # UCB1 selection strategy

        return node

    def expand(self, node):
        """ expand by creating the child node
        """

        if game.is_game_over(node.state):
            return

        legal_actions = game.get_all_moves(node.state)
        for action in legal_actions:
            if action not in node.children:
                child_state = game.get_successor_state(node.state, action)
                child_node = Node(child_state)
                child_node.parent = node
                node.children[action] = child_node

                if len(node.children) == len(legal_actions):
                    node.isFullyExpanded = True

                return child_node

    def simulate(self, node):
        """ pick a random child and simulate the action
        """

        current_state = node.state
        num_moves = 0
        while not game.is_game_over(current_state):
            num_moves += 1
            legal_moves = game.get_all_moves(current_state)
            
            # instead of random, we use heuristics to select a move
            # action = self.select_move_with_heuristics(current_state, legal_moves, num_moves)
            action = random.choice(legal_moves)
            current_state = game.get_successor_state(current_state, action)
        
        # evaluate final score using the heuristics
        score = self.evaluate_state(current_state, num_moves)
        return score

        # return game.game_score(current_state)

    def back_propagation(self, node, reward):
        """ propgate values back to root
        """

        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent

    def ucb(self, node, constant=1):
        """ UCB1 function from class
            exploration weight: sqrt(2)
        """

        bestNode = []
        bestAction = float('-inf')
        
        exploration_weight = math.sqrt(2)
        n_visits = math.log(node.visits)

        for child in node.children.values():
            exploitation = child.reward / child.visits
            exploration = exploration_weight * math.sqrt(n_visits / child.visits)

            temp = exploitation + constant * exploration
            if temp > bestAction:
                bestNode = [child]
                bestAction = temp
            elif temp == bestAction:
                bestNode.append(child)
                bestAction = temp
                    
        # store in array and randomly choose in case of ties
        return random.choice(bestNode) 
    
    ################################################################
    # Start of Heursitics
    ################################################################

    def monotonicity_penalty(self, state):
        """Count monotonicity penalty: monotonicity are trends like increasing or decreasing
        Input: state
        Output: score
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

    def merges(self, state):
        """ Count merges heuristic: number of potential merges and open spaces
            Input: state
            Output: score
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
    
    def open_tiles(self, state):
        """ Count open tiles heuristic: number of free spaces
            Input: state
            Output: score
        """

        open_spaces = 0

        for row in state:
            open_spaces += row.count(0)

        for col in zip(*state):
            open_spaces += col.count(0)

        return open_spaces

    def smoothness(self, state):
        """ Calculate smoothness heuristic: tiles closer in value that are closer to each other
            Input: current state
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

    def select_move_with_heuristics(self, state, legal_moves, move_count):
        best_move = None
        best_score = float('-inf')

        for move in legal_moves:
            successor_state = game.get_successor_state(state, move)
            score = self.evaluate_state(successor_state, move_count)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move
    
    def evaluate_state(self, state, moves):
        # Adjust the weights based on experimentation
        weight_monotonicity = 0.025
        weight_merges = 0.1
        weight_open_spaces = 0.1
        weight_smoothness = 0.075
        weight_game_score = 0.075
        weight_moves = 5.0

        monotonicity = self.monotonicity_penalty(state)
        merges = self.merges(state)
        open_spaces = self.open_tiles(state)
        smoothness = self.smoothness(state)
        game_score = game.game_score(state)

        # Combine the scores with weights
        total_score = (
            weight_monotonicity *  monotonicity +
            weight_merges * merges +
            weight_open_spaces * open_spaces +
            weight_smoothness * smoothness +
            weight_game_score * game_score +
            weight_moves * moves
        )

        # print(total_score, monotonicity, merges, open_spaces, smoothness, game_score, moves)
    
        return total_score
    
    ################################################################
    # End of Heursitics
    ################################################################

def mcts_heuristics_policy (input_time):
    def fxn (position, moves, move_count): 
        m = MCTS_HEURISTICS()
        best_action = m.search(position, input_time, move_count)

        if game.is_game_over(position):
            return None

        return best_action
    
    return fxn

if __name__ == "__main__":
    game.simulate_count_moves(mcts_heuristics_policy(0.05), show_board=True, show_score=True)
    # game.simulate_game(mcts_heuristics_policy(0.05), show_board=True, show_score=True)


"""
--------------------------
Iteration: 94
Best weights found: {'mono': 0.48, 'merge': 1.02, 'open': 1.01, 'smooth': 0.33, 'score/tile': 1.38, 'num moves': 0.99}
Best score found: 27036
Best tile found: 2048
"""