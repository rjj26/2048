import random
import math
import time
import game_2048

#EDITED HEURISTIC: instead of optimizing for best avg score, optimize for branch with most future moves
#later squares are worth exponentially more 

# ______________________________________________________________________________
# Monte Carlo Tree Search

# 1. Selection
# Starting at root node R, recursively select optimal child nodes (explained below) until a leaf node L is reached.

# 2. Expansion
# If L is a not a terminal node (i.e. it does not end the game) then create one or more child nodes and select one C.

# 3. Simulation
# Run a simulated playout from C until a result is achieved.

# 4. Backpropagation
# Update the current move sequence with the simulation result.

# Each node must contain two important pieces of information: an estimated value based on simulation results and the number of times it has been visited.
# In its simplest and most memory efficient implementation, MCTS will add one child node per iteration. Note, however, that it may be beneficial to add more than one child node per iteration depending on the application.

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.action = None


def monte_carlo_tree_search(root_state, cpu_time, move_count):
    #if node doesnt have children (leaf or terminal state), return node
    #else recursively optimize by choosing best child option per node
    root_node = Node(root_state)

    while time.time() < cpu_time:
        node_to_expand = select(root_node)
        new_node = expand(node_to_expand)
        simulation_result = simulate(new_node.state)
        simulation_result += move_count
        backpropagate(new_node, simulation_result)

    best_child = select_best_child(root_node, exploration_weight=0)
    return best_child.action


def select(node):
    while not game_2048.is_game_over(node.state) and node.children:
        node = select_best_child(node)
    return node


#creates next level of the tree by finding all legal actions based on the successor state
    #prioritize unexplored states
def expand(node):
    if not game_2048.is_game_over(node.state) and not node.children:
        for m in game_2048.get_all_moves(node.state):
            new_state = game_2048.get_successor_state(node.state, m)
            new_node = Node(new_state, parent=node)
            new_node.action = m
            node.children.append(new_node)
    return select_best_child(node)


#exploration of child state, no idea what best option is --> record net outcome and continue
def simulate(state):
    move_count = 0
    while not game_2048.is_game_over(state):
        legal_moves = game_2048.get_all_moves(state)
        chosen_move = random.choice(legal_moves)
        state = game_2048.get_successor_state(state, chosen_move)
        move_count += 1

    return move_count


#backpropogation: simulate runs until it finds a terminal state. trace back up the tree, update current move sequence
def backpropagate(node, result):
    while node.parent is not None:
        node.visits += 1
        node.value += result
        node = node.parent
    node.visits += 1


def select_best_child(node, exploration_weight=1.0):
    if not node.children:
        return node
    
    best_child = None
    best_value = -float('inf')

    for child in node.children:
        if child.visits == 0:
            return child
        value = child.value / child.visits + exploration_weight * math.sqrt(2 * math.log(node.visits) / child.visits)
        if value > best_value:
            best_value = value
            best_child = child

    return best_child

#Upper Confidence Bound = vi + C * sqrt(ln(N) / ni) s.t. 
#N  = parent node # visits 
#ni = # visits of curr node
#vi represents exploitation
#C * ... reprsents exploration parameter of reward for checking univisited nodes

def mcts_policy(cpu_time):    
    def fxn(pos, moves, move_count):
        start_time = time.time()
        
        if game_2048.is_game_over(pos):
            return None
        
        return monte_carlo_tree_search(pos, start_time + cpu_time, move_count)
        
    return fxn

game_2048.simulate_count_moves(mcts_policy(0.25), show_board=True, show_score=True)

#need to edit simulate game to keep track of how many states have been visited

#BEST SCORE:
# SCORE:  36848
# [256, 4, 2, 4]
# [4, 1024, 128, 512]
# [2048, 64, 8, 4]
# [8, 2, 32, 2]