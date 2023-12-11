import game_2048

def greedy_policy(game):
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
            score = game.game_score(successor)
            
            if score > max_score:
                max_score = score
                optimal_action = move

        return optimal_action
    
    return greedy_player

if __name__ == "__main__":
    greedy = greedy_policy(game_2048)
    game_2048.simulate_game(greedy, show_board=True, show_score=True)
