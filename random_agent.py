import random
import game_2048 as game

def random_policy():
    """ baseline agent:
        random agent just makes random moves at every turn
    """
            
    def random_player(matrix, moves):
        return random.choice(moves)
    
    return random_player
    
if __name__ == "__main__":
    random_agent = random_policy()
    game.simulate_game(random_agent, show_board=True, show_score=True)
