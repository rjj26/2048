import keras
import game_2048 as game
import numpy as np

def preprocess_state(state):
    # format game state to fit neural net input layer
    new = np.array([state])
    # convert all 0s in states to 1s for log base purposes
    new[new==0] = 1
    # log base anything of 1 = 0
    new = np.log2(new)
    new = new/16.0
    new = np.expand_dims(new, axis=3)  # Change this line

    return new

def nn_rewards_policy(game):
    def nn_score_agent(matrix, moves):
        max_score = -1
        optimal_action = ""

        for move in moves:
            successor = game.get_successor_state(matrix, move)
            # fit state to the model
            state = preprocess_state(successor)
            predicted_score = model.predict(state)
            score = np.argmax(predicted_score)

            print("in nn:", move)
            
            if score > max_score:
                max_score = score
                optimal_action = move

        print(optimal_action)
            
        return optimal_action
    return nn_score_agent


if __name__ == "__main__":
    model = keras.models.load_model("nn_reward_model_regression.keras") 
    nn_agent = nn_rewards_policy(game)

    # matrix = [
    #     [2048, 2, 4, 8],
    #     [2, 2, 128, 8],
    #     [1024, 256, 256, 512],
    #     [4, 4, 4, 4]

    # ]

    # moves = ["up", "down", "left", "right"]

    # state = preprocess_state(matrix)
    
    # score = model.predict(state)
    # print(np.argmax(score))

    game.simulate_game(nn_agent, show_board=False, show_score=True)
