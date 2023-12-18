import keras
import game_2048
import numpy as np

NN_classificiation_model = keras.models.load_model("sl_models/Supervised_classification.keras")

def NN_policy(matrix, moves):

    #format game state to fit neural net input layer
    state = np.array([matrix])
    #convert all 0s in states to 1s for log base purposes
    state[state==0] = 1
    #log base anything of 1 = 0
    state = np.log2(state)
    state = state/16.0
    
    scores = NN_classificiation_model.predict(state, verbose=0)

    move_rankings = np.argsort(-scores[0,:])
    move_names = ["up", "down", "left", "right"]

    for move in move_rankings:
        if move_names[move] in moves:
            return move_names[move]
        

if __name__ == "__main__":

    # fake_data = np.random.rand(1,4,4)
    # print(model.predict(fake_data))
    #gives normalized scores [up, down, left, right]
        
    game_2048.simulate_game(NN_policy, show_board=True, show_score=True)