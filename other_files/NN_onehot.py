import tensorflow
import keras
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

###Python Data from our own MCTS implementation
def load_python_data():
    with open("training_data.pkl", "rb") as f:

        training_data = pickle.load(f)

    states = []
    actions = []

    for sa_pair in training_data:
        state, action = sa_pair

        if action == "up":
            action = 0
        if action == "down":
            action = 1
        if action == "left":
            action = 2
        if action == "right":
            action = 3

        actions.append(action)
        state = np.array(state)

        #convert all 0s in states to 1s for log base purposes
        state[state==0] = 1
        #log base anything of 1 = 0
        state = np.log2(state)
        #simplifies the values (normalization), compresses the numbers on the board
        states.append(state)

    states = np.array(states)/16.0
    actions = np.array(actions)

    return states, actions

### C++ data used from running this established expectimax implementation: https://github.com/nneonneo/2048-ai.git
###Using supervised learning method to record this algorithms state action pairs to train NN
def load_cpp_data(num_data_points=None):
    with open("supervised_training.txt", "r") as f:

        training_data = f.readlines()

    states = []
    actions = []
    if num_data_points is None:
      num_data_points = len(training_data)

    for i in range(num_data_points):
        line = training_data[i]
        values = line.strip().split()


        if len(values) == 0  or "Game" in values[0]:
            #game is over
            continue

        action = int(values[16])
        state = [[int(values[0]), int(values[1]), int(values[2]), int(values[3])],
                 [int(values[4]), int(values[5]), int(values[6]), int(values[7])],
                 [int(values[8]), int(values[9]), int(values[10]), int(values[11])],
                 [int(values[12]), int(values[13]), int(values[14]), int(values[15])]]

        actions.append(action)
        state = np.array(state)

        #convert all 0s in states to 1s for log base purposes
        state[state==0] = 1
        #log base anything of 1 = 0
        state = np.log2(state).astype('int')
        state_one_hot = np.zeros((4, 4, 17))
        for i in range(4):
          for j in range(4):
            state_one_hot[i][j][state[i][j]] = 1
        #simplifies the values (normalization), compresses the numbers on the board
        states.append(state_one_hot)

    # states = np.array(states)/16.0
    return np.array(states), np.array(actions)

states, actions = load_cpp_data(500000)
# states, actions = load_python_data()

print(len(states), states[0].shape, len(actions), actions[0].shape)

x_train, x_val, y_train, y_val = train_test_split( states, actions, test_size=0.2)

#does the one hot encoding
#numclasses is only 4 cause only 4 actions
y_train = keras.utils.to_categorical( y_train, num_classes=4)
y_val = keras.utils.to_categorical( y_val, num_classes=4)

#initialize model
# model = keras.models.Sequential()

#convolutional filter, treat it as a grid image
# layer1 = keras.layers.Reshape( (4,4,1), input_shape=(4,4) )
#first num = how many convolution filters in layer to train
input = keras.Input(shape=(4,4, 17))

# "conv block"
conv2x2 = keras.layers.Conv2D(64, [2, 2], activation='relu', padding='same')(input)
conv3x3 = keras.layers.Conv2D(64, [3, 3], activation='relu', padding='same')(input)
feature_vector = keras.layers.Concatenate(axis=3)([conv2x2, conv3x3])

flattened_features = keras.layers.Flatten()(feature_vector)
linear1 = keras.layers.Dense(512, activation='sigmoid')(flattened_features)
linear1_dropout = keras.layers.Dropout(rate=0.5)(linear1)
output = keras.layers.Dense(4, activation='softmax')(linear1_dropout)

model = keras.Model(inputs=input, outputs=output)
#how many perceptrons are in this layer

#output layer has 4 perceptrons to choose action, pick whatever neuron has highest activation value
# layer7 = keras.layers.Dense( 4, activation='softmax' )

model.compile( optimizer=keras.optimizers.Adam(learning_rate=0.01, weight_decay=0.0005), loss='categorical_crossentropy', metrics=['acc'] )

# early_stopping = keras.callbacks.EarlyStopping( monitor='val_loss', patience=20 )
# lr_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=5, verbose=1 )

model.summary()

model.fit( x_train, y_train, validation_data=(x_val, y_val), batch_size=250, epochs=100,
           verbose=1)

model.save("Supervised_classification8.keras")
