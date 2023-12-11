import tensorflow
import keras
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

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

x_train, x_val, y_train, y_val = train_test_split( states, actions, test_size=0.2)

#does the one hot encoding
#numclasses is only 4 cause only 4 actions
y_train = keras.utils.to_categorical( y_train, num_classes=4)
y_val = keras.utils.to_categorical( y_val, num_classes=4)

#initialize model
model = keras.models.Sequential()

#convolutional filter, treat it as a grid image
layer1 = keras.layers.Reshape( (4,4,1), input_shape=(4,4) )
#first num = how many convolution filters in layer to train
layer2 = keras.layers.Conv2D(5000, [4, 4], activation='relu' )
layer3 = keras.layers.Flatten()
#how many perceptrons are in this layer
layer4 = keras.layers.Dense( 256, activation='relu' )
layer5 = keras.layers.Dense( 64, activation='relu' )
#output layer has 4 perceptrons to choose action, pick whatever neuron has highest activation value
layer6 = keras.layers.Dense( 4, activation='softmax' )

model.add( layer1 )
model.add( layer2 )
model.add( layer3 )
model.add( layer4 )
model.add( layer5 )
model.add( layer6 )

model.compile( optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['acc'] )

early_stopping = keras.callbacks.EarlyStopping( monitor='val_loss', patience=60 )
lr_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=20, verbose=1 )

model.summary()

model.fit( x_train, y_train, validation_data=(x_val, y_val), batch_size=200, epochs=10000, 
           verbose=1, callbacks=[early_stopping, lr_reduction] )
