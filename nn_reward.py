import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load data
def load_data():
    with open("/Users/vinhtran/Desktop/junior/fall/CS474/finalproj/sl_training_data/training_data_reward.txt", "r") as f:
        training_data = f.readlines()

    states = []
    rewards = []

    # make data readable, preprocess
    for line in training_data:
        values = line.strip().split()
        
        if len(values) == 0 or "Game" in values[0]:
            # game is over
            continue
        
        score = int(values[16])
        state = [[int(values[0]), int(values[1]), int(values[2]), int(values[3])],
                 [int(values[4]), int(values[5]), int(values[6]), int(values[7])],
                 [int(values[8]), int(values[9]), int(values[10]), int(values[11])],
                 [int(values[12]), int(values[13]), int(values[14]), int(values[15])]]

        rewards.append(score)
        state = np.array(state)

        # convert all 0s in states to 1s for log base purposes
        state[state == 0] = 1
        # log base anything of 1 = 0
        state = np.log2(state)
        # simplifies the values (normalization), compresses the numbers on the board
        states.append(state)

    states = np.array(states) / 16.0
    rewards = np.array(rewards)

    return states, rewards

if __name__ == "__main__":
    states, rewards = load_data()

    # Split data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(states, rewards, test_size=0.2)

    # Normalize rewards
    scaler = MinMaxScaler()
    y_train = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val = scaler.transform(y_val.reshape(-1, 1)).flatten()

    # Reshape states to include channel dimension
    x_train = x_train.reshape((-1, 4, 4, 1))
    x_val = x_val.reshape((-1, 4, 4, 1))

    num_classes = len(np.unique(rewards))

    # Initialize model
    model = keras.Sequential(
        [
            keras.Input(shape=(4, 4, 1)),
            keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(128, activation='linear'),
            keras.layers.Dense(num_classes, activation='linear'),  # Linear activation for regression
        ]
    )

    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_absolute_error"])

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=60)
    lr_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, verbose=1)

    model.summary()

    model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=128, epochs=50,
            verbose=1, callbacks=[early_stopping, lr_reduction])

    model.save("nn_reward_model_regression.keras")
