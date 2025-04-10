from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

model = Sequential()
model.add(Dense(128, input_shape=(144,), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(144, activation='linear'))  # assuming 144 possible moves
model.compile(optimizer=SGD(), loss='mse')

model.save("sid_dummy_144.h5")
print("✅ Dummy model with 144 input saved as sid_dummy_144.h5!")
