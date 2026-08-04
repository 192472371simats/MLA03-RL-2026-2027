import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Policy Network
model = Sequential([
    Dense(16, activation='relu', input_shape=(3,)),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy')

# Lane state: [Left Distance, Center Offset, Right Distance]
state = np.array([[0.8, 0.1, 0.9]])

prob = model.predict(state, verbose=0)

actions = ["Steer Left", "Steer Right"]

print("Action Probabilities:")
print(np.round(prob, 3))

print("\nSelected Action:", actions[np.argmax(prob)])