import gym
import tensorflow as tf
from tensorflow.python import keras
from keras.models import Sequential
from keras.layers import Dense

numberOfGamesToPlay = 10
env = gym.make("CartPole-v1", render_mode= "human")

def initializeModel(inputStateSize= 4, outputStateSize= 2, learningRate= 0.001):
    model = Sequential()
    model.add(Dense(16, input_dim= inputStateSize, activation='linear'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(outputStateSize, activation='linear'))
    lossFunction = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate= learningRate)
    model.compile(loss= lossFunction, optimizer= optimizer)
    return model

model = initializeModel()

for episode in range(numberOfGamesToPlay):
    currentState, _ = env.reset()
    done = False
    trainingMemory = []

    while not done:
        action = env.action_space.sample()
        currentState, reward, done, _, _ = env.step(action)
        trainingMemory.append([currentState, action, reward])

    print(f"Episode {episode} has finished")

env.close()
