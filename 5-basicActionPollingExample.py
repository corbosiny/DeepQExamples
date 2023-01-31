import gym
import numpy
import random
import tensorflow as tf
from tensorflow.python import keras
from keras.models import Sequential
from keras.layers import Dense

numberOfGamesToPlay = 10
env = gym.make("CartPole-v1", render_mode= "human")
chanceToExploreRandomMove = 1        # Epsilon Value
chanceToExploreDecayRate = .995      # How fast we switch from exploration to explotation
minChanceToExplore = .05             # The minimum amount of exploration we want to do each training run

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

def trainModel(model, trainingMemory):
    trainingLossHistory = []
    for state, action, reward in trainingMemory:
        trainingInput = numpy.reshape(state, [1, len(state)])
        modelOutput = model.predict(trainingInput)[0]
        modelOutput[action] = reward
        trainingData = numpy.reshape(modelOutput, [1, len(modelOutput)])
        trainingResults = model.fit(trainingInput, trainingData, verbose= 0)
        trainingLossHistory.extend(trainingResults.history['loss'])

    return trainingLossHistory

def getMove(state, model):
    if numpy.random.rand() <= chanceToExploreRandomMove:
        action = env.action_space.sample()
        return action
    else:
        modelInput = numpy.reshape(state, [1, len(state)])
        qTable = model.predict(modelInput)[0]
        action = numpy.argmax(qTable)
        return action

model = initializeModel()

for episode in range(numberOfGamesToPlay):
    currentState, _ = env.reset()
    done = False
    trainingMemory = []

    while not done:
        action = getMove(currentState, model)
        currentState, reward, done, _, _ = env.step(action)
        if done: reward -= 10
        trainingMemory.append([currentState, action, reward])

    print(f"Episode {episode} has finished")
    print("Beginning training...")
    trainModel(model, trainingMemory)
    print("Training complete")
    chanceToExploreRandomMove = max(chanceToExploreRandomMove * chanceToExploreDecayRate, minChanceToExplore)
    print(f"New epsilon value: {chanceToExploreRandomMove}")

env.close()
