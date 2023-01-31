import gym
import numpy
import random
import os
import tensorflow as tf
from tensorflow.python import keras
from keras.models import Sequential
from keras.layers import Dense

numberOfGamesToPlay = 1000
env = gym.make("CartPole-v1", render_mode= "human")
chanceToExploreRandomMove = 0        # Setting epsilon to 0 so we can watch the loaded model make all of it's own desicions, reset this to a normal value to train
chanceToExploreDecayRate = .995      # How fast we switch from exploration to explotation
minChanceToExplore = 0               # Setting epsilon to 0 so we can watch the loaded model make all of it's own desicions, reset this to a normal value to train
discountFactor = .6                  # How much to weight future predictions when factoring them into the Q score
modelName = "DeepQModel"
loadModel = True

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
    for state, action, nextState, reward, done in trainingMemory:
        trainingInput = numpy.reshape(state, [1, len(state)])
        modelOutput = model.predict(trainingInput)[0]
        if not done:
            nextTrainingInput = numpy.reshape(nextState, [1, len(nextState)])
            nextReward = max(model.predict(nextTrainingInput)[0])
            reward = reward + (discountFactor * nextReward)
        modelOutput[action] = reward
        trainingData = numpy.reshape(modelOutput, [1, len(modelOutput)])
        trainingResults = model.fit(trainingInput, trainingData, verbose= 0)
        trainingLossHistory.extend(trainingResults.history['loss'])

    return trainingLossHistory

def saveModel(model, modelName, lossHistory):
    modelWeightsName = modelName + ".weights"
    totalDirPath = os.path.join(os.getcwd(), "models")
    model.save_weights(os.path.join(totalDirPath, modelWeightsName))
    logFileName = modelName + ".log"
    with open(os.path.join(totalDirPath, logFileName), 'a+') as file:
        averageLoss = sum(lossHistory) / len(lossHistory)
        file.write(str(averageLoss))
        file.write('\n')

def loadModel(model, modelName):
    totalDirPath = os.path.join(os.getcwd(), "models")
    model.load_weights(os.path.join(totalDirPath, modelName + ".weights"))

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
if loadModel: loadModel(model, modelName)

for episode in range(numberOfGamesToPlay):
    currentState, _ = env.reset()
    steps = 0
    done = False
    trainingMemory = []

    while not done:
        action = getMove(currentState, model)
        nextState, reward, done, _, _ = env.step(action)
        steps += 1
        if done: reward -= 10
        trainingMemory.append([currentState, action, nextState, reward, done])
        currentState = nextState

    print(f"Episode {episode} has finished after {steps} steps")
    # Commented out training sections as this example is setup just to load and view an already trainined model
    #print("Beginning training...")
    #lossHistory = trainModel(model, trainingMemory)
    #print(f"Training complete with final training loss of {lossHistory[-1]}")
    #chanceToExploreRandomMove = max(chanceToExploreRandomMove * chanceToExploreDecayRate, minChanceToExplore)
    #print(f"New epsilon value: {chanceToExploreRandomMove}")
    #print("Saving model...")
    #saveModel(model, modelName, lossHistory)
    #print("Model saved")

env.close()
