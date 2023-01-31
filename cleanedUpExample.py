import gym
import numpy
import random
import os
import sys
import logging
import tensorflow as tf
from tensorflow.python import keras
from keras.models import Sequential
from keras.layers import Dense

class DeepQAgent():

    def __init__(self, modelName= "DeepQModel", epsilon= 1, epsilonDecay= .998, minEpsilon= 0.05, discountCost= .8, learningRate= .001, inputStateSize= 4, outputStateSize= 2, loadModel= False):
        # Model specific parameters
        self.modelName = modelName
        self.inputStateSize = inputStateSize
        self.outputStateSize = outputStateSize        

        # Training Parameters
        self.epsilon = epsilon
        self.epsilonDecayRate = epsilonDecay
        self.minEpsilon = minEpsilon
        self.discountCost = discountCost
        self.learningRate = learningRate

        self.env = gym.make("CartPole-v1", render_mode= "human")
        self.logger = self.initLogging()
        self.logger.info(f"Logging is setup")
        self.model = self.initModel()
        if loadModel: self.loadModel()

    def initModel(self):
        self.logger.info("Initializing model...")
        model = Sequential()
        model.add(Dense(16, input_dim= self.inputStateSize, activation='linear'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.outputStateSize, activation='linear'))
        lossFunction = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate= self.learningRate)
        model.compile(loss= lossFunction, optimizer= optimizer)
        self.logger.info("Model initialized")
        return model

    def initLogging(self):
        loggerName = os.path.basename(__file__)[:-2] + 'log'
        logger = logging.getLogger(loggerName)
        consoleHandler = logging.StreamHandler(sys.stdout)

        formatter = logging.Formatter('%(asctime)s:%(levelname)s:(%(threadName)-10s): %(message)s')
        consoleHandler.setFormatter(formatter)
        logger.addHandler(consoleHandler)
        logger.setLevel(logging.INFO)
        return logger

    def playGame(self):
        currentState, _ = self.env.reset()
        steps = 0
        done = False
        trainingMemory = []

        while not done:
            action = self.getMove(currentState)
            nextState, reward, done, _, _ = self.env.step(action)
            steps += 1
            if done: reward -= 50
            trainingMemory.append([currentState, action, nextState, reward, done])
            currentState = nextState

        self.logger.info(f"Episode {episode} complete after {steps} steps")
        return trainingMemory

    def getMove(self, gameState):
        if numpy.random.rand() <= self.epsilon:
            action = self.env.action_space.sample()
            return action
        else:
            modelInput = self.formatInputs(gameState)
            qTable = self.model.predict(modelInput)[0]
            action = numpy.argmax(qTable)
            return action

    def formatInputs(self, inputs):
        return numpy.reshape(inputs, [1, len(inputs)])

    def trainModel(self, trainingMemory):
        self.logger.info(f"Beginning training with {len(trainingMemory)} inputs...")
        trainingLossHistory = []
        for state, action, nextState, reward, done in trainingMemory:
            trainingInput = self.formatInputs(state)
            modelOutput = self.model.predict(trainingInput)[0]
            if not done:
                nextTrainingInput = self.formatInputs(nextState)
                nextReward = max(self.model.predict(nextTrainingInput)[0])
                reward = reward + (self.discountCost * nextReward)
            modelOutput[action] = reward
            trainingData = self.formatInputs(modelOutput)
            trainingResults = self.model.fit(trainingInput, trainingData, verbose= 0)
            trainingLossHistory.extend(trainingResults.history['loss'])

        self.epsilon = max(self.epsilon * self.epsilonDecayRate, self.minEpsilon)
        averageLoss = sum(trainingLossHistory) / len(trainingLossHistory)
        self.logger.info(f"Training of {len(trainingMemory)} data points complete with new epsilon value of: {self.epsilon}")
        self.logger.info(f"Average loss after training is: {averageLoss}")
        return trainingLossHistory

    def saveModel(self, lossHistory= None):
        self.logger.info(f"Saving model as {self.modelName}...")
        modelWeightsName = self.modelName + ".weights"
        totalDirPath = os.path.join(os.getcwd(), "models")
        self.model.save_weights(os.path.join(totalDirPath, modelWeightsName))
        self.logger.info("Model saved")

        if lossHistory is None: return # Dont write to log if not tracking loss history
        self.logger.info(f"Writing loss history to {self.modelName}'s training log...")
        logFileName = self.modelName + ".log"
        with open(os.path.join(totalDirPath, logFileName), 'a+') as file:
            averageLoss = sum(lossHistory) / len(lossHistory)
            file.write(str(averageLoss))
            file.write('\n')
        self.logger.info("Loss history logged")

    def loadModel(self):
        self.logger.info(f"Loading model {self.modelName}...")
        totalDirPath = os.path.join(os.getcwd(), "models")
        self.model.load_weights(os.path.join(totalDirPath, self.modelName + ".weights"))
        self.logger.info("Model successfully loaded")

if __name__ == "__main__":

    agent = DeepQAgent()

    numEpisodes = 1000
    for episode in range(numEpisodes):
        memory = agent.playGame()
        lossHistory = agent.trainModel(memory)
        agent.saveModel(lossHistory)

    agent.env.close()
