import gym

numberOfGamesToPlay = 10
env = gym.make("CartPole-v1", render_mode= "human")

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
