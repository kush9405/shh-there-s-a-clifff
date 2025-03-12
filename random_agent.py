import gym
import numpy as np

# created environment 
cliff_walking = gym.make('CliffWalking-v0',render_mode="human")

# making it false so that I can have infinite loop kinda thing
done = False
# resetting the environment to get the initial state
state = cliff_walking.reset()

# infinte loop so that it keeps running until it reaches the terminal state
while not done:
    # using render we can see the environment
    print("Render: ",cliff_walking.render())  # Remove mode='ansi'

    # predefined actions are 0,1,2,3
    action = cliff_walking.action_space.sample()
    # print(action)
    # 0 - up
    # 1 - right
    # 2 - down  
    # 3 - left
    print(state, "--->", ['up','right','down','left'][action])
    
    # taking the action in the environment
    state, reward,terminated, truncated, info = cliff_walking.step(action)

# closing the environment
cliff_walking.close()