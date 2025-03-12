import gym
import numpy as np
import pickle as pkl

# Creates the environment
cliffEnv = gym.make("CliffWalking-v0", render_mode=None)  # Or "human" if you want to see it

# Initializing Q Table it has 48 states and 4 actions(up,right,left,down)
q_table = np.zeros(shape=(48, 4))

# epsilon-greedy policy
def policy(state, explore_prob=0.0):
    if np.random.random() < explore_prob:
        return cliffEnv.action_space.sample()  # Return a random action
    else:
        return int(np.argmax(q_table[state]))  # Return the best action

# Parameters
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
NUM_EPISODES = 500

# Training for 500 episodes
for episode in range(NUM_EPISODES):

    # Initializing episode
    state, info = cliffEnv.reset()  # Get initial state and info
    state=state #the state can be accessed with state[0]
    action = policy(state, EPSILON)  # Choose first action

    total_reward = 0
    episode_length = 0
    terminated = False
    truncated = False

    # For every step of the episode
    while not terminated and not truncated:

        # Take an action in the environment
        next_state, reward, terminated, truncated, info = cliffEnv.step(action)
        next_state=next_state
        # Select the next action
        next_action = policy(next_state, EPSILON)

        # SARSA update
        q_table[state, action] = q_table[state, action] + ALPHA * (
            reward + GAMMA * q_table[next_state, next_action] - q_table[state, action]
        )

        state = next_state
        action = next_action

        total_reward += reward
        episode_length += 1

    print(f"Episode: {episode}, Episode Length: {episode_length}, Total Reward: {total_reward}")

cliffEnv.close()

pkl.dump(q_table, open("sarsa_q_table.pkl", "wb"))
print("Training Complete. Q Table Saved")