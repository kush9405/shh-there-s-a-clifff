import gym
import numpy as np
import pickle as pkl

cliff_walking = gym.make('CliffWalking-v0', render_mode=None)
q_table = np.zeros(shape=(48, 4))  # Corrected shape of Q-table: (num_states, num_actions)

EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
NUM_EPISODES = 500

def policy(state, explore_prob=0.0):
    action = np.argmax(q_table[state])
    if np.random.rand() < explore_prob:
        action = cliff_walking.action_space.sample()
    return action

# def policy(state, explore_prob=0.0):
#     if np.random.random() < explore_prob:
#         return cliff_walking.action_space.sample()  # Return a random action
#     else:
#         return int(np.argmax(q_table[state]))


# Q-learning algorithm
for epsiode in range(NUM_EPISODES):
    terminated = False
    truncated = False
    total_reward = 0
    epsiode_length = 0

    state, info = cliff_walking.reset()
    state = state

    while not terminated and not truncated:
        action = policy(state, EPSILON)

        next_state, reward, terminated, truncated, info = cliff_walking.step(action)
        next_state = next_state

        next_action = policy(next_state)

        q_table[state, action] = q_table[state, action] + ALPHA * (
            reward + GAMMA * q_table[next_state, next_action] - q_table[state, action]
        )
        state=next_state
        # action=next_action
        total_reward += reward
        epsiode_length += 1
    print(f"Episode: {epsiode}, Episode Length: {epsiode_length}, Total Reward: {total_reward}")
cliff_walking.close()
pkl.dump(q_table, open("q_learning_q_table.pkl", "wb"))
print("Q-table saved to q_table.pkl")