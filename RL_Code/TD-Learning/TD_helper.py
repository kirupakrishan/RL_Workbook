import numpy as np
import matplotlib.pyplot as plt

def generate_episode(env,policy):
    states, actions, rewards = [], [], []
    state,_ = env.reset()
    action = np.random.choice([0,1,2,3]) #Random first Choice
    while True:
        states.append(state)
        actions.append(action)
        next_state, reward, done, _,_ = env.step(action)
        rewards.append(reward)
        if done:
            break
        state = next_state
        action = int(np.random.choice([0,1,2,3],p=policy.get(next_state)))
        # action = policy.get(next_state,np.random.choice(env.action_space.n))
    return states, actions, rewards


def epsilon_greedy_policy(Q1, Q2, epsilon, nA):
    def policy_fn(observation):
        # Sum Q1 and Q2 for action value estimation
        action_values = Q1[observation] + Q2[observation]
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(action_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def print_policy(Q,Title):
    # Create a color map to visualize the directions
    # 0: empty space (white), 1: move right (green), 2: move down (blue), 3: move left (red)
    cmap = plt.cm.get_cmap('coolwarm', 4)

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the grid
    cax = ax.matshow(Q, cmap=cmap)
    ax.title.set_text(Title)
    # Add labels for directions
    directions = {0: '↑', 1: '→', 2: '↓', 3: '←'}

    # Overlay the direction labels
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            ax.text(j, i, directions[Q[i, j]], ha='center', va='center', color='black', fontsize=12)

    # Set the grid lines
    ax.set_xticks(np.arange(-0.5, Q.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, Q.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)

    return fig, ax
