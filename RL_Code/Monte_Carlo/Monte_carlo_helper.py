import numpy as np
import matplotlib.pyplot as plt

def generate_episode(env,policy):
    states, actions, rewards = [], [], []
    state,_ = env.reset()
    action = np.random.choice([0,1]) #Random first Choice
    while True:
        states.append(state)
        actions.append(action)
        next_state, reward, done, _,_ = env.step(action)
        rewards.append(reward)
        if done:
            break
        state = next_state
        action = int(np.random.choice([0,1],p=policy.get(next_state,[0.5,0.5])))
        # action = policy.get(next_state,np.random.choice(env.action_space.n))
    return states, actions, rewards

def plot_blackjack_V(V,Title):
    fig = plt.figure(figsize=(12, 6))

    player_sum = np.arange(12, 22)  # Player sum range 12-21
    dealer_show = np.arange(1, 11)  # Dealer cards 1-10
    usable_ace = np.array([False, True])  # No/Yes usable ace

    state_values = np.zeros((len(player_sum), len(dealer_show), len(usable_ace)))

    for i, player in enumerate(player_sum):
        for j, dealer in enumerate(dealer_show):
            for k, ace in enumerate(usable_ace):
                state_values[i, j, k] = V.get((player, dealer, int(ace)), 0)  # Use .get() to avoid errors
    
    X, Y = np.meshgrid(player_sum, dealer_show, indexing="ij")

    fig.suptitle(Title)
    # No Usable Ace
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_wireframe(X, Y, state_values[:, :, 0])
    ax1.set_title("No Usable Ace")

    # With Usable Ace
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_wireframe(X, Y, state_values[:, :, 1])
    ax2.set_title("With Usable Ace")

    for ax in [ax1, ax2]:
        ax.set_zlim(-1, 1)
        ax.set_ylabel('Dealer Showing')
        ax.set_xlabel('Player Sum')
        ax.set_zlabel('State Value')

    plt.show()

def plot_blackjack_Q(Q,Title):
    V = {(p,d,a):0 for p in range(32) for d in range(11) for a in range(2)}
    for key in Q.keys():
        for key2 in Q[key].keys():
            V[key] += Q[key][key2]
        V[key] /= len(Q[key].keys())
    plot_blackjack_V(V,Title)

def plot_policy(policy):
    player_sums = np.arange(12, 22)  # Player's sum (rows)
    dealer_cards = np.arange(1, 11)  # Dealer's face-up card (columns)

    policy_matrix_no_useable_ace = np.zeros((len(player_sums), len(dealer_cards)))
    policy_matrix_useable_ace = np.zeros((len(player_sums), len(dealer_cards)))

    for i, p in enumerate(player_sums):
        for j, d in enumerate(dealer_cards):
            policy_matrix_no_useable_ace[i, j] = np.argmax(policy.get((p, d, 0), 0))  # No usable ace
            policy_matrix_useable_ace[i, j] = np.argmax(policy.get((p, d, 1), 0))  # With usable ace

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Monte Carlo Policy Visualization (Hit = 1, Stick = 0)")

    # No Usable Ace
    im1 = axs[0].imshow(policy_matrix_no_useable_ace, cmap="coolwarm", aspect="auto", vmin=0, vmax=1)
    axs[0].set_title("No Usable Ace")
    axs[0].set_xlabel("Dealer's Face-up Card")
    axs[0].set_ylabel("Player's Sum")
    axs[0].set_xticks(np.arange(len(dealer_cards)))
    axs[0].set_yticks(np.arange(len(player_sums)))
    axs[0].set_xticklabels(dealer_cards)
    axs[0].set_yticklabels(player_sums)

    # With Usable Ace
    im2 = axs[1].imshow(policy_matrix_useable_ace, cmap="coolwarm", aspect="auto", vmin=0, vmax=1)
    axs[1].set_title("Usable Ace")
    axs[1].set_xlabel("Dealer's Face-up Card")
    axs[1].set_ylabel("Player's Sum")
    axs[1].set_xticks(np.arange(len(dealer_cards)))
    axs[1].set_yticks(np.arange(len(player_sums)))
    axs[1].set_xticklabels(dealer_cards)
    axs[1].set_yticklabels(player_sums)

    # Add colorbars
    fig.colorbar(im1, ax=axs[0], label="Action (0 = Stick, 1 = Hit)")
    fig.colorbar(im2, ax=axs[1], label="Action (0 = Stick, 1 = Hit)")

    plt.show()
