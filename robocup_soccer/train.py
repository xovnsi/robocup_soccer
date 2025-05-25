import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import pygame
import math
import time
from collections import deque, namedtuple

from CriticNetwork import ImprovedCriticNetwork
from FootballSimulation import ImprovedFootballSimulation
from ReplayMemory import ReplayMemory, Experience
from TeamAgent import ImprovedTeamAgent
from constants import *



def select_action_with_noise(policy_net, state, noise_scale, device, action_low=-1.0, action_high=1.0):
    """Select action with Gaussian noise for exploration"""
    policy_net.eval()
    with torch.no_grad():
        action = policy_net(state.to(device))
    policy_net.train()

    # Add Gaussian noise
    noise = noise_scale * torch.randn_like(action)
    action = action + noise

    # Clamp action within valid bounds
    action = torch.clamp(action, action_low, action_high)
    return action



def soft_update(target_net, source_net, tau):
    """Soft update target network"""
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def optimize_ddpg_improved(policy_net, critic_net, target_policy_net, target_critic_net,
                           optimizer_policy, optimizer_critic, memory, batch_size, device,
                           gamma=GAMMA, clip_q=True, q_clip_range=(-10.0, 10.0)):
    """Improved DDPG optimization with gradient clipping and optional Q-value clipping"""
    if len(memory) < batch_size:
        return None, None

    experiences = memory.sample(batch_size)
    batch = Experience(*zip(*experiences))

    states = torch.cat(batch.state).to(device)
    actions = torch.cat(batch.action).to(device)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.cat(batch.next_state).to(device)
    dones = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)

    # ------------------- Critic Update ------------------- #
    with torch.no_grad():
        next_actions = target_policy_net(next_states)
        target_q_values = target_critic_net(next_states, next_actions)
        target_q = rewards + (1 - dones) * gamma * target_q_values

        if clip_q:
            target_q = torch.clamp(target_q, q_clip_range[0], q_clip_range[1])

    current_q = critic_net(states, actions)
    critic_loss = F.mse_loss(current_q, target_q)

    optimizer_critic.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic_net.parameters(), 1.0)
    optimizer_critic.step()

    # ------------------- Actor Update ------------------- #
    pred_actions = policy_net(states)
    actor_loss = -critic_net(states, pred_actions).mean()

    optimizer_policy.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer_policy.step()

    return actor_loss.item(), critic_loss.item()


def train_improved_agents(num_episodes=2000, max_steps_per_episode=1000,
                          batch_size=BATCH_SIZE, num_players=3, render_every=100):
    """Improved training loop"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Initialize simulation
    sim = ImprovedFootballSimulation(num_players_per_team=num_players, use_rendering=False)

    # Network dimensions
    state_dim = 4 + 4 * num_players * 2  # ball + 2 teams * num_players * (pos + vel)
    action_dim_per_player = 6  # [move_x, move_y, kick, kick_power, kick_dir_x, kick_dir_y]
    total_action_dim = num_players * action_dim_per_player

    # Initialize networks for both teams
    # Team A networks
    policy_net_a = ImprovedTeamAgent(num_players, state_dim, action_dim_per_player).to(device)
    target_policy_net_a = ImprovedTeamAgent(num_players, state_dim, action_dim_per_player).to(device)
    critic_net_a = ImprovedCriticNetwork(state_dim, total_action_dim).to(device)
    target_critic_net_a = ImprovedCriticNetwork(state_dim, total_action_dim).to(device)

    # Team B networks
    policy_net_b = ImprovedTeamAgent(num_players, state_dim, action_dim_per_player).to(device)
    target_policy_net_b = ImprovedTeamAgent(num_players, state_dim, action_dim_per_player).to(device)
    critic_net_b = ImprovedCriticNetwork(state_dim, total_action_dim).to(device)
    target_critic_net_b = ImprovedCriticNetwork(state_dim, total_action_dim).to(device)

    # Initialize target networks
    target_policy_net_a.load_state_dict(policy_net_a.state_dict())
    target_critic_net_a.load_state_dict(critic_net_a.state_dict())
    target_policy_net_b.load_state_dict(policy_net_b.state_dict())
    target_critic_net_b.load_state_dict(critic_net_b.state_dict())

    # Optimizers
    optimizer_policy_a = optim.Adam(policy_net_a.parameters(), lr=LR_ACTOR)
    optimizer_critic_a = optim.Adam(critic_net_a.parameters(), lr=LR_CRITIC)
    optimizer_policy_b = optim.Adam(policy_net_b.parameters(), lr=LR_ACTOR)
    optimizer_critic_b = optim.Adam(critic_net_b.parameters(), lr=LR_CRITIC)

    # Replay memories
    memory_a = ReplayMemory(MEMORY_SIZE)
    memory_b = ReplayMemory(MEMORY_SIZE)

    # Training metrics
    episode_rewards_a = []
    episode_rewards_b = []
    actor_losses_a = []
    critic_losses_a = []
    actor_losses_b = []
    critic_losses_b = []

    # Noise parameters
    noise_scale = NOISE_SCALE_START

    print("Starting training...")

    for episode in range(num_episodes):
        state = sim.reset()
        episode_reward_a = 0
        episode_reward_b = 0

        for step in range(max_steps_per_episode):
            # Select actions with noise
            action_a = select_action_with_noise(policy_net_a, state, noise_scale, device)
            action_b = select_action_with_noise(policy_net_b, state, noise_scale, device)

            # Take step in environment
            next_state, (reward_a, reward_b), done, info = sim.step(action_a, action_b)

            episode_reward_a += reward_a
            episode_reward_b += reward_b

            # Store experiences
            memory_a.push(state, action_a, reward_a, next_state, done)
            memory_b.push(state, action_b, reward_b, next_state, done)

            state = next_state

            # Optimize networks
            if len(memory_a) >= batch_size:
                actor_loss_a, critic_loss_a = optimize_ddpg_improved(
                    policy_net_a, critic_net_a, target_policy_net_a, target_critic_net_a,
                    optimizer_policy_a, optimizer_critic_a, memory_a, batch_size, device,
                    clip_q=True, q_clip_range=(-10.0, 10.0))

                actor_loss_b, critic_loss_b = optimize_ddpg_improved(
                    policy_net_b, critic_net_b, target_policy_net_b, target_critic_net_b,
                    optimizer_policy_b, optimizer_critic_b, memory_b, batch_size, device,
                    clip_q=True, q_clip_range=(-10.0, 10.0))


                if actor_loss_a is not None:
                    actor_losses_a.append(actor_loss_a)
                    critic_losses_a.append(critic_loss_a)
                if actor_loss_b is not None:
                    actor_losses_b.append(actor_loss_b)
                    critic_losses_b.append(critic_loss_b)

                # Soft update target networks
                soft_update(target_policy_net_a, policy_net_a, TAU)
                soft_update(target_critic_net_a, critic_net_a, TAU)
                soft_update(target_policy_net_b, policy_net_b, TAU)
                soft_update(target_critic_net_b, critic_net_b, TAU)

            if done:
                break

        # Update noise
        noise_scale = max(NOISE_SCALE_END, noise_scale * NOISE_DECAY)

        # Record episode rewards
        episode_rewards_a.append(episode_reward_a)
        episode_rewards_b.append(episode_reward_b)

        # Print progress
        if episode % 10 == 0:
            avg_reward_a = np.mean(episode_rewards_a[-50:]) if len(episode_rewards_a) >= 50 else np.mean(episode_rewards_a)
            avg_reward_b = np.mean(episode_rewards_b[-50:]) if len(episode_rewards_b) >= 50 else np.mean(episode_rewards_b)
            print(f"Episode {episode}: Avg Reward A: {avg_reward_a:.2f}, Avg Reward B: {avg_reward_b:.2f}, "
                  f"Score: {sim.score_a}-{sim.score_b}, Noise: {noise_scale:.3f}")

        # Render periodically
        # if episode % render_every == 0 and episode > 0:
        #     print(f"Rendering episode {episode}...")
        #     evaluate_agents(policy_net_a, policy_net_b, num_players, render=True)

    return (policy_net_a, policy_net_b, critic_net_a, critic_net_b,
            episode_rewards_a, episode_rewards_b, actor_losses_a, critic_losses_a, actor_losses_b, critic_losses_b)


def save_models(policy_net_a, policy_net_b, critic_net_a, critic_net_b, episode, filepath_prefix="football_models"):
    """Save trained models"""
    torch.save({
        'policy_net_a': policy_net_a.state_dict(),
        'policy_net_b': policy_net_b.state_dict(),
        'critic_net_a': critic_net_a.state_dict(),
        'critic_net_b': critic_net_b.state_dict(),
        'episode': episode
    }, f"{filepath_prefix}_episode_{episode}.pth")
    print(f"Models saved as {filepath_prefix}_episode_{episode}.pth")

def load_models(policy_net_a, policy_net_b, critic_net_a, critic_net_b, filepath):
    """Load trained models"""
    checkpoint = torch.load(filepath)
    policy_net_a.load_state_dict(checkpoint['policy_net_a'])
    policy_net_b.load_state_dict(checkpoint['policy_net_b'])
    critic_net_a.load_state_dict(checkpoint['critic_net_a'])
    critic_net_b.load_state_dict(checkpoint['critic_net_b'])
    episode = checkpoint['episode']
    print(f"Models loaded from {filepath}, trained for {episode} episodes")
    return episode

def plot_training_results(episode_rewards_a, episode_rewards_b, actor_losses_a, critic_losses_a,
                         actor_losses_b, critic_losses_b):
    """Plot training results"""
    import matplotlib.pyplot as plt

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # Episode rewards
    ax1.plot(episode_rewards_a, label='Team A', alpha=0.7)
    ax1.plot(episode_rewards_b, label='Team B', alpha=0.7)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend()
    ax1.grid(True)

    # Moving average of rewards
    window = 50
    if len(episode_rewards_a) >= window:
        moving_avg_a = np.convolve(episode_rewards_a, np.ones(window)/window, mode='valid')
        moving_avg_b = np.convolve(episode_rewards_b, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(episode_rewards_a)), moving_avg_a, label='Team A (MA)', linewidth=2)
        ax2.plot(range(window-1, len(episode_rewards_b)), moving_avg_b, label='Team B (MA)', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Moving Average Reward')
    ax2.set_title(f'Moving Average Rewards (window={window})')
    ax2.legend()
    ax2.grid(True)

    # Actor losses
    if actor_losses_a:
        ax3.plot(actor_losses_a, label='Team A', alpha=0.7)
        ax3.plot(actor_losses_b, label='Team B', alpha=0.7)
        ax3.set_xlabel('Update Step')
        ax3.set_ylabel('Actor Loss')
        ax3.set_title('Actor Losses')
        ax3.legend()
        ax3.grid(True)

    # Critic losses
    if critic_losses_a:
        ax4.plot(critic_losses_a, label='Team A', alpha=0.7)
        ax4.plot(critic_losses_b, label='Team B', alpha=0.7)
        ax4.set_xlabel('Update Step')
        ax4.set_ylabel('Critic Loss')
        ax4.set_title('Critic Losses')
        ax4.legend()
        ax4.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    """Main training and evaluation function"""
    print("Football Simulation with Improved DDPG Training")
    print("=" * 50)

    # Training parameters
    num_episodes = 100
    num_players = 6
    render_every = 10

    try:
        # Train agents
        results = train_improved_agents(
            num_episodes=num_episodes,
            num_players=num_players,
            render_every=render_every
        )

        (policy_net_a, policy_net_b, critic_net_a, critic_net_b,
         episode_rewards_a, episode_rewards_b, actor_losses_a, critic_losses_a,
         actor_losses_b, critic_losses_b) = results

        # Save models
        save_models(policy_net_a, policy_net_b, critic_net_a, critic_net_b, num_episodes)

        # Final evaluation
        # print("\nFinal evaluation with rendering...")
        # evaluate_agents(policy_net_a, policy_net_b, num_players, num_episodes=10, render=True)

        # Plot results
        print("\nPlotting training results...")
        plot_training_results(episode_rewards_a, episode_rewards_b, actor_losses_a, critic_losses_a,
                            actor_losses_b, critic_losses_b)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()

def demo_pretrained():
    """Demo function to run with pretrained models if available"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_players = 3
    state_dim = 4 + 4 * num_players * 2
    action_dim_per_player = 6

    # Initialize networks
    policy_net_a = ImprovedTeamAgent(num_players, state_dim, action_dim_per_player).to(device)
    policy_net_b = ImprovedTeamAgent(num_players, state_dim, action_dim_per_player).to(device)
    critic_net_a = ImprovedCriticNetwork(state_dim, num_players * action_dim_per_player).to(device)
    critic_net_b = ImprovedCriticNetwork(state_dim, num_players * action_dim_per_player).to(device)

    try:
        # Try to load pretrained models
        episode = load_models(policy_net_a, policy_net_b, critic_net_a, critic_net_b,
                             "football_models_episode_2000.pth")
        print(f"Loaded pretrained models from episode {episode}")

        # Run evaluation
        # evaluate_agents(policy_net_a, policy_net_b, num_players, num_episodes=5, render=True)

    except FileNotFoundError:
        print("No pretrained models found. Please run training first.")
        print("Running with random agents for demonstration...")
        # evaluate_agents(policy_net_a, policy_net_b, num_players, num_episodes=3, render=True)

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_pretrained()
    else:
        main()