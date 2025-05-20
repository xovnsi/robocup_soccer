import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import pygame
import math
import time
from collections import deque, namedtuple

# Constants from original simulation
FIELD_WIDTH = 105.0  # width in meters
FIELD_HEIGHT = 68.0  # height in meters
MAX_SPEED = 8.0      # max player speed (m/s)
MAX_KICK_POWER = 25.0  # max kick power (m/s)
BALL_DECELERATION = 0.95  # ball slowdown factor
PLAYER_ACCELERATION = 2.0  # player acceleration
PLAYER_DECELERATION = 0.9  # player slowdown factor
CONTROL_DISTANCE = 1.5     # ball control distance
TIME_STEP = 0.05           # simulation time step (seconds)
SCREEN_WIDTH = 800         # screen width in pixels
SCREEN_HEIGHT = 600        # screen height in pixels

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 128, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Reinforcement Learning parameters
GAMMA = 0.99          # discount factor
MEMORY_SIZE = 10000   # replay memory size
BATCH_SIZE = 64       # minibatch size for training
LR = 0.001            # learning rate
EPSILON_START = 1.0   # initial exploration rate
EPSILON_END = 0.05    # final exploration rate
EPSILON_DECAY = 10000 # frames to decay epsilon
TAU = 0.001           # for soft update of target network

# Reward weights
GOAL_REWARD = 10.0
OPPONENT_GOAL_PENALTY = -10.0
BALL_POSSESSION_REWARD = 0.01
OPPONENT_HALF_REWARD = 0.005
SUCCESSFUL_PASS_REWARD = 0.5

# Experience replay memory
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Experience(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class TeamAgent(nn.Module):
    """Neural network that controls all players of a team"""
    def __init__(self, num_players, state_dim, action_dim_per_player):
        super(TeamAgent, self).__init__()
        
        self.num_players = num_players
        self.state_dim = state_dim
        self.action_dim_per_player = action_dim_per_player
        self.total_action_dim = num_players * action_dim_per_player
        
        # Shared feature extraction layers
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # Action output layers for each player
        self.action_heads = nn.ModuleList([
            nn.Linear(256, action_dim_per_player) for _ in range(num_players)
        ])
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Get actions for each player
        player_actions = []
        for i in range(self.num_players):
            # Get action for player i
            actions = self.action_heads[i](x)
            player_actions.append(actions)
        
        # Combine all player actions
        team_actions = torch.cat(player_actions, dim=1)
        return team_actions


class FootballSimulation:
    def __init__(self, num_players_per_team=5, use_rendering=False):
        # Initialize device (CPU/GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize simulation time
        self.time_factor = 1.0
        self.simulation_time = 0.0
        
        # Initialize field
        self.field_width = torch.tensor(FIELD_WIDTH, device=self.device)
        self.field_height = torch.tensor(FIELD_HEIGHT, device=self.device)
        
        # Initialize ball
        self.ball_pos = torch.tensor([FIELD_WIDTH/2, FIELD_HEIGHT/2], dtype=torch.float32, device=self.device)
        self.ball_vel = torch.zeros(2, dtype=torch.float32, device=self.device)
        
        # Initialize teams
        self.num_players_per_team = num_players_per_team
        self.team_a_pos = self._initialize_team_positions(team="A")
        self.team_b_pos = self._initialize_team_positions(team="B")
        self.team_a_vel = torch.zeros((num_players_per_team, 2), dtype=torch.float32, device=self.device)
        self.team_b_vel = torch.zeros((num_players_per_team, 2), dtype=torch.float32, device=self.device)
        
        # Initialize action vectors for each player
        # [direction_x, direction_y, kick, kick_power, kick_direction_x, kick_direction_y]
        self.team_a_actions = torch.zeros((num_players_per_team, 6), dtype=torch.float32, device=self.device)
        self.team_b_actions = torch.zeros((num_players_per_team, 6), dtype=torch.float32, device=self.device)
        
        # Initialize ball possession info
        self.ball_possession = {"team": None, "player_id": None}
        self.last_possession = {"team": None, "player_id": None}
        
        # Scores
        self.score_a = 0
        self.score_b = 0
        
        # Use pygame visualization?
        self.use_rendering = use_rendering
        
        # Initialize pygame for visualization if needed
        if self.use_rendering:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Football Simulation 2D")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 12)
        
        # Tracking rewards and game state
        self.reset_reward_tracking()
        
        # Episode tracking
        self.current_episode = 0
        self.max_episode_steps = 600  # 30 seconds at TIME_STEP=0.05

    def reset_reward_tracking(self):
        """Reset all reward-related tracking variables"""
        self.last_team_with_ball = None
        self.successful_pass = False
        self.last_ball_pos = self.ball_pos.clone()
        self.steps_since_last_touch = 0
        self.current_step = 0
    
    def reset(self):
        """Reset the simulation for a new episode"""
        # Reset ball
        self.ball_pos = torch.tensor([FIELD_WIDTH/2, FIELD_HEIGHT/2], dtype=torch.float32, device=self.device)
        self.ball_vel = torch.zeros(2, dtype=torch.float32, device=self.device)
        
        # Reset teams
        self.team_a_pos = self._initialize_team_positions(team="A")
        self.team_b_pos = self._initialize_team_positions(team="B")
        self.team_a_vel = torch.zeros((self.num_players_per_team, 2), dtype=torch.float32, device=self.device)
        self.team_b_vel = torch.zeros((self.num_players_per_team, 2), dtype=torch.float32, device=self.device)
        
        # Reset actions
        self.team_a_actions = torch.zeros((self.num_players_per_team, 6), dtype=torch.float32, device=self.device)
        self.team_b_actions = torch.zeros((self.num_players_per_team, 6), dtype=torch.float32, device=self.device)
        
        # Reset possession
        self.ball_possession = {"team": None, "player_id": None}
        self.last_possession = {"team": None, "player_id": None}
        
        # Reset scores
        self.score_a = 0
        self.score_b = 0
        
        # Reset simulation time
        self.simulation_time = 0.0
        
        # Reset reward tracking
        self.reset_reward_tracking()
        self.current_episode += 1
        self.current_step = 0
        
        # Return initial state
        return self.get_state_tensor()

    def _initialize_team_positions(self, team):
        """Initialize player positions"""
        positions = torch.zeros((self.num_players_per_team, 2), dtype=torch.float32, device=self.device)
        
        if team == "A":  # Left team
            x_base = FIELD_WIDTH / 4
            positions[0] = torch.tensor([5.0, FIELD_HEIGHT/2])  # Goalkeeper
        else:  # Right team
            x_base = 3 * FIELD_WIDTH / 4
            positions[0] = torch.tensor([FIELD_WIDTH - 5.0, FIELD_HEIGHT/2])  # Goalkeeper
        
        # Position other players
        for i in range(1, self.num_players_per_team):
            if team == "A":
                positions[i] = torch.tensor([
                    x_base + random.uniform(-10, 10),
                    random.uniform(10, FIELD_HEIGHT - 10)
                ])
            else:
                positions[i] = torch.tensor([
                    x_base + random.uniform(-10, 10),
                    random.uniform(10, FIELD_HEIGHT - 10)
                ])
        
        return positions
    
    def set_team_actions(self, team, actions_tensor):
        """Sets actions for all players of a team from a neural network output"""
        # Actions_tensor has shape [1, num_players * 6]
        # Need to reshape to [num_players, 6]
        actions = actions_tensor.view(self.num_players_per_team, 6)
        
        # Process raw network outputs into valid actions
        processed_actions = torch.zeros_like(actions)
        
        # Movement direction (already normalized in the network)
        processed_actions[:, :2] = actions[:, :2]
        
        # Kick decision (binary)
        processed_actions[:, 2] = torch.sigmoid(actions[:, 2]) > 0.5
        
        # Kick power (0 to MAX_KICK_POWER)
        processed_actions[:, 3] = torch.sigmoid(actions[:, 3]) * MAX_KICK_POWER
        
        # Kick direction
        processed_actions[:, 4:6] = actions[:, 4:6]
        
        # Set actions for the team
        if team == "A":
            self.team_a_actions = processed_actions
        else:
            self.team_b_actions = processed_actions
    
    def _apply_movement(self, positions, velocities, actions):
        """Apply player movements based on their actions"""
        # Normalize movement direction
        directions = actions[:, :2]
        norms = torch.norm(directions, dim=1, keepdim=True)
        mask = (norms > 0).squeeze(-1)
        normalized_directions = torch.zeros_like(directions)
        
        # Safe normalization only for non-zero directions
        for i in range(len(directions)):
            if mask[i]:
                normalized_directions[i] = directions[i] / norms[i]
        
        # Update velocities
        accelerations = normalized_directions * PLAYER_ACCELERATION
        velocities = velocities * PLAYER_DECELERATION + accelerations * TIME_STEP * self.time_factor
        
        # Limit max speed
        vel_norms = torch.norm(velocities, dim=1, keepdim=True)
        vel_mask = (vel_norms > MAX_SPEED).squeeze(-1)
        
        # Safe speed limiting
        for i in range(len(velocities)):
            if vel_mask[i]:
                velocities[i] = velocities[i] / vel_norms[i] * MAX_SPEED
        
        # Update positions
        positions = positions + velocities * TIME_STEP * self.time_factor
        
        # Constrain positions to field boundaries
        positions[:, 0] = torch.clamp(positions[:, 0], 0, self.field_width)
        positions[:, 1] = torch.clamp(positions[:, 1], 0, self.field_height)
        
        return positions, velocities
    
    def _update_ball(self):
        """Update ball position and velocity"""
        # Track the current and last ball possession for reward calculation
        self.last_possession = self.ball_possession.copy()
        
        # Check kicks
        for team, positions, actions in [
            ("A", self.team_a_pos, self.team_a_actions),
            ("B", self.team_b_pos, self.team_b_actions)
        ]:
            for player_id in range(self.num_players_per_team):
                dist_to_ball = torch.norm(positions[player_id] - self.ball_pos)
                
                # If player is close enough to the ball
                if dist_to_ball < CONTROL_DISTANCE:
                    # Update ball possession
                    self.ball_possession = {"team": team, "player_id": player_id}
                    self.steps_since_last_touch = 0
                    
                    # If player kicks the ball
                    if actions[player_id, 2] > 0:
                        kick_power = torch.clamp(actions[player_id, 3], 0, MAX_KICK_POWER)
                        kick_direction = actions[player_id, 4:6]
                        if torch.norm(kick_direction) > 0:
                            kick_direction = kick_direction / torch.norm(kick_direction)
                        else:
                            # Default direction if not specified
                            if team == "A":
                                kick_direction = torch.tensor([1.0, 0.0], device=self.device)
                            else:
                                kick_direction = torch.tensor([-1.0, 0.0], device=self.device)
                        
                        # Apply impulse to ball
                        self.ball_vel = kick_direction * kick_power
                        break
                else:
                    # If no one controls the ball
                    if self.ball_possession["team"] == team and self.ball_possession["player_id"] == player_id:
                        self.ball_possession = {"team": None, "player_id": None}
        
        # Update ball position
        self.ball_pos = self.ball_pos + self.ball_vel * TIME_STEP * self.time_factor
        
        # Slow down ball due to friction
        self.ball_vel = self.ball_vel * BALL_DECELERATION
        
        # Goals and ball bouncing off boundaries
        goal_scored = False
        goal_team = None
        
        # Goal parameters
        goal_width = 7.32  # goal width in meters
        goal_y_start = (FIELD_HEIGHT - goal_width) / 2
        goal_y_end = goal_y_start + goal_width
        
        # Check if ball entered a goal
        if self.ball_pos[0] < 0:
            if goal_y_start < self.ball_pos[1] < goal_y_end:
                # Goal for team B
                goal_scored = True
                goal_team = "B"
                # Reset ball to center
                self.ball_pos = torch.tensor([FIELD_WIDTH/2, FIELD_HEIGHT/2], 
                                          dtype=torch.float32, device=self.device)
                self.ball_vel = torch.zeros(2, dtype=torch.float32, device=self.device)
            else:
                # Bounce off wall
                self.ball_pos[0] = 0
                self.ball_vel[0] = -self.ball_vel[0] * 0.7
        elif self.ball_pos[0] > self.field_width:
            if goal_y_start < self.ball_pos[1] < goal_y_end:
                # Goal for team A
                goal_scored = True
                goal_team = "A"
                # Reset ball to center
                self.ball_pos = torch.tensor([FIELD_WIDTH/2, FIELD_HEIGHT/2], 
                                          dtype=torch.float32, device=self.device)
                self.ball_vel = torch.zeros(2, dtype=torch.float32, device=self.device)
            else:
                # Bounce off wall
                self.ball_pos[0] = self.field_width
                self.ball_vel[0] = -self.ball_vel[0] * 0.7
            
        if self.ball_pos[1] < 0:
            self.ball_pos[1] = 0
            self.ball_vel[1] = -self.ball_vel[1] * 0.7
        elif self.ball_pos[1] > self.field_height:
            self.ball_pos[1] = self.field_height
            self.ball_vel[1] = -self.ball_vel[1] * 0.7
        
        # Check for successful pass
        self.successful_pass = False
        if (self.last_possession["team"] is not None and 
            self.ball_possession["team"] == self.last_possession["team"] and
            self.ball_possession["player_id"] != self.last_possession["player_id"]):
            self.successful_pass = True
            
        # Increment counter for steps since last ball touch
        if self.ball_possession["team"] is None:
            self.steps_since_last_touch += 1
            
        return goal_scored, goal_team
    
    def calculate_rewards(self, goal_scored, goal_team):
        """Calculate rewards for both teams"""
        reward_a = 0
        reward_b = 0
        
        # Goal rewards
        if goal_scored:
            if goal_team == "A":
                reward_a += GOAL_REWARD
                reward_b += OPPONENT_GOAL_PENALTY
            else:
                reward_a += OPPONENT_GOAL_PENALTY
                reward_b += GOAL_REWARD
        
        # Ball possession rewards
        if self.ball_possession["team"] == "A":
            reward_a += BALL_POSSESSION_REWARD
        elif self.ball_possession["team"] == "B":
            reward_b += BALL_POSSESSION_REWARD
        
        # Reward for keeping ball in opponent's half
        if self.ball_pos[0] > FIELD_WIDTH / 2:  # Ball in B's half
            reward_a += OPPONENT_HALF_REWARD
        else:  # Ball in A's half
            reward_b += OPPONENT_HALF_REWARD
        
        # Successful pass rewards
        if self.successful_pass:
            if self.ball_possession["team"] == "A":
                reward_a += SUCCESSFUL_PASS_REWARD
            elif self.ball_possession["team"] == "B":
                reward_b += SUCCESSFUL_PASS_REWARD
        
        return reward_a, reward_b
    
    def step(self, action_a=None, action_b=None):
        """Take one step in the simulation with given actions"""
        # Apply actions if provided
        if action_a is not None:
            self.set_team_actions("A", action_a)
        if action_b is not None:
            self.set_team_actions("B", action_b)
        
        # Update positions and velocities of players
        self.team_a_pos, self.team_a_vel = self._apply_movement(
            self.team_a_pos, self.team_a_vel, self.team_a_actions)
        self.team_b_pos, self.team_b_vel = self._apply_movement(
            self.team_b_pos, self.team_b_vel, self.team_b_actions)
        
        # Update ball and check goals
        goal_scored, goal_team = self._update_ball()
        
        # Update score
        if goal_scored:
            if goal_team == "A":
                self.score_a += 1
            else:
                self.score_b += 1
        
        # Calculate rewards
        reward_a, reward_b = self.calculate_rewards(goal_scored, goal_team)
        
        # Update simulation time
        self.simulation_time += TIME_STEP * self.time_factor
        self.current_step += 1
        
        # Get next state
        next_state = self.get_state_tensor()
        
        # Check if episode is done
        done = self.current_step >= self.max_episode_steps
        
        # Save previous ball position for next reward calculation
        self.last_ball_pos = self.ball_pos.clone()
        
        return next_state, (reward_a, reward_b), done, {"goal_scored": goal_scored, "goal_team": goal_team}
    
    def get_state_tensor(self):
        """Returns the game state as a tensor"""
        # Create tensor with all game state information
        # Format: [ball_x, ball_y, ball_vx, ball_vy, 
        #          team_a_player1_x, team_a_player1_y, team_a_player1_vx, team_a_player1_vy,
        #          ...,
        #          team_b_player1_x, team_b_player1_y, team_b_player1_vx, team_b_player1_vy,
        #          ...]
        
        # Reshape to [1, state_dim] for network input
        state_dim = 4 + 4 * self.num_players_per_team * 2  # Ball (4) + players (4 per player)
        state = torch.zeros(1, state_dim, device=self.device)
        
        # Ball [x, y, vx, vy]
        state[0, 0:4] = torch.cat([self.ball_pos, self.ball_vel])
        
        # Team A [x, y, vx, vy] for each player
        for i in range(self.num_players_per_team):
            idx = 4 + i * 4
            state[0, idx:idx+4] = torch.cat([self.team_a_pos[i], self.team_a_vel[i]])
        
        # Team B [x, y, vx, vy] for each player
        for i in range(self.num_players_per_team):
            idx = 4 + self.num_players_per_team * 4 + i * 4
            state[0, idx:idx+4] = torch.cat([self.team_b_pos[i], self.team_b_vel[i]])
        
        # Normalize state values to [0,1] or [-1,1] range
        # Ball position normalized to field dimensions
        state[0, 0] /= FIELD_WIDTH
        state[0, 1] /= FIELD_HEIGHT
        # Ball velocity normalized to max kick power
        state[0, 2:4] /= MAX_KICK_POWER
        
        # Player positions normalized to field dimensions
        for i in range(self.num_players_per_team * 2):
            idx = 4 + i * 4
            state[0, idx] /= FIELD_WIDTH
            state[0, idx+1] /= FIELD_HEIGHT
            # Player velocities normalized to max speed
            state[0, idx+2:idx+4] /= MAX_SPEED
        
        return state
    
    def render(self):
        """Render simulation using pygame"""
        if not self.use_rendering:
            return
            
        # Calculate scaling factors
        scale_x = SCREEN_WIDTH / FIELD_WIDTH
        scale_y = SCREEN_HEIGHT / FIELD_HEIGHT
        
        # Fill background (field)
        self.screen.fill(GREEN)
        
        # Draw field lines
        pygame.draw.rect(self.screen, WHITE, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), 2)
        pygame.draw.line(self.screen, WHITE, (SCREEN_WIDTH/2, 0), (SCREEN_WIDTH/2, SCREEN_HEIGHT), 2)
        pygame.draw.circle(self.screen, WHITE, (int(SCREEN_WIDTH/2), int(SCREEN_HEIGHT/2)), 
                          int(min(FIELD_WIDTH, FIELD_HEIGHT)/10 * scale_x), 2)
        
        # Goals
        goal_width = 7.32  # goal width in meters
        goal_y_start = (FIELD_HEIGHT - goal_width) / 2
        goal_y_end = goal_y_start + goal_width
        
        # Left goal
        pygame.draw.line(
            self.screen, 
            WHITE, 
            (0, goal_y_start * scale_y), 
            (0, goal_y_end * scale_y), 
            5
        )
        
        # Right goal
        pygame.draw.line(
            self.screen, 
            WHITE, 
            (SCREEN_WIDTH, goal_y_start * scale_y), 
            (SCREEN_WIDTH, goal_y_end * scale_y), 
            5
        )
        
        # Draw team A players (red)
        for i in range(self.num_players_per_team):
            pos = self.team_a_pos[i].cpu().numpy()
            pygame.draw.circle(self.screen, RED, 
                              (int(pos[0] * scale_x), int(pos[1] * scale_y)), 
                              5)
            label = self.font.render(f"A{i}", True, WHITE)
            self.screen.blit(label, (int(pos[0] * scale_x) - 8, int(pos[1] * scale_y) - 8))
        
        # Draw team B players (blue)
        for i in range(self.num_players_per_team):
            pos = self.team_b_pos[i].cpu().numpy()
            pygame.draw.circle(self.screen, BLUE, 
                              (int(pos[0] * scale_x), int(pos[1] * scale_y)), 
                              5)
            label = self.font.render(f"B{i}", True, WHITE)
            self.screen.blit(label, (int(pos[0] * scale_x) - 8, int(pos[1] * scale_y) - 8))
        
        # Draw ball
        ball_pos = self.ball_pos.cpu().numpy()
        pygame.draw.circle(self.screen, YELLOW, 
                          (int(ball_pos[0] * scale_x), int(ball_pos[1] * scale_y)), 
                          4)
        
        # Draw simulation info
        info_text = f"Time: {self.simulation_time:.1f}s | Score: A {self.score_a} - {self.score_b} B | Time factor: x{self.time_factor:.1f}"
        if self.ball_possession["team"]:
            info_text += f" | Ball: {self.ball_possession['team']}{self.ball_possession['player_id']}"
        info_label = self.font.render(info_text, True, BLACK, WHITE)
        self.screen.blit(info_label, (10, 10))
        
        # Display episode and step info
        episode_text = f"Episode: {self.current_episode} | Step: {self.current_step}"
        episode_label = self.font.render(episode_text, True, BLACK, WHITE)
        self.screen.blit(episode_label, (10, 30))
        
        pygame.display.flip()
        self.clock.tick(60)  # max 60 FPS
    
    def set_time_factor(self, factor):
        """Set simulation time factor"""
        self.time_factor = max(0.1, factor)  # Minimum 0.1 to avoid stopping
    
    def check_for_events(self):
        """Check pygame events and respond to them"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_UP:
                    self.set_time_factor(self.time_factor * 1.5)
                elif event.key == pygame.K_DOWN:
                    self.set_time_factor(self.time_factor / 1.5)
                elif event.key == pygame.K_SPACE:
                    # Reset ball position
                    self.ball_pos = torch.tensor([FIELD_WIDTH/2, FIELD_HEIGHT/2], 
                                               dtype=torch.float32, device=self.device)
                    self.ball_vel = torch.zeros(2, dtype=torch.float32, device=self.device)
        return True


def train_agents(num_episodes=500, max_steps_per_episode=600, batch_size=BATCH_SIZE,
                num_players=5, render_every=100):
    """Train two team agents against each other"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create simulation environment
    env = FootballSimulation(num_players_per_team=num_players, use_rendering=False)
    
    # Calculate state and action dimensions
    state_dim = 4 + 4 * num_players * 2  # Ball (4) + players (4 per player)
    action_dim_per_player = 6  # [dir_x, dir_y, kick, kick_power, kick_dir_x, kick_dir_y]
    
    # Create the networks
    policy_net_a = TeamAgent(num_players, state_dim, action_dim_per_player).to(device)
    target_net_a = TeamAgent(num_players, state_dim, action_dim_per_player).to(device)
    target_net_a.load_state_dict(policy_net_a.state_dict())
    
    policy_net_b = TeamAgent(num_players, state_dim, action_dim_per_player).to(device)
    target_net_b = TeamAgent(num_players, state_dim, action_dim_per_player).to(device)
    target_net_b.load_state_dict(policy_net_b.state_dict())
    
    # Setup optimizers
    optimizer_a = optim.Adam(policy_net_a.parameters(), lr=LR)
    optimizer_b = optim.Adam(policy_net_b.parameters(), lr=LR)
    
    memory_a = ReplayMemory(MEMORY_SIZE)
    memory_b = ReplayMemory(MEMORY_SIZE)

def select_action(policy_net, state, epsilon, device):
    policy_net.eval()
    with torch.no_grad():
        action = policy_net(state.to(device))
    policy_net.train()
    
    # Add noise for exploration
    noise = torch.randn_like(action) * epsilon
    action = action + noise
    # Optionally clamp or normalize parts of action here
    return action.clamp(-1, 1)  # or appropriate limits

def soft_update(target_net, policy_net, tau):
    for target_param, param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


epsilon = EPSILON_START
epsilon_decay_step = (EPSILON_START - EPSILON_END) / EPSILON_DECAY

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward_a = 0
    total_reward_b = 0
    
    for step in range(max_steps_per_episode):
        # Select actions with exploration noise
        action_a = select_action(policy_net_a, state, epsilon, device)
        action_b = select_action(policy_net_b, state, epsilon, device)
        
        # Take a step in environment
        next_state, (reward_a, reward_b), done, info = env.step(action_a, action_b)
        
        # Store transitions in replay memory
        memory_a.push(state, action_a, reward_a, next_state, done)
        memory_b.push(state, action_b, reward_b, next_state, done)
        
        state = next_state
        total_reward_a += reward_a
        total_reward_b += reward_b
        
        # Sample minibatch and optimize policy networks
        if len(memory_a) >= batch_size:
            optimize_model(policy_net_a, target_net_a, optimizer_a, memory_a, batch_size, device)
        if len(memory_b) >= batch_size:
            optimize_model(policy_net_b, target_net_b, optimizer_b, memory_b, batch_size, device)
        
        # Soft update target networks
        soft_update(target_net_a, policy_net_a, TAU)
        soft_update(target_net_b, policy_net_b, TAU)
        
        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon - epsilon_decay_step)
        
        if done:
            break
    
    if (episode + 1) % render_every == 0:
        env.use_rendering = True
        env.render()
        env.use_rendering = False
    
    print(f"Episode {episode+1}, Reward A: {total_reward_a:.2f}, Reward B: {total_reward_b:.2f}, Epsilon: {epsilon:.3f}")


def optimize_model(policy_net, target_net, optimizer, memory, batch_size, device):
    experiences = memory.sample(batch_size)
    batch = Experience(*zip(*experiences))
    
    states = torch.cat(batch.state).to(device)
    actions = torch.cat(batch.action).to(device)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.cat(batch.next_state).to(device)
    dones = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)
    
    # Compute current Q values from policy net
    pred_actions = policy_net(states)
    
    # Compute target Q values from target net
    with torch.no_grad():
        target_actions = target_net(next_states)
    
    # Compute loss (e.g. MSE between predicted and target)
    # Since this is a continuous action output, you may want to use e.g. DDPG or actor-critic method
    # Here is a placeholder for a simple supervised loss:
    loss = F.mse_loss(pred_actions, actions)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
